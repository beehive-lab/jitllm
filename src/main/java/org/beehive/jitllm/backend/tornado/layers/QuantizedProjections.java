package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The single-token projections of a layer whose weights stay in the file's own representations:
 * each weight's kernel chosen from its type and from whether the packed-integer path runs, with the
 * grid that kernel needs.
 *
 * <p>The decode counterpart of {@link FP16Projections}. A layer that holds one representation keeps
 * its family's fused kernels; a layer that holds several, as a Q4_K_M file's do, projects through
 * here one weight at a time. Every type the device stores natively has a floating-point kernel and
 * a packed one: Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K.
 *
 * <p><b>Packed or not is one answer for the whole layer</b> ({@link
 * PackedActivationTasks#enabled}), because the RMS norm that feeds the projections quantizes the
 * activation or does not. Where it does, it still writes {@code wrapXb}, so a floating-point kernel
 * could read it too; none has to.
 *
 * <p>The gate and up projections run as one fused task where both are the same type and a fused
 * kernel exists (Q8_0, Q4_0, Q4_K; Q5_K in floating point), and otherwise as two projections into
 * {@code wrapHb} and {@code wrapHb2} and an elementwise SwiGLU, so a layer that splits them needs
 * {@code wrapHb2} bound.
 */
// @formatter:on
public final class QuantizedProjections {

    private final KernelContext context;
    private final TornadoWorkspace ws;
    private final int floatLocal;
    private final int packedLocal;

    /**
     * @param floatLocal the lanes per output row of the floating-point kernels
     * @param packedLocal the lanes per output row of the packed kernels, a multiple of 32
     */
    public QuantizedProjections(
            KernelContext context, TornadoWorkspace ws, int floatLocal, int packedLocal) {
        this.context = context;
        this.ws = ws;
        this.floatLocal = floatLocal;
        this.packedLocal = packedLocal;
    }

    /**
     * Floating-point projections only, for a layer whose graphs bind no quantized activation: the
     * packed path never runs, whatever the device supports.
     */
    public QuantizedProjections(KernelContext context, TornadoWorkspace ws, int floatLocal) {
        this(context, ws, floatLocal, 0);
    }

    /** Whether the projections read the activation quantized to eight bits. */
    public boolean packed() {
        return packedLocal > 0 && PackedActivationTasks.enabled();
    }

    /**
     * {@code wrapXb = rms * temp[0] * wrapX}, quantized as well where the projections are packed.
     */
    public void rmsApply(TaskGraph graph, String name, FloatArray rmsWeights, FloatArray temp) {
        if (packed()) {
            PackedActivationTasks.rmsApplyAndQuantize(graph, name, context, ws, rmsWeights, temp);
        } else {
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    ws.wrapXb,
                    ws.wrapX,
                    rmsWeights,
                    temp);
        }
    }

    /** The grid of {@link #rmsApply} over {@code dim} activations. */
    public WorkerGrid rmsApplyWorker(int dim) {
        return packed()
                ? PackedActivationTasks.quantizeWorker(dim)
                : WorkerGridFactory.createRmsNormWorker(dim, 256);
    }

    /** {@code out = w·wrapXb}. */
    public void matvec(
            TaskGraph graph, String name, FloatArray out, TornadoTensor w, int n, int d) {
        DataType type = w.dataType();
        if (packed() && type == DataType.Q8_0) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    out,
                    bytes(w),
                    n,
                    d,
                    packedLocal);
        } else if (packed()) {
            graph.task(
                    name,
                    switch (type) {
                        case Q4_0 -> TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A;
                        case Q4_1 -> TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1DP4A;
                        case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KDP4A;
                        case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_KDP4A;
                        case Q6_K -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A;
                        default -> throw unsupported(w);
                    },
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapXbSums,
                    out,
                    bytes(w),
                    n,
                    d,
                    packedLocal);
        } else {
            floatMatvec(graph, name, ws.wrapXb, out, w, n, d);
        }
    }

    /**
     * {@code out = w·x} in floating point, for an activation other than {@code wrapXb}, which the
     * packed path holds no quantization of.
     */
    public void floatMatvec(
            TaskGraph graph,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        graph.task(
                name,
                switch (w.dataType()) {
                    case Q8_0 -> TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte;
                    case Q4_0 -> TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0;
                    case Q4_1 -> TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1;
                    case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K;
                    case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K;
                    case Q6_K -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K;
                    default -> throw unsupported(w);
                },
                context,
                x,
                out,
                bytes(w),
                n,
                d,
                floatLocal);
    }

    /**
     * {@code wrapX += w·x}. Where the projections are packed, {@code x} is quantized first by
     * {@code quantizeName}.
     */
    public void residualMatvec(
            TaskGraph graph,
            String quantizeName,
            String name,
            FloatArray x,
            TornadoTensor w,
            int n,
            int d) {
        DataType type = w.dataType();
        if (packed()) {
            PackedActivationTasks.quantize(graph, quantizeName, context, ws, x);
        }
        if (packed() && type == DataType.Q8_0) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapX,
                    bytes(w),
                    n,
                    d,
                    packedLocal);
        } else if (packed()) {
            graph.task(
                    name,
                    switch (type) {
                        case Q4_0 ->
                                TransformerComputeKernelsQ4_0
                                        ::matrixVectorGenericWithResidualQ4_0DP4A;
                        case Q4_1 ->
                                TransformerComputeKernelsQ4_1
                                        ::matrixVectorGenericWithResidualQ4_1DP4A;
                        case Q4_K ->
                                TransformerComputeKernelsQ4_K
                                        ::matrixVectorGenericWithResidualQ4_KDP4A;
                        case Q5_K ->
                                TransformerComputeKernelsQ5_K
                                        ::matrixVectorGenericWithResidualQ5_KDP4A;
                        case Q6_K ->
                                TransformerComputeKernelsQ6_K
                                        ::matrixVectorGenericWithResidualQ6_KDP4A;
                        default -> throw unsupported(w);
                    },
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapXbSums,
                    ws.wrapX,
                    bytes(w),
                    n,
                    d,
                    packedLocal);
        } else {
            graph.task(
                    name,
                    switch (type) {
                        case Q8_0 ->
                                TransformerComputeKernelsLayered
                                        ::matrixVectorGenericWithResidualQ8_0Byte;
                        case Q4_0 ->
                                TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0;
                        case Q4_1 ->
                                TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1;
                        case Q4_K ->
                                TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K;
                        case Q5_K ->
                                TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_K;
                        case Q6_K ->
                                TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K;
                        default -> throw unsupported(w);
                    },
                    context,
                    x,
                    ws.wrapX,
                    bytes(w),
                    n,
                    d,
                    floatLocal);
        }
    }

    /**
     * Whether {@link #gateUp} runs {@code w1} and {@code w3} as one fused task: both the same type,
     * with a fused kernel on the path that runs (Q8_0, Q4_0 and Q4_K packed; those and Q5_K in
     * floating point).
     */
    public boolean fusedGateUp(TornadoTensor w1, TornadoTensor w3) {
        DataType type = w1.dataType();
        return type == w3.dataType()
                && (type == DataType.Q8_0
                        || type == DataType.Q4_0
                        || type == DataType.Q4_K
                        || (type == DataType.Q5_K && !packed()));
    }

    /**
     * {@code wrapHb = SiLU(w1·wrapXb) * (w3·wrapXb)}: task {@code name} where {@link #fusedGateUp}
     * holds; otherwise tasks {@code name + "_gate"} and {@code name + "_up"} into {@code wrapHb}
     * and {@code wrapHb2}, and {@code name} for the SwiGLU.
     */
    public void gateUp(
            TaskGraph graph, String name, TornadoTensor w1, TornadoTensor w3, int n, int d) {
        if (!fusedGateUp(w1, w3)) {
            matvec(graph, name + "_gate", ws.wrapHb, w1, n, d);
            matvec(graph, name + "_up", ws.wrapHb2, w3, n, d);
            graph.task(
                    name,
                    TransformerComputeKernelsQ6_K::siluAndMultiply,
                    context,
                    ws.wrapHb,
                    ws.wrapHb2,
                    d);
            return;
        }
        DataType type = w1.dataType();
        if (packed() && type == DataType.Q8_0) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapHb,
                    bytes(w1),
                    bytes(w3),
                    n,
                    d,
                    packedLocal);
        } else if (packed()) {
            graph.task(
                    name,
                    type == DataType.Q4_0
                            ? TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A
                            : TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_KDP4A,
                    context,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapXbSums,
                    ws.wrapHb,
                    bytes(w1),
                    bytes(w3),
                    n,
                    d,
                    packedLocal);
        } else {
            graph.task(
                    name,
                    switch (type) {
                        case Q8_0 ->
                                TransformerComputeKernelsLayered
                                        ::fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte;
                        case Q4_0 -> TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0;
                        case Q5_K -> TransformerComputeKernelsQ5_K::fusedFFNGateUpSiLUQ5_K;
                        default -> TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_K;
                    },
                    context,
                    ws.wrapXb,
                    ws.wrapHb,
                    bytes(w1),
                    bytes(w3),
                    n,
                    d,
                    floatLocal);
        }
    }

    /** The grid of a projection to {@code rows} outputs: one workgroup per row. */
    public WorkerGrid worker(int rows) {
        int local = packed() ? packedLocal : floatLocal;
        return WorkerGridFactory.genericWorker(rows * local, local);
    }

    /** Registers the grids of the tasks {@link #gateUp} added under {@code prefix + name}. */
    public void addGateUpGrids(
            GridScheduler scheduler,
            String prefix,
            String name,
            TornadoTensor w1,
            TornadoTensor w3,
            int d) {
        if (fusedGateUp(w1, w3)) {
            scheduler.addWorkerGrid(prefix + name, worker(d));
            return;
        }
        scheduler.addWorkerGrid(prefix + name + "_gate", worker(d));
        scheduler.addWorkerGrid(prefix + name + "_up", worker(d));
        scheduler.addWorkerGrid(
                prefix + name, WorkerGridFactory.genericWorker((d + 255) / 256 * 256, 256));
    }

    /**
     * {@code w}'s blocks as the file stores them. A weight the plan prepacked into tiles has no
     * kernel here, and is refused by name rather than read as the format it no longer is.
     */
    private ByteArray bytes(TornadoTensor w) {
        if (ws.prepacking != null && ws.prepacking.prepacked(w)) {
            throw WeightPrepacking.noPrepackedKernel(
                    "single-token " + w.dataType() + " projection");
        }
        return w.asByteArray();
    }

    private static IllegalStateException unsupported(TornadoTensor w) {
        return new IllegalStateException(
                "no single-token projection kernel for a " + w.dataType() + " weight");
    }
}
