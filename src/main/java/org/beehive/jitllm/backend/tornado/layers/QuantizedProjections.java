package org.beehive.jitllm.backend.tornado.layers;

import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsIQ4_XS;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
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
 * a packed one: Q8_0, Q4_0, Q4_1, Q5_0, Q4_K, Q5_K and Q6_K.
 *
 * <p><b>Packed or not is one answer for the whole layer</b> ({@link
 * PackedActivationTasks#enabled}), because the RMS norm that feeds the projections quantizes the
 * activation or does not. Where it does, it still writes {@code wrapXb}, so a floating-point kernel
 * could read it too; none has to.
 *
 * <p>The gate and up projections run as one fused task where both are the same type and a fused
 * kernel exists (Q8_0, Q4_0, Q4_K and Q5_K), and otherwise as two projections into {@code wrapHb}
 * and {@code wrapHb2} and an elementwise SwiGLU, so a layer that splits them needs {@code wrapHb2}
 * bound.
 */
// @formatter:on
public final class QuantizedProjections {

    private final KernelContext context;
    private final TornadoWorkspace ws;
    private final int floatLocal;
    private final int packedLocal;

    /** The grids of the SIMD-group tasks, by {@code graph.task} name; see {@link #addGrids}. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

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

    /**
     * Whether {@code w} runs the SIMD-group Q4_K kernels, four output rows per SIMD group: a Q4_K
     * weight on the floating-point path, on a device with a verified 32-lane shuffle (Metal).
     */
    private boolean simdgroup(TornadoTensor w) {
        return !packed()
                && w.dataType() == DataType.Q4_K
                && SchedulerDetectionService.isSubgroupShuffle32Supported();
    }

    /** Records the grid of the SIMD-group task {@code name} of {@code graph}, over {@code rows}. */
    private void simdgroupGrid(TaskGraph graph, String name, int rows) {
        int groups =
                (rows + TransformerComputeKernelsQ4_K.SIMDGROUP_GROUP_ROWS - 1)
                        / TransformerComputeKernelsQ4_K.SIMDGROUP_GROUP_ROWS;
        int threads = 32 * TransformerComputeKernelsQ4_K.SIMDGROUPS;
        grids.put(
                graph.getTaskGraphName() + "." + name,
                WorkerGridFactory.genericWorker(groups * threads, threads));
    }

    /**
     * Sets the grids of the SIMD-group tasks, over the one {@link #worker} gave each: call it last
     * when filling a grid scheduler. Example: {@code layer_3.ffn_gate_up} over 14336 rows runs 1792
     * workgroups of 64 threads, not 14336 of 32.
     */
    public void addGrids(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
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
        if (type == DataType.F16) {
            // F16 has no packed kernel; the packed RMS apply writes wrapXb as well. The lanes per
            // row are the grid's (worker).
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    ws.wrapXb,
                    out,
                    w.asHalfFloatArray(),
                    n,
                    d,
                    packed() ? packedLocal : floatLocal);
        } else if (packed() && type == DataType.Q8_0) {
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
                        case Q5_0 -> TransformerComputeKernelsQ5_0::matrixVectorGenericQ5_0DP4A;
                        case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KDP4A;
                        case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_KDP4A;
                        case Q6_K -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A;
                        case IQ4_XS ->
                                TransformerComputeKernelsIQ4_XS::matrixVectorGenericIQ4_XSDP4A;
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
        if (w.dataType() == DataType.F16) {
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    x,
                    out,
                    w.asHalfFloatArray(),
                    n,
                    d,
                    floatLocal);
            return;
        }
        if (simdgroup(w)) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_K::matrixVectorQ4_KSimdgroup,
                    context,
                    x,
                    out,
                    bytes(w),
                    n,
                    d);
            simdgroupGrid(graph, name, d);
            return;
        }
        graph.task(
                name,
                switch (w.dataType()) {
                    case Q8_0 -> TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte;
                    case Q4_0 -> TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0;
                    case Q4_1 -> TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1;
                    case Q5_0 -> TransformerComputeKernelsQ5_0::matrixVectorGenericQ5_0;
                    case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K;
                    case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K;
                    case Q6_K -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K;
                    case IQ4_XS -> TransformerComputeKernelsIQ4_XS::matrixVectorGenericIQ4_XS;
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
        if (type == DataType.F16) {
            // F16 has no packed kernel, so x is read as it is and not quantized.
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    x,
                    ws.wrapX,
                    w.asHalfFloatArray(),
                    n,
                    d,
                    packed() ? packedLocal : floatLocal);
            return;
        }
        if (simdgroup(w)) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_K::matrixVectorWithResidualQ4_KSimdgroup,
                    context,
                    x,
                    ws.wrapX,
                    bytes(w),
                    n,
                    d);
            simdgroupGrid(graph, name, d);
            return;
        }
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
                        case Q5_0 ->
                                TransformerComputeKernelsQ5_0
                                        ::matrixVectorGenericWithResidualQ5_0DP4A;
                        case Q4_K ->
                                TransformerComputeKernelsQ4_K
                                        ::matrixVectorGenericWithResidualQ4_KDP4A;
                        case Q5_K ->
                                TransformerComputeKernelsQ5_K
                                        ::matrixVectorGenericWithResidualQ5_KDP4A;
                        case Q6_K ->
                                TransformerComputeKernelsQ6_K
                                        ::matrixVectorGenericWithResidualQ6_KDP4A;
                        case IQ4_XS ->
                                TransformerComputeKernelsIQ4_XS
                                        ::matrixVectorGenericWithResidualIQ4_XSDP4A;
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
                        case Q5_0 ->
                                TransformerComputeKernelsQ5_0::matrixVectorGenericWithResidualQ5_0;
                        case Q4_K ->
                                TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K;
                        case Q5_K ->
                                TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_K;
                        case Q6_K ->
                                TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K;
                        case IQ4_XS ->
                                TransformerComputeKernelsIQ4_XS
                                        ::matrixVectorGenericWithResidualIQ4_XS;
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
     * with a fused kernel on the path that runs (Q8_0, Q4_0, Q4_K and Q5_K, packed and in floating
     * point).
     */
    public boolean fusedGateUp(TornadoTensor w1, TornadoTensor w3) {
        DataType type = w1.dataType();
        return type == w3.dataType()
                && (type == DataType.Q8_0
                        || type == DataType.Q4_0
                        || type == DataType.Q4_K
                        || type == DataType.Q5_K);
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
        if (simdgroup(w1)) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_KSimdgroup,
                    context,
                    ws.wrapXb,
                    ws.wrapHb,
                    bytes(w1),
                    bytes(w3),
                    n,
                    d);
            simdgroupGrid(graph, name, d);
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
                    switch (type) {
                        case Q4_0 -> TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A;
                        case Q5_K -> TransformerComputeKernelsQ5_K::fusedFFNGateUpSiLUQ5_KDP4A;
                        default -> TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_KDP4A;
                    },
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
