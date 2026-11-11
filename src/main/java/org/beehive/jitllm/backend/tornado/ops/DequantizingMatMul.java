package org.beehive.jitllm.backend.tornado.ops;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.layers.SplitKGemm;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The batched prefill on a tensor-core GPU for quantized weights decoded one projection at a time:
 * each weight of a projection is decoded into one contiguous FP16 scratch, end to end, then one
 * GEMM ({@link SplitKGemm}) multiplies the scratch. The weights stay quantized between chunks, at
 * the cost of a scratch the size of the widest projection and a decode pass per projection per
 * chunk. The subclass picks the GEMM: cuBLAS or the JIT one.
 *
 * <p>Example: Llama Q4_0 and Devstral Q4_K_M on CUDA. Q, K and V are decoded contiguously ({@code
 * qDecode}, {@code kDecode}, {@code vDecode}), then {@code qkvProj} writes the packed rows.
 */
public abstract class DequantizingMatMul implements Implementation {

    /** The types decoded here: Q4_0 (from the file's blocks or prepacked), Q4_1, Q4_K, Q6_K. */
    private static final Set<DataType> DECODED =
            Set.of(DataType.Q4_0, DataType.Q4_1, DataType.Q4_K, DataType.Q6_K);

    /** Whether the GEMMs are cuBLAS's ({@code true}) or the JIT one ({@code false}). */
    protected abstract boolean nativeGemm();

    @Override
    public List<Fusion> fusions() {
        return TensorCoreSteps.FUSIONS;
    }

    /**
     * Whether {@code target} has tensor cores and every projection of {@code step} decodes here.
     */
    protected static boolean tensorCoresOverDecoded(Step step, Target target) {
        if (!target.supports(DeviceCapability.TENSOR_CORE_MMA)) {
            return false;
        }
        for (MatMul m : step.matMuls()) {
            if (!DECODED.contains(m.dataType())) {
                return false;
            }
        }
        return true;
    }

    @Override
    public BatchPrefillWorkspace needs(List<Step> steps, Target target) {
        Sizes s = Sizes.of(steps);
        return BatchPrefillWorkspace.withoutResidencyChoice(
                SplitKGemm.bytes(s.rows, s.widestSplit, nativeGemm()) + 2L * s.widest);
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        Sizes s = Sizes.of(steps);
        return new Tasks(
                context,
                prepacking,
                new SplitKGemm(context, s.rows, s.widestSplit, nativeGemm()),
                new HalfFloatArray(Math.toIntExact(s.widest)));
    }

    /** The primary's partial products and scratch, split as the primary splits. */
    @Override
    public Instance narrowed(
            Instance primary,
            List<Step> steps,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        Sizes s = Sizes.of(steps);
        Tasks wider = (Tasks) primary;
        return new Tasks(
                context,
                prepacking,
                wider.splitK.narrowed(s.rows, s.widestSplit),
                wider.contiguousScratch);
    }

    /** The task that decodes a projection's weight, by its role: {@code qDecode} for the query. */
    private static String decodeTask(MatMul projection) {
        return switch (projection.weight().role()) {
            case ATTENTION_QUERY -> "qDecode";
            case ATTENTION_KEY -> "kDecode";
            case ATTENTION_VALUE -> "vDecode";
            case ATTENTION_OUTPUT -> "woDecode";
            case FFN_GATE -> "gateDecode";
            case FFN_UP -> "upDecode";
            case FFN_DOWN -> "downDecode";
            default ->
                    throw new IllegalArgumentException(
                            "no decode task for " + projection.weight().role());
        };
    }

    /** The decodes into the scratch, then the split-K GEMM over it. */
    private static final class Tasks extends TensorCoreSteps {

        private final WeightPrepacking prepacking;
        private final SplitKGemm splitK;
        private final HalfFloatArray contiguousScratch;

        /** The decode tasks added, by graph-qualified name, with their element counts. */
        private final Map<String, Integer> decodes = new LinkedHashMap<>();

        Tasks(
                KernelContext context,
                WeightPrepacking prepacking,
                SplitKGemm splitK,
                HalfFloatArray contiguousScratch) {
            super(context);
            this.prepacking = prepacking;
            this.splitK = splitK;
            this.contiguousScratch = contiguousScratch;
        }

        @Override
        public Object[] buffers() {
            return new Object[] {splitK.partial(), contiguousScratch};
        }

        @Override
        protected MatMul project(
                TaskGraph graph,
                String task,
                Step step,
                Operands operands,
                WeightBindings bindings) {
            List<MatMul> matMuls = step.matMuls();
            TornadoTensor[] w = weights(step, operands);
            int offset = 0;
            for (int i = 0; i < w.length; i++) {
                int n = matMuls.get(i).rows() * matMuls.get(i).columns();
                decode(graph, decodeTask(matMuls.get(i)), w[i], offset, n);
                offset += n;
            }
            MatMul first = matMuls.get(0);
            splitK.gemm(
                    graph,
                    task,
                    (HalfFloatArray) operands.activation(first.input()),
                    contiguousScratch,
                    (FloatArray) operands.activation(first.output()),
                    first.batchSize(),
                    step.outputWidth(),
                    first.columns());
            return first;
        }

        /** Adds the task {@code task} that decodes {@code n} values of {@code w} at {@code at}. */
        private void decode(TaskGraph graph, String task, TornadoTensor w, int at, int n) {
            decodes.put(graph.getTaskGraphName() + "." + task, n);
            if (prepacking.prepacked(w)) {
                graph.task(
                        task,
                        TransformerComputeKernelsQ4_0::dequantizeQ4_0PackedToFP16,
                        context,
                        w.asByteArray(),
                        contiguousScratch,
                        at,
                        n,
                        Math.toIntExact(w.shape().dimension(0)));
                return;
            }
            switch (w.dataType()) {
                case Q4_0 ->
                        graph.task(
                                task,
                                TransformerComputeKernelsQ4_0::dequantizeQ4_0ToFP16,
                                context,
                                w.asByteArray(),
                                contiguousScratch,
                                at,
                                n);
                case Q4_1 ->
                        graph.task(
                                task,
                                TransformerComputeKernelsQ4_0::dequantizeQ4_1ToFP16,
                                context,
                                w.asByteArray(),
                                contiguousScratch,
                                at,
                                n);
                case Q4_K ->
                        graph.task(
                                task,
                                TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16,
                                context,
                                w.asByteArray(),
                                contiguousScratch,
                                at,
                                n);
                case Q6_K ->
                        graph.task(
                                task,
                                TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                                context,
                                w.asByteArray(),
                                contiguousScratch,
                                at,
                                n);
                default ->
                        throw new UnsupportedOperationException(
                                "no decoder for "
                                        + w.dataType()
                                        + " ("
                                        + task
                                        + "); the projections must be Q4_0, Q4_1, Q4_K or Q6_K");
            }
        }

        @Override
        protected void addProjectionGrids(GridScheduler scheduler) {
            decodes.forEach((task, n) -> scheduler.addWorkerGrid(task, elementwiseAny(n)));
            splitK.addGrids(scheduler);
        }

        /** One thread per value, 256 per workgroup, rounded up to whole workgroups. */
        private static WorkerGrid elementwiseAny(int n) {
            return WorkerGridFactory.genericWorker(((n + 255) / 256) * 256, 256);
        }
    }

    /**
     * The rows, the widest one-matrix output (the split GEMM's buffer) and the widest projection.
     */
    private record Sizes(int rows, int widestSplit, long widest) {

        static Sizes of(List<Step> steps) {
            int rows = 0;
            int widestSplit = 0;
            long widest = 0;
            for (Step step : steps) {
                List<MatMul> matMuls = step.matMuls();
                if (matMuls.isEmpty()) {
                    continue;
                }
                rows = matMuls.get(0).batchSize();
                widest = Math.max(widest, step.weights());
                if (matMuls.size() == 1) {
                    widestSplit = Math.max(widestSplit, step.outputWidth());
                }
            }
            return new Sizes(rows, widestSplit, widest);
        }
    }
}
