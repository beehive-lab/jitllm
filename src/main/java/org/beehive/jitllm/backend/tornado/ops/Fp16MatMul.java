package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.SplitKGemm;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The batched prefill on a tensor-core GPU over F16 weights, which need no decode. The JIT GEMMs
 * read them where they are: q, k and v in one launch ({@code gemmMMAQKV}), gate and up likewise
 * ({@code gemmMMAGateUp}), every other projection through {@link SplitKGemm}. cuBLAS multiplies one
 * matrix per call, so under it q|k|v and gate|up are copied side by side first ({@link
 * FP16Projections}), and the rest go to cuBLAS as they are. The subclass picks the GEMM.
 *
 * <p>Example: Llama-3.2-1B F16 on the JIT GEMMs uploads its weights once and copies none; on cuBLAS
 * it keeps a stacked copy of q|k|v and of gate|up.
 */
public abstract class Fp16MatMul implements Implementation {

    /** Whether the GEMMs are cuBLAS's ({@code true}) or the JIT ones ({@code false}). */
    protected abstract boolean nativeGemm();

    @Override
    public List<Fusion> fusions() {
        return TensorCoreSteps.FUSIONS;
    }

    /** Whether every projection of {@code step} is F16, so that it needs no decode. */
    protected static boolean f16(Step step) {
        for (MatMul m : step.matMuls()) {
            if (m.dataType() != DataType.F16) {
                return false;
            }
        }
        return true;
    }

    /** Whether {@code target} has tensor cores and every projection of {@code step} is F16. */
    protected static boolean tensorCoresOverF16(Step step, Target target) {
        return target.supports(DeviceCapability.TENSOR_CORE_MMA) && f16(step);
    }

    @Override
    public BatchPrefillWorkspace needs(List<Step> steps, Target target) {
        Sizes s = Sizes.of(steps);
        long fixed = SplitKGemm.bytes(s.rows, s.widestSplit, nativeGemm());
        if (!nativeGemm()) {
            return BatchPrefillWorkspace.fixed(fixed);
        }
        // As the stacked copies have always been charged: every projection's weights, although
        // only q|k|v and gate|up are copied.
        return new BatchPrefillWorkspace(
                s.weights,
                fixed
                        + FP16Projections.residentBytes(s.weights)
                        + FP16Projections.workspaceBytes(
                                true, s.widestStacked, s.rows, s.widestSplit, s.stacked, true),
                fixed
                        + FP16Projections.workspaceBytes(
                                false, s.widestStacked, s.rows, s.widestSplit, s.stacked, true));
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        Sizes s = Sizes.of(steps);
        SplitKGemm splitK = new SplitKGemm(context, s.rows, s.widestSplit, nativeGemm());
        FP16Projections stacked =
                nativeGemm()
                        ? new FP16Projections(
                                context,
                                prepacking,
                                s.widestStacked,
                                s.rows,
                                s.widestSplit,
                                FP16Projections.resident(target.policy(), s.weights),
                                s.stacked,
                                true)
                        : null;
        return new Tasks(context, splitK, stacked);
    }

    /** The projections: in place on the JIT GEMMs, q|k|v and gate|up stacked for cuBLAS. */
    private static final class Tasks extends TensorCoreSteps {

        private final SplitKGemm splitK;
        private final FP16Projections stacked;

        Tasks(KernelContext context, SplitKGemm splitK, FP16Projections stacked) {
            super(context);
            this.splitK = splitK;
            this.stacked = stacked;
        }

        @Override
        public Object[] buffers() {
            if (stacked == null) {
                return new Object[] {splitK.partial()};
            }
            Object[] copies = stacked.buffers();
            Object[] all = new Object[copies.length + 1];
            all[0] = splitK.partial();
            System.arraycopy(copies, 0, all, 1, copies.length);
            return all;
        }

        @Override
        protected MatMul project(
                TaskGraph graph,
                String task,
                Step step,
                Operands operands,
                WeightBindings bindings) {
            List<MatMul> matMuls = step.matMuls();
            MatMul first = matMuls.get(0);
            HalfFloatArray a = (HalfFloatArray) operands.activation(first.input());
            FloatArray c = (FloatArray) operands.activation(first.output());
            TornadoTensor[] w = weights(step, operands);
            if (matMuls.size() == 1) {
                splitK.gemm(
                        graph,
                        task,
                        a,
                        w[0].asHalfFloatArray(),
                        c,
                        first.batchSize(),
                        first.rows(),
                        first.columns());
            } else if (stacked != null) {
                stacked.project(graph, task, a, c, first.batchSize(), first.columns(), w);
            } else if (matMuls.size() == 3) {
                graph.task(
                        task,
                        TransformerBatchPrefillKernels::gemmMMAQKV,
                        context,
                        a,
                        w[0].asHalfFloatArray(),
                        w[1].asHalfFloatArray(),
                        w[2].asHalfFloatArray(),
                        c,
                        first.batchSize(),
                        first.rows(),
                        matMuls.get(1).rows(),
                        first.columns());
            } else {
                graph.task(
                        task,
                        TransformerBatchPrefillKernels::gemmMMAGateUp,
                        context,
                        a,
                        w[0].asHalfFloatArray(),
                        w[1].asHalfFloatArray(),
                        c,
                        first.batchSize(),
                        first.rows(),
                        first.columns());
            }
            return first;
        }

        @Override
        protected void addProjectionGrids(GridScheduler scheduler) {
            splitK.addGrids(scheduler);
            if (stacked != null) {
                stacked.addGrids(scheduler);
            }
        }
    }

    /**
     * The sizes the GEMMs need: every projection's weights, the rows, the widest output a
     * one-matrix GEMM writes (sizing the split GEMM's partial products), and of the q|k|v and
     * gate|up projections, which cuBLAS reads stacked, the largest and how many there are.
     */
    private record Sizes(long weights, int rows, int widestSplit, long widestStacked, int stacked) {

        static Sizes of(List<Step> steps) {
            long weights = 0;
            int rows = 0;
            int widestSplit = 0;
            long widestStacked = 0;
            int stacked = 0;
            for (Step step : steps) {
                List<MatMul> matMuls = step.matMuls();
                if (matMuls.isEmpty()) {
                    continue;
                }
                weights += step.weights();
                rows = matMuls.get(0).batchSize();
                if (TensorCoreSteps.stacks(matMuls.get(0))) {
                    widestStacked = Math.max(widestStacked, step.weights());
                    stacked++;
                } else {
                    widestSplit = Math.max(widestSplit, step.outputWidth());
                }
            }
            return new Sizes(weights, rows, widestSplit, widestStacked, stacked);
        }
    }
}
