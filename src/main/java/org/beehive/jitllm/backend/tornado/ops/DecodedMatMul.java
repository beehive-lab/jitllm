package org.beehive.jitllm.backend.tornado.ops;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.program.op.MatMul;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The batched prefill on a tensor-core GPU. The projections run in two steps: decode each weight to
 * FP16 (whatever its type in the file, Q8_0, Q4_K, ...), then one FP16 GEMM that writes FP32. The
 * subclass picks the GEMM: the JIT GEMM or cuBLAS. The decode and the GEMM are done by the existing
 * {@link FP16Projections}.
 *
 * <p>Its fusion table also runs the steps around the GEMMs ({@link TensorCoreSteps}): the RMS
 * norms, SwiGLU and the residual adds. The residual add after attention is merged into the next RMS
 * norm.
 */
public abstract class DecodedMatMul implements Implementation {

    /** Whether the GEMM is cuBLAS ({@code true}) or the JIT GEMM ({@code false}). */
    protected abstract boolean nativeGemm();

    @Override
    public List<Fusion> fusions() {
        return TensorCoreSteps.FUSIONS;
    }

    @Override
    public BatchPrefillWorkspace needs(List<Step> steps, Target target) {
        Sizes s = Sizes.of(steps);
        return new BatchPrefillWorkspace(
                s.decoded,
                FP16Projections.residentBytes(s.decoded)
                        + FP16Projections.workspaceBytes(
                                true, s.widest, s.rows, s.widestSplit, s.calls, nativeGemm()),
                FP16Projections.workspaceBytes(
                        false, s.widest, s.rows, s.widestSplit, s.calls, nativeGemm()));
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        Sizes s = Sizes.of(steps);
        FP16Projections projections =
                new FP16Projections(
                        context,
                        prepacking,
                        s.widest,
                        s.rows,
                        s.widestSplit,
                        FP16Projections.resident(target.policy(), s.decoded),
                        s.calls,
                        nativeGemm());
        return new Tasks(context, projections);
    }

    /** The primary's decoded copies and scratch, split as the primary splits. */
    @Override
    public Instance narrowed(
            Instance primary,
            List<Step> steps,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        Sizes s = Sizes.of(steps);
        return new Tasks(context, ((Tasks) primary).projections.narrowed(s.rows, s.widestSplit));
    }

    /** The projections through {@link FP16Projections}: each weight decoded, then one GEMM. */
    private static final class Tasks extends TensorCoreSteps {

        private final FP16Projections projections;

        Tasks(KernelContext context, FP16Projections projections) {
            super(context);
            this.projections = projections;
        }

        @Override
        public Object[] buffers() {
            return projections.buffers();
        }

        @Override
        protected MatMul project(
                TaskGraph graph,
                String task,
                Step step,
                Operands operands,
                WeightBindings bindings) {
            MatMul first = step.matMuls().get(0);
            projections.project(
                    graph,
                    task,
                    (HalfFloatArray) operands.activation(first.input()),
                    (FloatArray) operands.activation(first.output()),
                    first.batchSize(),
                    first.columns(),
                    weights(step, operands));
            return first;
        }

        @Override
        protected void addProjectionGrids(GridScheduler scheduler) {
            projections.addGrids(scheduler);
        }
    }

    /**
     * The sizes {@link FP16Projections} needs: all weight values, the largest GEMM, the number of
     * rows, the smallest output width (used to size the split GEMM's buffer) and the number of
     * GEMMs.
     */
    private record Sizes(long decoded, long widest, int rows, int widestSplit, int calls) {

        static Sizes of(List<Step> steps) {
            long decoded = 0;
            long widest = 0;
            int rows = 0;
            int widestSplit = Integer.MAX_VALUE;
            int calls = 0;
            List<Step> gemms = new ArrayList<>();
            for (Step step : steps) {
                if (!step.matMuls().isEmpty()) {
                    gemms.add(step);
                }
            }
            for (Step step : gemms) {
                decoded += step.weights();
                widest = Math.max(widest, step.weights());
                widestSplit = Math.min(widestSplit, step.outputWidth());
                rows = step.matMuls().get(0).batchSize();
                calls++;
            }
            return new Sizes(decoded, widest, rows, widestSplit, calls);
        }
    }
}
