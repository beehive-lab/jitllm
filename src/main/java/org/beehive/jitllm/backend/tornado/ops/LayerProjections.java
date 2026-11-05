package org.beehive.jitllm.backend.tornado.ops;

import static org.beehive.jitllm.program.op.OperationKind.MAT_MUL;
import static org.beehive.jitllm.program.op.OperationKind.RESIDUAL_ADD;

import java.util.List;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * The projections of a batched-prefill layer class that adds the tasks of each one itself, on the
 * JIT kernels or on cuBLAS. The layer class keeps the kernels, uploads the weights and registers
 * the grids; the plan picks, for each projection, which of the two runs it.
 *
 * <p>Example: Qwen3.5's Q8_0 ssm_alpha has 16 outputs, which the int8 GEMM does not tile. It runs
 * on cuBLAS TF32 under native libraries, and on the warp kernel without them.
 */
public abstract class LayerProjections implements Implementation {

    /** The tasks of each projection, added by the layer class that owns the kernels. */
    public interface Tasks {
        /** Whether the layer can run {@code step}'s projection on cuBLAS. */
        boolean cublasForm(Step step);

        /** Adds the tasks of {@code step}'s projection, on cuBLAS or on the JIT kernels. */
        void project(TaskGraph graph, Step step, Operands operands, boolean cublas);
    }

    /** One projection. */
    public static final Fusion PROJECTION = Fusion.of("projection", MAT_MUL);

    /** One projection whose output is added to the residual stream. */
    public static final Fusion PROJECTION_ADD =
            Fusion.of("projection + residual add", MAT_MUL, RESIDUAL_ADD);

    /**
     * Three projections of one input that the layer runs as one GEMM, for example Gemma 4's q, k
     * and v.
     */
    public static final Fusion QKV =
            Fusion.of("q|k|v", MAT_MUL, MAT_MUL, MAT_MUL).when(Fusion::sameInput);

    /**
     * Two projections of one input that the layer runs as one GEMM, for example Qwen2-MoE's shared
     * expert gate and up.
     */
    public static final Fusion GATE_UP =
            Fusion.of("gate|up", MAT_MUL, MAT_MUL).when(Fusion::sameInput);

    private static final List<Fusion> FUSIONS = List.of(QKV, GATE_UP, PROJECTION_ADD, PROJECTION);

    protected final Tasks tasks;

    protected LayerProjections(Tasks tasks) {
        this.tasks = tasks;
    }

    /** Whether this implementation runs the projections on cuBLAS. */
    protected abstract boolean cublas();

    @Override
    public List<Fusion> fusions() {
        return FUSIONS;
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        return new Instance() {
            @Override
            public void emit(
                    TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
                tasks.project(graph, step, operands, cublas());
            }

            @Override
            public void addGrids(GridScheduler scheduler) {}
        };
    }
}
