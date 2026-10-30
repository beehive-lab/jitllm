package org.beehive.jitllm.backend.tornado.ops;

import static org.beehive.jitllm.program.op.OperationKind.MAT_MUL;
import static org.beehive.jitllm.program.op.OperationKind.RESIDUAL_ADD;
import static org.beehive.jitllm.program.op.OperationKind.RMS_NORM;
import static org.beehive.jitllm.program.op.OperationKind.SWIGLU;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.ResidualAdd;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The batched prefill on a tensor-core GPU. The projections run in two steps: decode each weight to
 * FP16 (whatever its type in the file, Q8_0, Q4_K, ...), then one FP16 GEMM that writes FP32. The
 * subclass picks the GEMM: the JIT GEMM or cuBLAS. The decode and the GEMM are done by the existing
 * {@link FP16Projections}.
 *
 * <p>Its fusion table also runs the steps around the GEMMs: the RMS norms (writing the FP16 input
 * the GEMM reads), SwiGLU and the residual adds. The residual add after attention is merged into
 * the next RMS norm.
 */
public abstract class DecodedMatMul implements Implementation {

    /** Threads per row of the RMS reductions. */
    static final int RMS_LOCAL_SIZE = 256;

    static final Fusion QKV = Fusion.of("q|k|v", MAT_MUL, MAT_MUL, MAT_MUL).when(Fusion::sameInput);

    static final Fusion GATE_UP_SWIGLU =
            Fusion.of("gate|up + SwiGLU", MAT_MUL, MAT_MUL, SWIGLU).when(Fusion::sameInput);

    static final Fusion ADD_NORM = Fusion.of("residual add + RMS norm", RESIDUAL_ADD, RMS_NORM);

    static final Fusion DOWN_ADD =
            Fusion.of("down + residual add", MAT_MUL, RESIDUAL_ADD)
                    .when(ops -> role(ops.get(0)) == TensorRole.FFN_DOWN);

    static final Fusion PROJECTION = Fusion.of("projection", MAT_MUL);

    static final Fusion NORM = Fusion.of("RMS norm", RMS_NORM);

    private static final List<Fusion> FUSIONS =
            List.of(QKV, GATE_UP_SWIGLU, ADD_NORM, DOWN_ADD, PROJECTION, NORM);

    private static TensorRole role(Operation op) {
        return ((MatMul) op).weight().role();
    }

    /** Whether the GEMM is cuBLAS ({@code true}) or the JIT GEMM ({@code false}). */
    protected abstract boolean nativeGemm();

    @Override
    public List<Fusion> fusions() {
        return FUSIONS;
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

    /** Adds the tasks of each step, under the task names the plan has always had. */
    private static final class Tasks implements Instance {

        private final KernelContext context;
        private final FP16Projections projections;

        /** The grid of each task added, by graph-qualified task name. */
        private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

        Tasks(KernelContext context, FP16Projections projections) {
            this.context = context;
            this.projections = projections;
        }

        @Override
        public Object[] buffers() {
            return projections.buffers();
        }

        @Override
        public void emit(TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
            Fusion f = step.fusion();
            if (f == NORM) {
                norm(graph, step, step.first(RmsNorm.class), null, operands, bindings);
            } else if (f == ADD_NORM) {
                norm(
                        graph,
                        step,
                        step.first(RmsNorm.class),
                        step.first(ResidualAdd.class),
                        operands,
                        bindings);
            } else if (f == QKV) {
                project(graph, "qkvProj", step, operands, bindings);
            } else if (f == GATE_UP_SWIGLU) {
                MatMul gate = project(graph, "gateUpProj", step, operands, bindings);
                graph.task(
                        "swiglu",
                        TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context,
                        (HalfFloatArray) operands.activation(step.ops().get(2).outputs().get(0)),
                        (FloatArray) operands.activation(gate.output()),
                        gate.rows());
                grid(graph, "swiglu", elementwise(step.tokens() * gate.rows()));
            } else if (f == DOWN_ADD) {
                MatMul down = project(graph, "w2Proj", step, operands, bindings);
                ResidualAdd add = step.first(ResidualAdd.class);
                graph.task(
                        "w2Resid",
                        TransformerBatchPrefillKernels::batchedResidualAddFP32,
                        context,
                        (FloatArray) operands.activation(add.left()),
                        (FloatArray) operands.activation(down.output()));
                grid(graph, "w2Resid", elementwise(step.tokens() * down.rows()));
            } else if (f == PROJECTION) {
                project(graph, "woProj", step, operands, bindings);
            } else {
                throw new IllegalArgumentException("not a fusion of this implementation: " + f);
            }
        }

        /**
         * RMS norm into the FP16 input of the next GEMM: the row scales, then the scaled rows. With
         * {@code add}, the residual add before it runs inside the reduction.
         */
        private void norm(
                TaskGraph graph,
                Step step,
                RmsNorm norm,
                ResidualAdd add,
                Operands operands,
                WeightBindings bindings) {
            boolean attention = norm.weight().orElseThrow().role() == TensorRole.ATTENTION_NORM;
            String prefix = attention ? "batch_attn_rms" : "batch_ffn_rms";
            FloatArray x = (FloatArray) operands.activation(norm.input());
            FloatArray scale = (FloatArray) operands.activation(scaleOf(norm));
            TornadoTensor weight = operands.weight(norm.weight().orElseThrow(), step.layer());
            int dim = norm.groupLength();
            if (add == null) {
                graph.task(
                        prefix,
                        TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                        context,
                        x,
                        scale,
                        dim,
                        norm.epsilon(),
                        RMS_LOCAL_SIZE);
            } else {
                graph.task(
                        prefix,
                        TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                        context,
                        x,
                        (FloatArray) operands.activation(add.right()),
                        scale,
                        dim,
                        norm.epsilon(),
                        RMS_LOCAL_SIZE);
            }
            HalfFloatArray out = (HalfFloatArray) operands.activation(norm.output());
            if (attention) {
                graph.task(
                        prefix + "_apply",
                        TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                        context,
                        out,
                        x,
                        weight.asFloatArray(),
                        scale,
                        dim);
            } else {
                graph.task(
                        prefix + "_apply",
                        TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                        context,
                        out,
                        x,
                        weight.asFloatArray(),
                        scale,
                        dim);
            }
            bindings.record(weight.asFloatArray(), graph.getTaskGraphName());
            grid(
                    graph,
                    prefix,
                    WorkerGridFactory.genericWorker(
                            step.tokens() * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE));
            grid(
                    graph,
                    prefix + "_apply",
                    WorkerGridFactory.genericWorker(step.tokens() * dim, 256));
        }

        /** One GEMM over the step's projections, outputs side by side; returns the first. */
        private MatMul project(
                TaskGraph graph,
                String task,
                Step step,
                Operands operands,
                WeightBindings bindings) {
            List<MatMul> matMuls = step.matMuls();
            MatMul first = matMuls.get(0);
            TornadoTensor[] weights = new TornadoTensor[matMuls.size()];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = operands.weight(matMuls.get(i).weight(), step.layer());
            }
            projections.project(
                    graph,
                    task,
                    (HalfFloatArray) operands.activation(first.input()),
                    (FloatArray) operands.activation(first.output()),
                    first.batchSize(),
                    first.columns(),
                    weights);
            for (TornadoTensor w : weights) {
                bindings.record(w.deviceArray(), graph.getTaskGraphName());
            }
            grid(graph, task, mmaGrid(first.batchSize(), step.outputWidth()));
            return first;
        }

        private void grid(TaskGraph graph, String task, WorkerGrid grid) {
            grids.put(graph.getTaskGraphName() + "." + task, grid);
        }

        @Override
        public void addGrids(GridScheduler scheduler) {
            grids.forEach(scheduler::addWorkerGrid);
            projections.addGrids(scheduler);
        }
    }

    /** The name of the row-scale buffer of {@code norm}: its output's name and ".scale". */
    public static String scaleOf(RmsNorm norm) {
        return norm.output().name() + ".scale";
    }

    /** The grid of the tensor-core GEMMs: 256 threads per 128×128 block of the output. */
    static WorkerGrid mmaGrid(int paddedM, int n) {
        WorkerGrid2D g = new WorkerGrid2D((paddedM / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** One thread per value, 256 per workgroup; {@code n} is a multiple of 256. */
    static WorkerGrid elementwise(int n) {
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
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
