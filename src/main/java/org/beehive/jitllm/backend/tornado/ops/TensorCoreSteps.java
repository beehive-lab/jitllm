package org.beehive.jitllm.backend.tornado.ops;

import static org.beehive.jitllm.program.op.OperationKind.MAT_MUL;
import static org.beehive.jitllm.program.op.OperationKind.RESIDUAL_ADD;
import static org.beehive.jitllm.program.op.OperationKind.RMS_NORM;
import static org.beehive.jitllm.program.op.OperationKind.SWIGLU;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
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
 * The tasks a tensor-core batched prefill adds for each step, whatever way its weights reach the
 * GEMM: the RMS norms (writing the FP16 input the GEMM reads), SwiGLU and the residual adds, under
 * the task names the plan has always had. A subclass adds the projections: {@link DecodedMatMul}
 * decodes each weight to FP16 first, {@link Fp16MatMul} reads F16 weights where they are, {@link
 * DequantizingMatMul} decodes one projection at a time into a scratch.
 *
 * <p>Example: Llama's layer before attention is one {@link #NORM} step ({@code batch_attn_rms}, its
 * {@code _apply}) and one {@link #QKV} step ({@code qkvProj}).
 */
abstract class TensorCoreSteps implements Instance {

    /** Threads per row of the RMS reductions. */
    static final int RMS_LOCAL_SIZE = 256;

    static final Fusion QKV = Fusion.of("q|k|v", MAT_MUL, MAT_MUL, MAT_MUL).when(Fusion::sameInput);

    static final Fusion GATE_UP_SWIGLU =
            Fusion.of("gate|up + SwiGLU", MAT_MUL, MAT_MUL, SWIGLU).when(Fusion::sameInput);

    static final Fusion ADD_NORM = Fusion.of("residual add + RMS norm", RESIDUAL_ADD, RMS_NORM);

    static final Fusion DOWN_ADD =
            Fusion.of("down + residual add", MAT_MUL, RESIDUAL_ADD)
                    .when(ops -> role(ops.get(0)) == TensorRole.FFN_DOWN);

    /**
     * Gate and up with no SwiGLU after them: a family that runs its own SwiGLU, as Granite does.
     */
    static final Fusion GATE_UP =
            Fusion.of("gate|up", MAT_MUL, MAT_MUL)
                    .when(Fusion::sameInput)
                    .when(
                            ops ->
                                    role(ops.get(0)) == TensorRole.FFN_GATE
                                            && role(ops.get(1)) == TensorRole.FFN_UP);

    static final Fusion PROJECTION = Fusion.of("projection", MAT_MUL);

    static final Fusion NORM = Fusion.of("RMS norm", RMS_NORM);

    /**
     * A residual add on its own, where a family runs a task of its own between the down projection
     * and the add, as Granite scales the projection's output.
     */
    static final Fusion RESIDUAL = Fusion.of("residual add", RESIDUAL_ADD);

    /** The fusion table of every tensor-core implementation, longest first. */
    static final List<Fusion> FUSIONS =
            List.of(QKV, GATE_UP_SWIGLU, ADD_NORM, DOWN_ADD, GATE_UP, PROJECTION, NORM, RESIDUAL);

    static TensorRole role(Operation op) {
        return ((MatMul) op).weight().role();
    }

    /**
     * The task name of a projection GEMM, by the weight it multiplies: the names these layers have
     * always had. Example: the down projection is {@code w2Proj}.
     */
    static String taskName(MatMul projection) {
        return switch (projection.weight().role()) {
            case ATTENTION_QUERY, ATTENTION_QKV -> "qkvProj";
            case ATTENTION_OUTPUT -> "woProj";
            case FFN_GATE, FFN_GATE_UP -> "gateUpProj";
            case FFN_DOWN -> "w2Proj";
            default ->
                    throw new IllegalArgumentException(
                            "no projection task for " + projection.weight().role());
        };
    }

    /** Whether {@code projection} is q (of q|k|v) or gate (of gate|up), split or fused. */
    static boolean startsNativeLibContiguousProjection(MatMul projection) {
        return switch (projection.weight().role()) {
            case ATTENTION_QUERY, ATTENTION_QKV, FFN_GATE, FFN_GATE_UP -> true;
            default -> false;
        };
    }

    protected final KernelContext context;

    /** The grid of each task added, by graph-qualified task name. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

    /**
     * The last projection that wrote each activation, by name: a residual add on its own takes its
     * width from the projection whose output it adds.
     */
    private final Map<String, MatMul> producers = new LinkedHashMap<>();

    TensorCoreSteps(KernelContext context) {
        this.context = context;
    }

    /**
     * Adds the GEMM task {@code task} over the step's projections, outputs side by side, and
     * returns the first projection.
     */
    protected abstract MatMul project(
            TaskGraph graph, String task, Step step, Operands operands, WeightBindings bindings);

    /** Registers the grids the projections' own helpers keep (split GEMMs, decodes). */
    protected abstract void addProjectionGrids(GridScheduler scheduler);

    @Override
    public final void emit(TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
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
        } else if (f == QKV || f == GATE_UP || f == PROJECTION) {
            projected(graph, taskName(step.matMuls().get(0)), step, operands, bindings);
        } else if (f == RESIDUAL) {
            ResidualAdd add = step.first(ResidualAdd.class);
            MatMul down = producers.get(add.right().name());
            if (down == null || down.weight().role() != TensorRole.FFN_DOWN) {
                throw new IllegalArgumentException(
                        "a residual add on its own follows the down projection, not "
                                + add.right().name());
            }
            residualAdd(graph, step, add, down, operands);
        } else if (f == GATE_UP_SWIGLU) {
            MatMul gate = projected(graph, "gateUpProj", step, operands, bindings);
            graph.task(
                    "swiglu",
                    TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                    context,
                    (HalfFloatArray) operands.activation(step.ops().get(2).outputs().get(0)),
                    (FloatArray) operands.activation(gate.output()),
                    gate.rows());
            grid(graph, "swiglu", elementwise(step.tokens() * gate.rows()));
        } else if (f == DOWN_ADD) {
            MatMul down = projected(graph, "w2Proj", step, operands, bindings);
            residualAdd(graph, step, step.first(ResidualAdd.class), down, operands);
        } else {
            throw new IllegalArgumentException("not a fusion of this implementation: " + f);
        }
    }

    /** {@link #project}, then the bookkeeping every projection gets. */
    private MatMul projected(
            TaskGraph graph, String task, Step step, Operands operands, WeightBindings bindings) {
        MatMul first = project(graph, task, step, operands, bindings);
        for (MatMul m : step.matMuls()) {
            bindings.record(
                    operands.weight(m.weight(), step.layer()).deviceArray(),
                    graph.getTaskGraphName());
        }
        producers.put(first.output().name(), first);
        grid(graph, task, mmaGrid(first.batchSize(), step.outputWidth()));
        return first;
    }

    /** The weights of the step's projections, in order. */
    static TornadoTensor[] weights(Step step, Operands operands) {
        List<MatMul> matMuls = step.matMuls();
        TornadoTensor[] weights = new TornadoTensor[matMuls.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = operands.weight(matMuls.get(i).weight(), step.layer());
        }
        return weights;
    }

    /** {@code x += down}: the down projection's output added to the residual stream. */
    private void residualAdd(
            TaskGraph graph, Step step, ResidualAdd add, MatMul down, Operands operands) {
        graph.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                (FloatArray) operands.activation(add.left()),
                (FloatArray) operands.activation(down.output()));
        grid(graph, "w2Resid", elementwise(step.tokens() * down.rows()));
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
                WorkerGridFactory.genericWorker(step.tokens() * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE));
        grid(graph, prefix + "_apply", WorkerGridFactory.genericWorker(step.tokens() * dim, 256));
    }

    /** Records {@code grid} for task {@code task} of {@code graph}. */
    protected final void grid(TaskGraph graph, String task, WorkerGrid grid) {
        grids.put(graph.getTaskGraphName() + "." + task, grid);
    }

    @Override
    public final void addGrids(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
        addProjectionGrids(scheduler);
    }

    /** The name of the row-scale buffer of {@code norm}: its output's name and ".scale". */
    static String scaleOf(RmsNorm norm) {
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
}
