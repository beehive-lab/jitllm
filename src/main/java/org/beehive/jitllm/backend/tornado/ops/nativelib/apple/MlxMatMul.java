package org.beehive.jitllm.backend.tornado.ops.nativelib.apple;

import static org.beehive.jitllm.program.op.OperationKind.MAT_MUL;
import static org.beehive.jitllm.program.op.OperationKind.RESIDUAL_ADD;
import static org.beehive.jitllm.program.op.OperationKind.RMS_NORM;
import static org.beehive.jitllm.program.op.OperationKind.SWIGLU;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.ops.Fusion;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Instance;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The batched prefill's projections on Apple MLX. MLX multiplies in FP16, so JIT kernels around
 * each MLX GEMM convert to FP16 and back, and keep FP16 values in range ({@link
 * MlxPrefillProjections}). Those conversions are merged into the steps around the GEMM: the RMS
 * norm writes FP16, and the residual add reads FP16.
 *
 * <p>Its fusion table: RMS norm with q, k and v; the output projection with its residual add; and
 * the whole feed-forward block; the first and last also over Phi-3's fused weights.
 */
public final class MlxMatMul implements Implementation {

    static final Fusion ATTENTION_INPUT =
            Fusion.of("RMS norm + q|k|v", RMS_NORM, MAT_MUL, MAT_MUL, MAT_MUL)
                    .when(Fusion::sameInput);

    static final Fusion OUTPUT =
            Fusion.of("output + residual add", MAT_MUL, RESIDUAL_ADD)
                    .when(
                            ops ->
                                    ((MatMul) ops.get(0)).weight().role()
                                            == TensorRole.ATTENTION_OUTPUT);

    static final Fusion FEED_FORWARD =
            Fusion.of(
                    "RMS norm + gate|up + SwiGLU + down + residual add",
                    RMS_NORM,
                    MAT_MUL,
                    MAT_MUL,
                    SWIGLU,
                    MAT_MUL,
                    RESIDUAL_ADD);

    /** The same steps over Phi-3's fused q|k|v and gate|up weights. */
    static final Fusion ATTENTION_INPUT_FUSED =
            Fusion.of("RMS norm + q|k|v", RMS_NORM, MAT_MUL)
                    .when(ops -> ((MatMul) ops.get(1)).weight().role() == TensorRole.ATTENTION_QKV);

    static final Fusion FEED_FORWARD_FUSED =
            Fusion.of(
                            "RMS norm + gate|up + SwiGLU + down + residual add",
                            RMS_NORM,
                            MAT_MUL,
                            SWIGLU,
                            MAT_MUL,
                            RESIDUAL_ADD)
                    .when(ops -> ((MatMul) ops.get(1)).weight().role() == TensorRole.FFN_GATE_UP);

    private static final List<Fusion> FUSIONS =
            List.of(
                    FEED_FORWARD,
                    ATTENTION_INPUT,
                    OUTPUT,
                    FEED_FORWARD_FUSED,
                    ATTENTION_INPUT_FUSED);

    private final MlxPrefillProjections projections;

    /** MLX for the layers whose projections {@code projections} holds in MLX's forms. */
    public MlxMatMul(MlxPrefillProjections projections) {
        this.projections = projections;
    }

    @Override
    public String library() {
        return "MLX";
    }

    @Override
    public List<Fusion> fusions() {
        return FUSIONS;
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return true;
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        return new Instance() {

            /** The tokens of a chunk, by graph name. */
            private final Map<String, Integer> graphs = new LinkedHashMap<>();

            @Override
            public void emit(
                    TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
                graphs.put(graph.getTaskGraphName(), step.tokens());
                if (step.fusion() == ATTENTION_INPUT || step.fusion() == ATTENTION_INPUT_FUSED) {
                    reduce(graph, "batch_attn_rms", step, operands, bindings, true);
                    projections.qkv(graph, step.layer());
                } else if (step.fusion() == OUTPUT) {
                    projections.attentionOutput(graph, step.layer());
                } else if (step.fusion() == FEED_FORWARD || step.fusion() == FEED_FORWARD_FUSED) {
                    reduce(graph, "batch_ffn_rms", step, operands, bindings, false);
                    projections.ffn(graph, step.layer());
                } else {
                    throw new IllegalArgumentException("not a fusion of MLX: " + step.fusion());
                }
            }

            /** The RMS row scales; MLX's helper then applies them while converting to FP16. */
            private void reduce(
                    TaskGraph graph,
                    String task,
                    Step step,
                    Operands operands,
                    WeightBindings bindings,
                    boolean attention) {
                RmsNorm norm = step.first(RmsNorm.class);
                FloatArray x = (FloatArray) operands.activation(norm.input());
                FloatArray scale =
                        (FloatArray) operands.activation(norm.output().name() + ".scale");
                if (attention) {
                    graph.task(
                            task,
                            TransformerBatchPrefillKernels::batchedRmsReduce,
                            context,
                            x,
                            scale,
                            norm.groupLength(),
                            norm.epsilon());
                } else {
                    graph.task(
                            task,
                            TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                            context,
                            x,
                            scale,
                            norm.groupLength(),
                            norm.epsilon());
                }
                bindings.record(
                        operands.weight(norm.weight().orElseThrow(), step.layer()).asFloatArray(),
                        graph.getTaskGraphName());
            }

            @Override
            public void addGrids(GridScheduler scheduler) {
                graphs.forEach(
                        (graph, tokens) -> {
                            String p = graph + ".";
                            scheduler.addWorkerGrid(
                                    p + "batch_attn_rms",
                                    WorkerGridFactory.genericWorker(tokens, 1));
                            projections.addGrids(scheduler, p);
                            scheduler.addWorkerGrid(
                                    p + "batch_ffn_rms",
                                    WorkerGridFactory.genericWorker(tokens, 1));
                        });
            }
        };
    }
}
