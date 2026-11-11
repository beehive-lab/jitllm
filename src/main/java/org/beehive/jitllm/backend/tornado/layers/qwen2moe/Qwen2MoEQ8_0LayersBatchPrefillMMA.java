package org.beehive.jitllm.backend.tornado.layers.qwen2moe;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeBatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.qwen2.Qwen2Q8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.backend.tornado.ops.jit.JitLayerProjections;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasLayerProjections;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * Qwen2-MoE's tensor-core batched prefill: Qwen2's attention on the tensor cores ({@link
 * Qwen2Q8_0LayersBatchPrefillMMA}), and a mixture-of-experts feed-forward in place of the dense
 * one.
 *
 * <p>The feed-forward adds the attention output to the stream, normalizes and quantizes the chunk,
 * and routes it with Qwen2-MoE's own router and top-k: a softmax over every expert, the chosen
 * experts keeping their share of it unrenormalized, for any expert count. From the routed ids and
 * weights on it runs {@link Qwen35MoeBatchFeedForward}'s expert tasks: the assignments grouped by
 * expert, the grouped int8 tensor-core GEMMs over the stacked Q8_0 experts, the shared expert
 * behind its sigmoid gate, and the weighted combine into the residual stream.
 *
 * <p>A shared expert the int8 GEMMs do not read (any weight other than Q8_0 or Q4_0) runs here
 * instead, as the dense layers run their feed-forward ({@link FP16Projections}): gate and up
 * decoded to FP16 and multiplied in one GEMM, SwiGLU, then down. The routed experts' grouped int8
 * GEMMs, run as one expert over the chunk, took about five times as long as the dense int8 GEMMs
 * over the same work, and on Q6_K half again as long as these.
 */
// @formatter:on
public class Qwen2MoEQ8_0LayersBatchPrefillMMA extends Qwen2Q8_0LayersBatchPrefillMMA
        implements LayerProjections.Tasks {

    /** Threads per router row, as the single-graph Qwen2-MoE batched layers use. */
    private static final int ROUTER_LOCAL = 32;

    // Created on first use: the base builds the graphs in its constructor, before this class's
    // fields are assigned, so these fields take no initializer.
    private Qwen35MoeBatchFeedForward moe;
    private FP16Projections sharedProjections;
    private FloatArray sharedGateUpBatch;
    private HalfFloatArray sharedHiddenFP16;

    /** One layer's FP16 shared expert: gate|up, then down. */
    private record SharedRegions(Region gateUp, Region down) {}

    /** By layer, the FP16 shared expert's regions; null where the int8 GEMMs run it. */
    private SharedRegions[] sharedRegions;

    /** Which of cuBLAS and the JIT GEMMs runs the FP16 shared experts. */
    private WorkPlan sharedPlan;

    /** The plan's arrays: the tasks this class adds name their buffers themselves. */
    private static final Operands NO_OPERANDS =
            Operands.of(java.util.Map.of(), (role, layer) -> null, List.of());

    public Qwen2MoEQ8_0LayersBatchPrefillMMA(
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            int batchSize) {
        super(state, weights, config, batchSize);
    }

    private Qwen2MoETornadoWeights moeWeights() {
        return (Qwen2MoETornadoWeights) weights;
    }

    private Qwen2MoEConfiguration moeConfig() {
        return (Qwen2MoEConfiguration) config;
    }

    private Qwen35MoeBatchFeedForward moe() {
        if (moe == null) {
            Qwen2MoEConfiguration c = moeConfig();
            Qwen2MoETornadoWeights w = moeWeights();
            moe =
                    new Qwen35MoeBatchFeedForward(
                            state.workspace,
                            new Qwen35Experts(
                                    c.numberOfExperts(),
                                    c.numberOfExpertsUsed(),
                                    c.moeHiddenDim(),
                                    c.sharedExpertHiddenDim()),
                            new Qwen35ExpertWeights<TornadoTensor>(
                                    w.routerGateLayered,
                                    w.gateExpertsLayered,
                                    w.upExpertsLayered,
                                    w.downExpertsLayered,
                                    w.sharedGateLayered,
                                    w.sharedUpLayered,
                                    w.sharedDownLayered,
                                    w.sharedGateInputLayered),
                            c.dim(),
                            batchSize);
        }
        return moe;
    }

    /** Whether any layer's shared expert runs here, on {@link #sharedProjections}. */
    private boolean anyFP16Shared() {
        for (int i = 0; i < moeConfig().numberOfLayers(); i++) {
            if (!moe().int8Shared(i)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The FP16 projections of the shared experts that run here, decoded per chunk, and their packed
     * gate/up output and FP16 hidden activation; created with the first, after the plan that picks
     * their GEMM.
     */
    private FP16Projections sharedProjections() {
        if (sharedProjections == null) {
            Qwen2MoEConfiguration c = moeConfig();
            int shared = c.sharedExpertHiddenDim();
            // The narrow family takes the primary's scratch and buffers.
            Qwen2MoEQ8_0LayersBatchPrefillMMA primary =
                    family.primary(Qwen2MoEQ8_0LayersBatchPrefillMMA.class);
            sharedPlan = sharedPlan(primary != null ? primary.sharedPlan : null);
            sharedProjections =
                    primary != null
                            ? primary.sharedProjections()
                                    .narrowed(paddedBatch, Math.max(2 * shared, c.dim()))
                            : new FP16Projections(
                                    context,
                                    state.workspace.prepacking,
                                    2L * shared * c.dim(),
                                    paddedBatch,
                                    Math.max(2 * shared, c.dim()),
                                    false,
                                    2 * c.numberOfLayers(),
                                    "cuBLAS".equals(sharedPlan.libraries()));
            sharedGateUpBatch =
                    primary != null
                            ? primary.sharedGateUpBatch
                            : new FloatArray(paddedBatch * 2 * shared);
            sharedHiddenFP16 =
                    primary != null
                            ? primary.sharedHiddenFP16
                            : new HalfFloatArray(paddedBatch * shared);
        }
        return sharedProjections;
    }

    /** Attention's weights and the experts', in place of a dense gate, up and down. */
    /** The experts replace the dense feed-forward ({@link #feedForward}). */
    @Override
    protected boolean denseFeedForward() {
        return false;
    }

    @Override
    protected Object[] layerWeights(int layerIndex) {
        Qwen2MoETornadoWeights w = moeWeights();
        return new Object[] {
            w.rms_att_weightLayered[layerIndex].asFloatArray(),
            w.wqLayered[layerIndex].deviceArray(),
            w.wkLayered[layerIndex].deviceArray(),
            w.wvLayered[layerIndex].deviceArray(),
            w.woLayered[layerIndex].deviceArray(),
            w.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            w.routerGateLayered[layerIndex].asFloatArray(),
            w.sharedGateInputLayered[layerIndex].asFloatArray(),
            w.gateExpertsLayered[layerIndex].deviceArray(),
            w.upExpertsLayered[layerIndex].deviceArray(),
            w.downExpertsLayered[layerIndex].deviceArray(),
            w.sharedGateLayered[layerIndex].deviceArray(),
            w.sharedUpLayered[layerIndex].deviceArray(),
            w.sharedDownLayered[layerIndex].deviceArray()
        };
    }

    /**
     * The chunk's routing and expert scratch, and the FP16 shared expert's where a layer runs one.
     */
    private Object[] moeScratch() {
        Object[] experts = moe().scratch();
        Object[] shared =
                anyFP16Shared()
                        ? java.util.stream.Stream.concat(
                                        java.util.Arrays.stream(sharedProjections().buffers()),
                                        java.util.stream.Stream.of(
                                                sharedGateUpBatch, sharedHiddenFP16))
                                .toArray()
                        : new Object[0];
        Object[] scratch = new Object[experts.length + 4 + shared.length];
        scratch[0] = state.workspace.wrapRouterLogitsBatch;
        scratch[1] = state.workspace.wrapNormedBatch;
        scratch[2] = state.workspace.wrapQ8ActBatch;
        scratch[3] = state.workspace.wrapQ8ActScales;
        System.arraycopy(experts, 0, scratch, 4, experts.length);
        System.arraycopy(shared, 0, scratch, 4 + experts.length, shared.length);
        return scratch;
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.activeBatchSizeHolder);
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, moeScratch());
        } else {
            String pred = family.batchPrefillLayer(layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(pred, state.workspace.activeBatchSizeHolder);
            batchPrefillLayer.consumeFromDevice(pred, moeScratch());
        }
    }

    @Override
    protected void feedForward(TaskGraph batchPrefillLayer, int layerIndex) {
        Qwen2MoEConfiguration c = moeConfig();
        int dim = c.dim();
        // x += woOut, fused into the RMS reduction, as the dense layers do.
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                c.rmsNormEps(),
                RMS_LOCAL_SIZE);
        batchPrefillLayer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapXBatch,
                moeWeights().rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        batchPrefillLayer.task(
                "moe_act_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                dim);
        batchPrefillLayer.task(
                "batch_router",
                Qwen2MoEBatchKernels::batchedRouterProjection,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapRouterLogitsBatch,
                moeWeights().routerGateLayered[layerIndex].asFloatArray(),
                state.workspace.activeBatchSizeHolder,
                dim,
                c.numberOfExperts(),
                ROUTER_LOCAL);
        batchPrefillLayer.task(
                "batch_topk",
                Qwen2MoEBatchKernels::batchedSoftmaxAndTopK,
                context,
                state.workspace.wrapRouterLogitsBatch,
                state.workspace.wrapMoeIdsBatch,
                state.workspace.wrapMoeWeightsBatch,
                state.workspace.activeBatchSizeHolder,
                c.numberOfExperts(),
                c.numberOfExpertsUsed());
        moe().appendSharedGateInput(batchPrefillLayer, context, layerIndex);
        boolean fp16Shared = !moe().int8Shared(layerIndex);
        if (fp16Shared) {
            appendSharedExpertFP16(batchPrefillLayer, layerIndex);
        }
        moe().appendExperts(batchPrefillLayer, context, layerIndex, !fp16Shared);
    }

    /**
     * The shared expert over the chunk into {@code wrapMoeSharedOut}, which the combine gates and
     * adds: the normed chunk in FP16, gate and up in one GEMM, SwiGLU, down.
     */
    private void appendSharedExpertFP16(TaskGraph layer, int layerIndex) {
        Qwen2MoEConfiguration c = moeConfig();
        Qwen2MoETornadoWeights w = moeWeights();
        int dim = c.dim();
        int shared = c.sharedExpertHiddenDim();
        sharedProjections();
        layer.task(
                "shared_rms_apply16",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                w.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        sharedPlan.emit(
                layer,
                sharedRegions[layerIndex].gateUp(),
                NO_OPERANDS,
                state.workspace.weightBindings);
        layer.task(
                "shared_swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                sharedHiddenFP16,
                sharedGateUpBatch,
                shared);
        sharedPlan.emit(
                layer,
                sharedRegions[layerIndex].down(),
                NO_OPERANDS,
                state.workspace.weightBindings);
    }

    /**
     * The plan of the FP16 shared experts: for each layer that runs one here, gate|up and down,
     * each on cuBLAS or on the JIT GEMMs ({@link FP16Projections}).
     */
    private WorkPlan sharedPlan(WorkPlan primary) {
        Qwen2MoEConfiguration c = moeConfig();
        Qwen2MoETornadoWeights w = moeWeights();
        int dim = c.dim();
        int shared = c.sharedExpertHiddenDim();
        sharedRegions = new SharedRegions[c.numberOfLayers()];
        List<Region> all = new ArrayList<>();
        for (int i = 0; i < c.numberOfLayers(); i++) {
            if (moe().int8Shared(i)) {
                continue;
            }
            Region gateUp =
                    new Region(
                            i,
                            "shared gate|up",
                            batchSize,
                            List.of(
                                    projection(
                                            TensorRole.MOE_SHARED_GATE,
                                            "sharedIn",
                                            "sharedGateUp",
                                            shared,
                                            dim,
                                            paddedBatch,
                                            w.sharedGateLayered[i]),
                                    projection(
                                            TensorRole.MOE_SHARED_UP,
                                            "sharedIn",
                                            "sharedGateUp",
                                            shared,
                                            dim,
                                            paddedBatch,
                                            w.sharedUpLayered[i])));
            Region down =
                    new Region(
                            i,
                            "shared down",
                            batchSize,
                            List.of(
                                    projection(
                                            TensorRole.MOE_SHARED_DOWN,
                                            "sharedHidden",
                                            "sharedOut",
                                            dim,
                                            shared,
                                            batchSize,
                                            w.sharedDownLayered[i])));
            sharedRegions[i] = new SharedRegions(gateUp, down);
            all.add(gateUp);
            all.add(down);
        }
        return WorkPlan.of(
                primary,
                all,
                List.of(new CublasLayerProjections(this), new JitLayerProjections(this)),
                Target.of(state.executionPolicy()),
                context,
                state.workspace.prepacking);
    }

    /** {@code input} times {@code weight}: {@code outputs} values for each of {@code rows} rows. */
    private static MatMul projection(
            TensorRole role,
            String input,
            String output,
            int outputs,
            int inputs,
            int rows,
            TornadoTensor weight) {
        return new MatMul(
                new OperandRef.Weight(role.name().toLowerCase(Locale.ROOT), role),
                new OperandRef.Activation(input),
                new OperandRef.Activation(output),
                outputs,
                inputs,
                rows,
                weight.dataType());
    }

    @Override
    public boolean cublasForm(Step step) {
        return true;
    }

    /**
     * Adds a shared expert's gate|up or down GEMM. {@link FP16Projections} runs every one on the
     * GEMM the plan picked for them all, cuBLAS or JIT.
     */
    @Override
    public void project(TaskGraph graph, Step step, Operands operands, boolean cublas) {
        if (cublas != sharedProjections.nativeGemm()) {
            throw new IllegalStateException("the shared experts run on one GEMM: " + step);
        }
        Qwen2MoETornadoWeights w = moeWeights();
        int layerIndex = step.layer();
        switch (step.first(MatMul.class).weight().role()) {
            case MOE_SHARED_GATE ->
                    sharedProjections.project(
                            graph,
                            "sharedGateUpProj",
                            state.workspace.normedXFFNFP16,
                            sharedGateUpBatch,
                            paddedBatch,
                            moeConfig().dim(),
                            w.sharedGateLayered[layerIndex],
                            w.sharedUpLayered[layerIndex]);
            case MOE_SHARED_DOWN ->
                    sharedProjections.project(
                            graph,
                            "sharedDownProj",
                            sharedHiddenFP16,
                            state.workspace.wrapMoeSharedOut,
                            batchSize,
                            moeConfig().sharedExpertHiddenDim(),
                            w.sharedDownLayered[layerIndex]);
            default ->
                    throw new IllegalArgumentException("not a shared expert projection: " + step);
        }
    }

    @Override
    protected void addFeedForwardGrids(GridScheduler scheduler, String p, int layerIndex) {
        int dim = moeConfig().dim();
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms",
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms_apply", WorkerGridFactory.genericWorker(batchSize * dim, 256));
        scheduler.addWorkerGrid(
                p + "moe_act_q8", WorkerGridFactory.genericWorker(batchSize * dim, 128));
        scheduler.addWorkerGrid(
                p + "batch_router",
                WorkerGridFactory.genericWorker(
                        batchSize * moeConfig().numberOfExperts() * ROUTER_LOCAL, ROUTER_LOCAL));
        scheduler.addWorkerGrid(p + "batch_topk", WorkerGridFactory.genericWorker(batchSize, 1));
        moe().addSharedGateInputGrid(scheduler, p);
        boolean fp16Shared = !moe().int8Shared(layerIndex);
        if (fp16Shared) {
            int shared = moeConfig().sharedExpertHiddenDim();
            scheduler.addWorkerGrid(
                    p + "shared_rms_apply16",
                    WorkerGridFactory.genericWorker(batchSize * dim, 256));
            scheduler.addWorkerGrid(p + "sharedGateUpProj", mmaGrid(paddedBatch, 2 * shared));
            scheduler.addWorkerGrid(p + "shared_swiglu", elementwiseGrid(batchSize * shared));
            scheduler.addWorkerGrid(p + "sharedDownProj", mmaGrid(paddedBatch, dim));
            // After the GEMMs' grids, which a split projection's replace.
            sharedProjections().addGrids(scheduler);
        }
        moe().addExpertGrids(scheduler, p, layerIndex, !fp16Shared);
    }
}
