package org.beehive.jitllm.backend.tornado.layers.devstral;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.layers.MlxWeightCache;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * Devstral's batched prefill on Metal, on Apple MLX: the four projections of every layer as MLX
 * GEMMs over the weights repacked into MLX's affine format ({@link MlxPrefillProjections}), and the
 * attention as MLX's fused causal attention over the gathered cache ({@link MlxPrefillAttention}).
 * The RMS reductions and the rotation into the paged key/value cache stay on the JIT kernels.
 *
 * <p>There is no JIT form: Devstral's weights are Q4_K, which no Metal batched kernel reads, so on
 * Metal this family prefills in batches only through MLX ({@code --with-native-libraries}, on by
 * default where MLX is installed) and refuses by name without it.
 *
 * <p>The repack of a 24B model is about 15 GB, so it is kept in {@link MlxWeightCache}: built once
 * per model, then mapped from disk rather than held in host memory.
 *
 * <p>The query width is the model's own {@code heads * headDim} (4096 for Devstral Small 2, whose
 * width is 5120), which the projections, the rotation and the attention all read from {@link
 * DevstralConfiguration#qDim()}.
 */
// @formatter:on
public class DevstralLayersBatchPrefillMlx implements BatchPrefillTransformerLayerTaskGraphs {

    /** One workgroup per token row for the RMS reductions. */
    private static final int RMS_LOCAL = 256;

    private final DevstralState state;
    private final LlamaTornadoWeights weights;
    private final DevstralConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final MlxPrefill mlx;
    private final MlxPrefillAttention attention;
    private final MlxPrefillProjections projections;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public DevstralLayersBatchPrefillMlx(
            DevstralState state,
            LlamaTornadoWeights weights,
            DevstralConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.mlx =
                MlxPrefill.resolve(state.executionPolicy(), weights.dataType(), config)
                        .orElseThrow(
                                () ->
                                        new UnsupportedOperationException(
                                                "the devstral batched prefill on Metal runs on"
                                                        + " Apple MLX: add --with-native-libraries"
                                                        + " (and install MLX: brew install mlx),"
                                                        + " or drop --batch-prefill-size"));
        this.attention =
                new MlxPrefillAttention(
                        state,
                        context,
                        batchSize,
                        config.contextLength(),
                        config.numberOfHeads(),
                        config.numberOfKeyValueHeads(),
                        config.headSize(),
                        (float) (1.0 / Math.sqrt(config.headSize())));
        this.projections =
                MlxPrefillProjections.of(
                        mlx,
                        state,
                        weights,
                        context,
                        batchSize,
                        config.numberOfLayers(),
                        config.dim(),
                        config.qDim(),
                        config.kvDim(),
                        config.hiddenDim(),
                        1.0f,
                        i -> MlxPrefillProjections.LayerWeights.of(weights, i),
                        MlxWeightCache.standard());
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    @Override
    public String describeAttention() {
        return "MLX fused causal attention over the gathered cache";
    }

    @Override
    public String describeProjections() {
        return mlx.projections() == MlxPrefill.Projections.AFFINE
                ? "MLX quantized GEMMs and attention (Q4_K/Q6_K weights repacked to MLX's affine"
                        + " format)"
                : "MLX GEMMs and attention (weights decoded once to resident FP16 copies)";
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);
        String pred = "batchPrefillLayer_" + (layerIndex - 1);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    keyCache(),
                    valueCache());
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            // EVERY_EXECUTION: acquiring or releasing a lease rewrites the table.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapBlockTable);
        }
        attention.declare(layer, layerIndex, pred);
        projections.declare(layer, layerIndex, pred);

        // The norms and the RoPE tables; the projections are MLX's, declared above.
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());

        int qDim = config.qDim();
        int kvDim = config.kvDim();

        // ── Attention ──────────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                config.dim(),
                config.rmsNormEps(),
                RMS_LOCAL);
        projections.qkv(layer, layerIndex);
        if (state.usesFp16KeyValueCache()) {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCacheFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }
        attention.attention(layer, "batch_attention", layerIndex);
        projections.attentionOutput(layer, layerIndex);

        // ── Feed-forward ───────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                config.dim(),
                config.rmsNormEps(),
                RMS_LOCAL);
        projections.ffn(layer, layerIndex);

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        attention.persist(layer);
        projections.persist(layer);
        return layer;
    }

    // @formatter:on

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rms = WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL, RMS_LOCAL);
        int ropeGlobal = batchSize * (config.qDim() / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid rope = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_rope_kv", rope);
            projections.addGrids(scheduler, p);
            attention.addGrids(scheduler, p, "batch_attention");
        }
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }

    private Object keyCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private Object valueCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }
}
