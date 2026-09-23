package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan (Q8_0).
 *
 * <p>Mirrors {@link
 * org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill} but uses
 * Q8_0 kernels with inline dequantization. Key differences from the FP16 path:
 *
 * <ul>
 *   <li>{@code wrapXBatch} is filled with dequantized FP32 embeddings by the host before the
 *       activation graph runs (no on-device FP16→FP32 conversion).
 *   <li>{@code wrapXbBatch} (FP32) is reused as the normalized xb intermediate: written by {@code
 *       batchedRmsApplyFP32}, read by {@code batchedFusedQKVMatmulQ8}, then overwritten by flash
 *       attention output.
 *   <li>{@code wrapXbFP16Batch} is not used.
 *   <li>Weight matrices are {@code ByteArray} (Q8_0 format).
 * </ul>
 */
public class LlamaQ8_0LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaQ8_0LayersBatchPrefill(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph layer = new TaskGraph(graphName);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            // Allocate GPU-side batch intermediates once.
            // wrapXBatch is filled with dequantized FP32 by the host, persisted by
            // prefillActivation.
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache());
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch);
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            layer.consumeFromDevice(
                    "batchPrefillLayer_" + (layerIndex - 1), state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once (Q8_0 format)
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

        // Writes FP32 normalized xb into wrapXbBatch (reused later by flash attention)
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        layer.task(
                "batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim,
                kvDim,
                LOCAL_WORK_GROUP_SIZE);

        if (useFp16KVCache()) {
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
                    dim);
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
                    dim);
        }

        // Overwrites wrapXbBatch with attention output
        if (useFp16KVCache()) {
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        } else {
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        }

        layer.task(
                "batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim,
                dim,
                LOCAL_WORK_GROUP_SIZE);

        // ── FFN Block ──────────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        layer.task(
                "batch_ffn_gate_up",
                TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUpQ8,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                dim,
                hidDim,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_ffn_down",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context,
                state.workspace.wrapHbBatch,
                state.workspace.wrapXBatch,
                weights.w2Layered[layerIndex].asByteArray(),
                hidDim,
                dim,
                LOCAL_WORK_GROUP_SIZE);

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());

        return layer;
    }

    // @formatter:on

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        int qkvRows = dim + 2 * kvDim;
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int optLocal = findOptimalLocalSize(headSz);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * optLocal, optLocal);
        WorkerGrid matVecDimWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid matVecHidWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            scheduler.addWorkerGrid(p + "batch_attn_out", matVecDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", matVecHidWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_down", matVecDimWorker);
        }
    }

    private static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) {
                    optimal = s;
                    break;
                }
            }
        }
        return optimal;
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }

    /**
     * Whether the cache is half precision: the representation the state holds, which the decode
     * layers after this prefill bind as well.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    /** The key cache every graph of this family binds: FP16 when the state holds one. */
    private Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache, following {@link #keyCache()}. */
    private Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }
}
