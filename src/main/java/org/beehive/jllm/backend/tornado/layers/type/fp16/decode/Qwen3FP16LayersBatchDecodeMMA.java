package org.beehive.jllm.backend.tornado.layers.type.fp16.decode;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Qwen3 batched-DECODE transformer-layer TaskGraphs: B independent sequences, each with its own KV
 * region and position.
 *
 * <p>Structurally identical to {@link
 * org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillMMA}
 * (13-task MMA pipeline including the Qwen3 per-head Q/K RMS norm). Swaps the two KV-addressing
 * kernels for the per-slot decode variants over a B-sized KV cache:
 *
 * <ul>
 *   <li>{@code batch_rope_kv} → {@link Qwen3Kernels#batchedDecodeRopeWithKVCacheQwen3Packed}
 *   <li>{@code batch_attention} → {@link
 *       TransformerBatchPrefillKernels#batchedDecodeAttentionFP16Out}
 * </ul>
 */
public class Qwen3FP16LayersBatchDecodeMMA {

    static final int RMS_LOCAL_SIZE = 256;

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int paddedBatch;
    private final int decodeCtx;
    private final int nHeadKv;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    // The engine's pool in the representation it was built with: exactly one pair is set.
    private final FloatArray keyCacheBatch;
    private final FloatArray valueCacheBatch;
    private final HalfFloatArray keyCacheBatchFP16;
    private final HalfFloatArray valueCacheBatchFP16;
    private final IntArray seqPositions;
    private final boolean paged;
    private final IntArray blockTable;
    private final int blockSize;
    private final int maxBlocksPerSlot;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen3FP16LayersBatchDecodeMMA(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize,
            int decodeCtx,
            Object keyCacheBatch,
            Object valueCacheBatch,
            IntArray seqPositions) {
        this(
                state,
                weights,
                config,
                batchSize,
                decodeCtx,
                keyCacheBatch,
                valueCacheBatch,
                seqPositions,
                null,
                0,
                0);
    }

    /** Paged constructor: keyCacheBatch/valueCacheBatch are the shared block pools. */
    public Qwen3FP16LayersBatchDecodeMMA(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize,
            int decodeCtx,
            Object keyCacheBatch,
            Object valueCacheBatch,
            IntArray seqPositions,
            IntArray blockTable,
            int blockSize,
            int maxBlocksPerSlot) {
        this.paged = blockTable != null;
        this.blockTable = blockTable;
        this.blockSize = blockSize;
        this.maxBlocksPerSlot = maxBlocksPerSlot;
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        this.decodeCtx = decodeCtx;
        this.keyCacheBatch = keyCacheBatch instanceof FloatArray f ? f : null;
        this.valueCacheBatch = valueCacheBatch instanceof FloatArray f ? f : null;
        this.keyCacheBatchFP16 = keyCacheBatch instanceof HalfFloatArray h ? h : null;
        this.valueCacheBatchFP16 = valueCacheBatch instanceof HalfFloatArray h ? h : null;
        if (blockTable == null && this.keyCacheBatchFP16 != null) {
            throw new IllegalArgumentException(
                    "the contiguous (non-paged) batch cache is FP32 only; the FP16 pool is paged");
        }
        if ((this.keyCacheBatch == null) == (this.keyCacheBatchFP16 == null)) {
            throw new IllegalArgumentException(
                    "the pool must be one FloatArray or HalfFloatArray pair");
        }
        this.seqPositions = seqPositions;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        this.kvDim = config.numberOfHeadsValue() * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    // @formatter:off
    private TaskGraph createLayerTaskGraph(int layerIndex) {
        String graphName = "batchDecodeLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph g = new TaskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        if (layerIndex == 0) {
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, seqPositions);
            if (paged) {
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, blockTable);
            }
            g.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyPool(),
                    valuePool(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            g.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchDecodeLayer_" + (layerIndex - 1);
            if (paged) {
                g.consumeFromDevice(pred, context, blockTable);
            }
            g.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    seqPositions,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyPool(),
                    valuePool(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
        }

        g.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

        g.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        g.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        g.task(
                "qkvProj",
                TransformerBatchPrefillKernels::gemmMMAQKV,
                context,
                state.workspace.wrapXbFP16Batch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                state.workspace.qkvResultBatch,
                paddedBatch,
                qDim,
                kvDim,
                dim);

        // Qwen3: per-head Q/K RMS norm before RoPE (per-token, no KV → reused as-is).
        g.task(
                "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNormPacked,
                context,
                state.workspace.qkvResultBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                nHeadKv,
                nEmbdHead,
                qDim,
                kvDim,
                config.rmsNormEps());

        // DECODE: per-slot position + per-slot KV region (paged via block table, or contiguous).
        if (paged) {
            int blockCfg = blockSize | (maxBlocksPerSlot << 16);
            if (keyCacheBatchFP16 != null) {
                g.task(
                        "batch_rope_kv",
                        Qwen3Kernels::batchedDecodePagedRopeWithKVCacheQwen3PackedKVFP16,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        config.ropeTheta(),
                        kvDim,
                        nEmbdHead,
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        qDim);
            } else {
                g.task(
                        "batch_rope_kv",
                        Qwen3Kernels::batchedDecodePagedRopeWithKVCacheQwen3Packed,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatch,
                        valueCacheBatch,
                        config.ropeTheta(),
                        kvDim,
                        nEmbdHead,
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        qDim);
            }

            // Shared paged attention: qDim == nHeads*nEmbdHead for Qwen3.
            if (keyCacheBatchFP16 != null) {
                g.task(
                        "batch_attention",
                        TransformerBatchPrefillKernels::batchedDecodePagedAttentionFP16OutKVFP16,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        nEmbdHead,
                        kvDim,
                        gqa,
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg);
            } else {
                g.task(
                        "batch_attention",
                        TransformerBatchPrefillKernels::batchedDecodePagedAttentionFP16Out,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatch,
                        valueCacheBatch,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        nEmbdHead,
                        kvDim,
                        gqa,
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg);
            }
        } else {
            g.task(
                    "batch_rope_kv",
                    Qwen3Kernels::batchedDecodeRopeWithKVCacheQwen3Packed,
                    context,
                    seqPositions,
                    state.workspace.qkvResultBatch,
                    keyCacheBatch,
                    valueCacheBatch,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    config.numberOfLayers(),
                    decodeCtx,
                    qDim);

            g.task(
                    "batch_attention",
                    TransformerBatchPrefillKernels::batchedDecodeAttentionFP16Out,
                    context,
                    seqPositions,
                    state.workspace.qkvResultBatch,
                    keyCacheBatch,
                    valueCacheBatch,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    config.numberOfLayers(),
                    decodeCtx,
                    qDim);
        }

        g.task(
                "woProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.attnOutFP16,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                state.workspace.woOut,
                paddedBatch,
                dim,
                qDim);

        g.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        g.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        g.task(
                "gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUp,
                context,
                state.workspace.normedXFFNFP16,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                hidDim,
                dim);

        g.task(
                        "swiglu",
                        TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context,
                        state.workspace.wrapHbFP16Batch,
                        state.workspace.gateUpResultBatch,
                        hidDim)
                .task(
                        "w2Proj",
                        TransformerBatchPrefillKernels::gemmMMA,
                        context,
                        state.workspace.wrapHbFP16Batch,
                        weights.w2Layered[layerIndex].asHalfFloatArray(),
                        state.workspace.w2Out,
                        paddedBatch,
                        dim,
                        hidDim)
                .task(
                        "w2Resid",
                        TransformerBatchPrefillKernels::batchedResidualAddFP32,
                        context,
                        state.workspace.wrapXBatch,
                        state.workspace.w2Out);

        g.persistOnDevice(state.workspace.wrapXBatch, keyPool(), valuePool());
        return g;
    }

    // @formatter:on

    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;
        int nBlocks = N / 128;
        WorkerGrid2D grid = new WorkerGrid2D(mBlocks * 256, nBlocks);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    static WorkerGrid elementwiseGrid(int n) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();

        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (nHeads + nHeadKv) * nEmbdHead, nEmbdHead);

        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int attnLocal = Math.min(nEmbdHead, 128);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        WorkerGrid mmaQkvWorker = mmaGrid(paddedBatch, qDim + 2 * kvDim);
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim);
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim);

        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim);
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchDecodeLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj", mmaQkvWorker);
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", ffnRmsApplyWorker);
            scheduler.addWorkerGrid(p + "gateUpProj", mmaGateUpWorker);
            scheduler.addWorkerGrid(p + "swiglu", ewHidWorker);
            scheduler.addWorkerGrid(p + "w2Proj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "w2Resid", ewDimWorker);
        }
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

    /** The key pool the graphs bind, in whichever representation the engine built. */
    private Object keyPool() {
        return keyCacheBatchFP16 != null ? keyCacheBatchFP16 : keyCacheBatch;
    }

    /** The value pool, following {@link #keyPool()}. */
    private Object valuePool() {
        return valueCacheBatchFP16 != null ? valueCacheBatchFP16 : valueCacheBatch;
    }
}
