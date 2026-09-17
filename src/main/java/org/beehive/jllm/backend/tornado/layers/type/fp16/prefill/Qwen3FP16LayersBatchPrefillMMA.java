package org.beehive.jllm.backend.tornado.layers.type.fp16.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jllm.backend.tornado.kernels.CuDnnPrefillAttentionKernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
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
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cudnn.CuDnn;

/**
 * Qwen3 batched-prefill transformer-layer TaskGraphs on the tensor-core (MMA) pipeline. Mirrors
 * {@link LlamaFP16LayersBatchPrefillMMA} with the Qwen3 architectural additions: per-head Q/K RMS
 * normalization between the QKV projection and RoPE, split-half RoPE pairing, and qDim-shaped
 * attention (qDim = nHeads * headDim, which may differ from the model dim).
 *
 * <p>Tensor-core layer pipeline (13 tasks, all GEMMs on MMA):
 *
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_qk_rmsnorm      per-head Q/K RMS norm over the packed buffer
 *   batch_rope_kv         split-half RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM [dim × qDim] → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>Requires dim, qDim, kvDim, and hidDim to be multiples of 128 (holds for all standard Qwen3
 * checkpoints).
 */
public class Qwen3FP16LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    private final Qwen3State state;
    private final boolean packedHalf2Attention;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM); see the Llama
    // planner for the padding rationale. Non-GEMM kernels use the true batchSize.
    private final int paddedBatch;
    private final int nHeadKv;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final boolean cudnnAttention;
    private final boolean cublasProjection;
    /** Diagnostics only: makes the cuDNN staging buffers host-readable after each layer. */
    private static final boolean CUDNN_DEBUG =
            Boolean.getBoolean("jllm.attention.cudnnPrefill.debug");
    /** Which layer the debug transfer captures (the staging buffers are shared by all). */
    /** Diagnostics only: seed the staging buffers with a finite sentinel instead of zero. */
    private static final boolean CUDNN_CANARY =
            Boolean.getBoolean("jllm.attention.cudnnPrefill.canary");
    private static final int CUDNN_DEBUG_LAYER =
            Integer.getInteger("jllm.attention.cudnnPrefill.debugLayer", 0);
    /** Diagnostics only: last-built staging buffers, for the NaN hunt. */
    public static HalfFloatArray dbgQ, dbgK, dbgV, dbgOut;
    /**
     * Staging for the cuDNN call: contiguous FP16 Q, GQA-expanded K/V, and its output, all
     * {@code [head][tok][headDim]}. Allocated once and reused by every layer graph, because
     * the layers run one after another.
     */
    private final HalfFloatArray cudnnQ;
    private final HalfFloatArray cudnnK;
    private final HalfFloatArray cudnnV;
    private final HalfFloatArray cudnnOut;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /**
     * The batched-prefill graphs only run on the CUDA backend (tensor-core gated), which is the
     * same backend the FP16 KV cache path targets.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    public Qwen3FP16LayersBatchPrefillMMA(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.state = state;
        // Resolved once from the session's policy, not read from a class constant.
        this.packedHalf2Attention = state.executionPolicy().packedHalf2Attention();
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        if (batchSize % 128 != 0) {
            System.out.printf(
                    "[jllm] prefill batch %d padded to %d for tensor-core tiles; "
                            + "GEMM efficiency is %d/%d — use a multiple of 128 for best throughput.%n",
                    batchSize, paddedBatch, batchSize, paddedBatch);
        }
        this.cudnnAttention =
                Boolean.getBoolean("jllm.attention.cudnnPrefill") && state.usesFp16KeyValueCache();
        this.cublasProjection = Boolean.getBoolean("jllm.projection.cublas");
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        this.kvDim = config.numberOfHeadsValue() * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
        int cudnnElems = cudnnAttention ? qDim * batchSize : 0;
        this.cudnnQ = new HalfFloatArray(cudnnElems);
        this.cudnnK = new HalfFloatArray(cudnnElems);
        this.cudnnV = new HalfFloatArray(cudnnElems);
        this.cudnnOut = new HalfFloatArray(cudnnElems);
        if (cudnnAttention) {
            HalfFloat seed = new HalfFloat(CUDNN_CANARY ? 1234.0f : 0.0f);
            cudnnQ.init(seed);
            cudnnK.init(seed);
            cudnnV.init(seed);
            cudnnOut.init(seed);
        }
        if (cudnnAttention) {
            org.beehive.jllm.backend.tornado.TornadoBatchPrefillPass.cudnnGraphBatchWidth = batchSize;
            dbgQ = cudnnQ; dbgK = cudnnK; dbgV = cudnnV; dbgOut = cudnnOut;
            System.out.printf(
                    "[jllm] prefill attention: cuDNN SDPA (first chunk only), staging %d MiB%n",
                    4L * cudnnElems * Short.BYTES / (1024 * 1024));
        }
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

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        Object keyCache =
                useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            batchPrefillLayer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            batchPrefillLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once
        batchPrefillLayer.transferToDevice(
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

        // ── Attention Block ────────────────────────────────────────────────────
        batchPrefillLayer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        // Q, K, V in ONE tensor-core launch → packed [q|k|v] rows (stride qDim+2*kvDim)
        batchPrefillLayer.task(
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

        // Qwen3: per-head RMS norm on Q and K before RoPE
        batchPrefillLayer.task(
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

        // Register-partitioned flash attention over the packed buffer.
        // The 'dim' parameter doubles as the packed-Q stride base and the
        // attnOutFP16 row width — both are qDim for Qwen3.
        if (useFp16KVCache()) {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);

            if (cudnnAttention) {
                // Q from the packed FP32 QKV buffer, K/V gathered out of the paged cache with
                // the 16:8 group expansion cuDNN's single head count cannot express, then the
                // library call, then the output back into attnOutFP16's [tok][qDim] layout.
                // Same graph as the surrounding JIT tasks: a library task does observe their
                // writes, so no extra graph boundary is needed.
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, cudnnQ, cudnnK, cudnnV, cudnnOut);
                batchPrefillLayer.task(
                        "cudnn_pack_q",
                        CuDnnPrefillAttentionKernels::packQ,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        cudnnQ,
                        nEmbdHead,
                        batchSize,
                        qDim + 2 * kvDim);
                batchPrefillLayer.task(
                        "cudnn_gather_kv",
                        CuDnnPrefillAttentionKernels::gatherKvExpanded,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapBlockTable,
                        cudnnK,
                        cudnnV,
                        nEmbdHead,
                        batchSize,
                        kvDim,
                        gqa,
                        layerIndex,
                        state.kvBlockCfg,
                        state.kvBlockStride);
                batchPrefillLayer.libraryTask(
                        "cudnn_sdpa",
                        CuDnn::sdpaForward,
                        cudnnQ,
                        cudnnK,
                        cudnnV,
                        cudnnOut,
                        1,
                        config.numberOfHeads(),
                        batchSize,
                        batchSize,
                        nEmbdHead,
                        (float) (1.0 / Math.sqrt(nEmbdHead)),
                        true);
                batchPrefillLayer.task(
                        "cudnn_scatter",
                        CuDnnPrefillAttentionKernels::scatterAttnOut,
                        cudnnOut,
                        state.workspace.attnOutFP16,
                        nEmbdHead,
                        batchSize,
                        qDim);
                if (CUDNN_DEBUG && layerIndex == CUDNN_DEBUG_LAYER) {
                    batchPrefillLayer.transferToHost(
                            DataTransferMode.EVERY_EXECUTION,
                            cudnnQ, cudnnK, cudnnV, cudnnOut,
                            state.workspace.qkvResultBatch,
                            state.workspace.wrapKeyCacheFP16,
                            state.workspace.wrapValueCacheFP16,
                            state.workspace.attnOutFP16);
                }
            } else {
            if (CUDNN_DEBUG && layerIndex == CUDNN_DEBUG_LAYER) {
                batchPrefillLayer.transferToHost(
                        DataTransferMode.EVERY_EXECUTION,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.attnOutFP16);
            }
            batchPrefillLayer.task(
                    "batch_attention",
                    packedHalf2Attention
                            ? TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16PackedTilePaged
                            : TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
            }
        } else {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);

            batchPrefillLayer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionFP16OutPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        // Output projection: [M=batch, N=dim, K=qDim]
        if (cublasProjection) {
            // gemmMMA computes C[m][n] = sum_k A[m][k] * B[n][k] with A and B both row-major
            // and B already stored transposed, i.e. C = A * B^T. cuBLAS is column-major, where
            // that same buffer layout reads as C' = B'^T * A' -- hence OP_T on the weight,
            // OP_N on the activation, and (m,n) swapped to (dim, batch).
            // FP16 in, FP32 out, FP32 accumulate: same accumulation precision and the same
            // woOut representation the fused RMS/residual consumer expects.
            batchPrefillLayer.libraryTask(
                    "woProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    dim,
                    paddedBatch,
                    qDim,
                    1.0f,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    qDim,
                    state.workspace.attnOutFP16,
                    qDim,
                    0.0f,
                    state.workspace.woOut,
                    dim);
        } else {
            batchPrefillLayer.task(
                    "woProj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    qDim);
        }

        // ── FFN Block ──────────────────────────────────────────────────────────
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        batchPrefillLayer.task(
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

        batchPrefillLayer
                .task(
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

        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache, valueCache);

        return batchPrefillLayer;
    }

    // @formatter:on

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128; // BM
        int nBlocks = N / 128; // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    static WorkerGrid elementwiseGrid(int n) { // n must be a multiple of 256
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();

        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);

        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // Q/K per-head RMS norm: one nEmbdHead-thread workgroup per (token, head)
        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (nHeads + nHeadKv) * nEmbdHead, nEmbdHead);

        // Split-half RoPE: B*(qDim/2) threads
        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention: B*nHeads workgroups × min(nEmbdHead,128) threads
        int attnLocal = Math.min(nEmbdHead, 128);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        // MMA grids
        WorkerGrid mmaQkvWorker = mmaGrid(paddedBatch, qDim + 2 * kvDim); // fused QKV
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim); // woProj, w2Proj
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim); // fused W1/W3

        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim); // w2Resid
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim); // swiglu

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj", mmaQkvWorker);
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (cudnnAttention) {
                WorkerGrid cudnnWorker = elementwiseGrid(qDim * batchSize);
                scheduler.addWorkerGrid(p + "cudnn_pack_q", cudnnWorker);
                scheduler.addWorkerGrid(p + "cudnn_gather_kv", cudnnWorker);
                scheduler.addWorkerGrid(p + "cudnn_scatter", cudnnWorker);
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            if (!cublasProjection) {
                scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            }
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
}
