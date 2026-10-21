package org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.BatchMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.LlamaQ4_0BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TensorCoreAttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;

// @formatter:off
/**
 * Batched prefill for Llama layers that keep their weights as Q4_0 (and Q4_1 for the {@code
 * ffn_down} tensors {@code llama-quantize} gives more bits), with the projections run by cuBLAS on
 * tensor cores.
 *
 * <p>cuBLAS cannot read Q4_0, so each projection first decodes its weight into one shared FP16
 * scratch with a JIT kernel, and a {@code cublasGemmExFP16FP32} then multiplies the whole prompt
 * chunk by it. This is llama.cpp's cuBLAS route for quantized prompt processing: the decode costs
 * one pass over the weight, the GEMM then runs at tensor-core rate, and the weights stay 4-bit in
 * device memory. Only the scratch is FP16, sized for the largest projection.
 *
 * <p>Per layer, in one graph:
 *
 * <pre>
 *   batch_attn_rms, batch_attn_rms_apply   RMS norm → FP16 activation
 *   qDequant, kDequant, vDequant          Wq, Wk, Wv → scratch, stacked
 *   qkvProj                               cuBLAS → packed [q|k|v] rows
 *   batch_rope_kv, batch_attention        RoPE + KV write, flash attention → FP16
 *   woDequant, woProj                     cuBLAS → woOut
 *   batch_ffn_rms, batch_ffn_rms_apply    x += woOut, RMS norm → FP16
 *   gateDequant, gateProj                 cuBLAS → gate
 *   upDequant, upProj                     cuBLAS → up
 *   swiglu                                silu(gate) * up → FP16
 *   downDequant, downProj, w2Resid        cuBLAS → w2Out, x += w2Out
 * </pre>
 *
 * <p>Gate and up are separate GEMMs on purpose: decoding both into one scratch would need room for
 * {@code 2 * hiddenDim * dim} halves, 940 MB on Llama-3.3-70B, against 470 MB this way.
 *
 * <p>Every per-layer weight reaches a task here, so this graph uploads it and the decode layers
 * bind that copy (see {@code LlamaQ4_0FFNLayersDecode.weightSourceGraphName}).
 */
// @formatter:on
public class LlamaQ4_0LayersBatchPrefillNative implements BatchPrefillTransformerLayerTaskGraphs {

    static final int RMS_LOCAL_SIZE = 256;

    private static final int CUBLAS_OP_N = 0;
    private static final int CUBLAS_OP_T = 1;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int firstLayer;
    private final int endLayer;
    private final HalfFloatArray scratch;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /**
     * Whether Q4_0 projections run as block-scaled int8 tensor-core GEMMs that read the Q4_0 blocks
     * where they lie, the activation quantized per 32-block, instead of a decode to FP16 and
     * cuBLAS. Q4_1 projections keep cuBLAS.
     */
    private final boolean int8;

    private final ByteArray q8;
    private final FloatArray q8Scales;
    private final FloatArray splitPartial;
    private final java.util.Map<String, WorkerGrid> int8Grids = new java.util.LinkedHashMap<>();

    /**
     * Whether the attention runs as FP16 tensor-core GEMMs over the cache gathered per key/value
     * head: scores, a causal softmax, and the weighted values, a few query heads per pass. Needs
     * the FP16 cache.
     */
    private final boolean tcAttention;

    private final int paddedPositions;
    private final int headsPerPass;
    private HalfFloatArray queriesF16;
    private HalfFloatArray keysF16;
    private HalfFloatArray valuesTF16;
    private FloatArray scores;
    private HalfFloatArray probs;
    private FloatArray attnF32;

    public LlamaQ4_0LayersBatchPrefillNative(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize) {
        this(state, weights, config, batchSize, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * The first layer of the range takes the role layer 0 has otherwise, and the key/value cache of
     * {@code state} holds just these layers.
     */
    public LlamaQ4_0LayersBatchPrefillNative(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        if (firstLayer < 0 || endLayer > config.numberOfLayers() || firstLayer >= endLayer) {
            throw new IllegalArgumentException(
                    "layer range ["
                            + firstLayer
                            + ", "
                            + endLayer
                            + ") outside the model's "
                            + config.numberOfLayers()
                            + " layers");
        }
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.scratch = scratchFor(state, config);
        this.int8 =
                batchSize % Int8GemmKernels.I8_BM == 0
                        && org.beehive.jitllm.backend.tornado.TensorCoreSupport.isInt8MmaCapable();
        int widest = Math.max(config.dim(), config.hiddenDim());
        this.q8 = int8 ? TornadoWorkspaces.bytes(batchSize * widest) : null;
        this.q8Scales = int8 ? TornadoWorkspaces.floats(batchSize * widest / 32) : null;
        this.splitPartial =
                int8 ? TornadoWorkspaces.floats(SPLITS_MAX * batchSize * config.dim()) : null;
        this.tcAttention =
                state.usesFp16KeyValueCache()
                        && batchSize % 128 == 0
                        && config.headSize() % 128 == 0
                        && org.beehive.jitllm.backend.tornado.TensorCoreSupport
                                .isTensorCoreCapableBackend();
        this.paddedPositions = (config.contextLength() + 127) / 128 * 128;
        this.headsPerPass = headsPerPass(config.numberOfHeads(), batchSize, paddedPositions);
        if (tcAttention) {
            int rows = batchSize * headsPerPass * paddedPositions;
            queriesF16 = TornadoWorkspaces.halfFloats(batchSize * config.dim());
            keysF16 = TornadoWorkspaces.halfFloats(config.kvDim() * paddedPositions);
            valuesTF16 = TornadoWorkspaces.halfFloats(config.kvDim() * paddedPositions);
            scores = TornadoWorkspaces.floats(rows);
            probs = TornadoWorkspaces.halfFloats(rows);
            attnF32 = TornadoWorkspaces.floats(batchSize * config.dim());
        }
        this.layerITGs =
                IntStream.range(firstLayer, endLayer)
                        .mapToObj(this::createLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    @Override
    public String describeProjections() {
        return int8
                ? "int8 tensor-core GEMMs on Q4_0 weights; cuBLAS FP16 GEMM for Q4_1"
                : "cuBLAS FP16 GEMM (Q4_0 weights decoded to FP16 per projection)";
    }

    @Override
    public String describeNativeLibraries() {
        return "cuBLAS";
    }

    /** Halves the largest projection needs: the stacked Q/K/V, or one FFN matrix. */
    static long scratchElements(LlamaConfiguration config) {
        long qkv = (long) (config.dim() + 2 * config.kvDim()) * config.dim();
        long ffn = (long) config.hiddenDim() * config.dim();
        return Math.max(qkv, ffn);
    }

    /** The state's FP16 weight scratch, allocated here the first time a plan needs it. */
    private static HalfFloatArray scratchFor(LlamaState state, LlamaConfiguration config) {
        long elements = scratchElements(config);
        HalfFloatArray current = state.workspace.weightsF16Scratch;
        if (current == null || current.getSize() < elements) {
            state.workspace.weightsF16Scratch =
                    TornadoWorkspaces.halfFloats(Math.toIntExact(elements));
        }
        return state.workspace.weightsF16Scratch;
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

    // @formatter:off
    private TaskGraph createLayer(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == endLayer - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);

        Object[] intermediates = {
            context,
            state.workspace.attnScaleBatch,
            state.workspace.ffnScaleBatch,
            state.workspace.wrapXbFP16Batch,
            state.workspace.qkvResultBatch,
            keyCache(),
            valueCache(),
            state.workspace.normedXFFNFP16,
            state.workspace.ffnGateResult,
            state.workspace.ffnUpResult,
            state.workspace.attnOutFP16,
            state.workspace.woOut,
            state.workspace.wrapHbFP16Batch,
            state.workspace.w2Out,
            scratch
        };
        if (tcAttention) {
            intermediates = java.util.Arrays.copyOf(intermediates, intermediates.length + 6);
            intermediates[intermediates.length - 6] = queriesF16;
            intermediates[intermediates.length - 5] = keysF16;
            intermediates[intermediates.length - 4] = valuesTF16;
            intermediates[intermediates.length - 3] = scores;
            intermediates[intermediates.length - 2] = probs;
            intermediates[intermediates.length - 1] = attnF32;
        }
        if (int8) {
            intermediates = java.util.Arrays.copyOf(intermediates, intermediates.length + 3);
            intermediates[intermediates.length - 3] = q8;
            intermediates[intermediates.length - 2] = q8Scales;
            intermediates[intermediates.length - 1] = splitPartial;
        }
        if (layerIndex == firstLayer) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, intermediates);
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            // Every execution: a lease change rewrites the table.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred, intermediates);
            layer.consumeFromDevice(
                    pred,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapBlockTable);
        }

        state.workspace.packedRepack.upload(
                layer,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex],
                weights.wkLayered[layerIndex],
                weights.wvLayered[layerIndex],
                weights.woLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex],
                weights.w2Layered[layerIndex],
                weights.w3Layered[layerIndex],
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int qkvDim = dim + 2 * kvDim;
        // The stage's cache holds only its own layers.
        int kvLayer = layerIndex - firstLayer;

        // ── Attention ──────────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        if (int8(layerIndex, weights.wqLayered, weights.wkLayered, weights.wvLayered)) {
            // Three GEMMs over one quantized activation, each into its slot of the packed row.
            quantize(layer, "qkvQ8", state.workspace.wrapXbFP16Batch, dim);
            int8Gemm(
                    layer,
                    "qProj",
                    weights.wqLayered[layerIndex],
                    state.workspace.qkvResultBatch,
                    dim,
                    dim,
                    qkvDim,
                    0,
                    false);
            int8Gemm(
                    layer,
                    "kProj",
                    weights.wkLayered[layerIndex],
                    state.workspace.qkvResultBatch,
                    kvDim,
                    dim,
                    qkvDim,
                    dim,
                    false);
            int8Gemm(
                    layer,
                    "vProj",
                    weights.wvLayered[layerIndex],
                    state.workspace.qkvResultBatch,
                    kvDim,
                    dim,
                    qkvDim,
                    dim + kvDim,
                    false);
        } else {
            // Wq, Wk, Wv stacked in the scratch: one GEMM writes packed [q|k|v] rows.
            dequantize(layer, "qDequant", weights.wqLayered[layerIndex], 0);
            dequantize(layer, "kDequant", weights.wkLayered[layerIndex], dim * dim);
            dequantize(layer, "vDequant", weights.wvLayered[layerIndex], (dim + kvDim) * dim);
            gemm(
                    layer,
                    "qkvProj",
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    qkvDim,
                    dim);
        }

        if (state.usesFp16KeyValueCache()) {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
            if (tcAttention) {
                tensorCoreAttention(layer, kvLayer);
            } else {
                layer.task(
                        "batch_attention",
                        // The tiled variant the FP16 layers take under the same policy flag.
                        state.executionPolicy().packedHalf2Attention()
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
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        kvLayer,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        dim);
            }
        } else {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionFP16OutPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        }

        if (int8(layerIndex, weights.woLayered)) {
            quantize(layer, "woQ8", state.workspace.attnOutFP16, dim);
            int8Gemm(
                    layer,
                    "woProj",
                    weights.woLayered[layerIndex],
                    state.workspace.woOut,
                    dim,
                    dim,
                    dim,
                    0,
                    true);
        } else {
            dequantize(layer, "woDequant", weights.woLayered[layerIndex], 0);
            gemm(layer, "woProj", state.workspace.attnOutFP16, state.workspace.woOut, dim, dim);
        }

        // ── Feed-forward ───────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        if (int8(layerIndex, weights.w1Layered, weights.w3Layered)) {
            quantize(layer, "ffnQ8", state.workspace.normedXFFNFP16, dim);
            int8Gemm(
                    layer,
                    "gateProj",
                    weights.w1Layered[layerIndex],
                    state.workspace.ffnGateResult,
                    hidDim,
                    dim,
                    hidDim,
                    0,
                    false);
            int8Gemm(
                    layer,
                    "upProj",
                    weights.w3Layered[layerIndex],
                    state.workspace.ffnUpResult,
                    hidDim,
                    dim,
                    hidDim,
                    0,
                    false);
        } else {
            dequantize(layer, "gateDequant", weights.w1Layered[layerIndex], 0);
            gemm(
                    layer,
                    "gateProj",
                    state.workspace.normedXFFNFP16,
                    state.workspace.ffnGateResult,
                    hidDim,
                    dim);
            dequantize(layer, "upDequant", weights.w3Layered[layerIndex], 0);
            gemm(
                    layer,
                    "upProj",
                    state.workspace.normedXFFNFP16,
                    state.workspace.ffnUpResult,
                    hidDim,
                    dim);
        }

        layer.task(
                "swiglu",
                LlamaQ4_0BatchPrefillKernels::swiGLUFP16,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.ffnGateResult,
                state.workspace.ffnUpResult,
                batchSize * hidDim);

        if (int8(layerIndex, weights.w2Layered)) {
            quantize(layer, "downQ8", state.workspace.wrapHbFP16Batch, hidDim);
            int8Gemm(
                    layer,
                    "downProj",
                    weights.w2Layered[layerIndex],
                    state.workspace.w2Out,
                    dim,
                    hidDim,
                    dim,
                    0,
                    true);
        } else {
            dequantize(layer, "downDequant", weights.w2Layered[layerIndex], 0);
            gemm(
                    layer,
                    "downProj",
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out,
                    dim,
                    hidDim);
        }
        layer.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        return layer;
    }

    // @formatter:on

    /** The most query heads per attention pass whose scores fit in 64 MB, a divisor of all. */
    private static int headsPerPass(int heads, int rows, int positions) {
        long budget = 64L << 20;
        for (int h = heads; h >= 1; h--) {
            if (heads % h == 0 && (long) rows * h * positions * 4 <= budget) {
                return h;
            }
        }
        return 1;
    }

    // @formatter:off
    /**
     * The chunk's attention as FP16 tensor-core GEMMs: the cache gathered per key/value head, the
     * queries in FP16, then per pass of {@link #headsPerPass} query heads the scores (each head
     * against its key/value head), the causal softmax, and the weighted values into the FP32
     * output; finally the output in FP16 for the output projection. Positions past the chunk's last
     * one are neither scored nor read.
     */
    // @formatter:on
    private void tensorCoreAttention(TaskGraph layer, int kvLayer) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hs = config.headSize();
        int heads = config.numberOfHeads();
        int kvMul = config.kvMul();
        int npad = paddedPositions;
        String graph = layer.getTaskGraphName() + ".";
        layer.task(
                "attn_gather",
                TensorCoreAttentionKernels::gatherPagedKeyValues,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapKeyCacheFP16,
                state.workspace.wrapValueCacheFP16,
                keysF16,
                valuesTF16,
                kvLayer,
                kvDim,
                hs,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                npad);
        int8Grids.put(graph + "attn_gather", elementwise(npad * kvDim));
        layer.task(
                "attn_q16",
                TensorCoreAttentionKernels::queriesToHalf,
                context,
                state.workspace.qkvResultBatch,
                queriesF16,
                dim,
                dim + 2 * kvDim,
                batchSize * dim);
        int8Grids.put(graph + "attn_q16", elementwise(batchSize * dim));
        int hp = headsPerPass;
        for (int pass = 0; pass * hp < heads; pass++) {
            int z0 = pass * hp;
            String sTask = "attn_scores_" + pass;
            layer.task(
                    sTask,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    queriesF16,
                    keysF16,
                    scores,
                    batchSize,
                    npad,
                    hs,
                    dim,
                    hs,
                    hp * npad,
                    hs,
                    npad * hs,
                    npad,
                    state.workspace.batchStartPosHolder,
                    1,
                    0,
                    z0,
                    z0,
                    kvMul,
                    0);
            int8Grids.put(graph + sTask, mmaGrid(batchSize, npad, hp));
            String pTask = "attn_softmax_" + pass;
            layer.task(
                    pTask,
                    BatchMmaKernels::causalSoftmax,
                    context,
                    state.workspace.batchStartPosHolder,
                    scores,
                    probs,
                    hp,
                    npad,
                    (float) (1.0 / Math.sqrt(hs)));
            WorkerGrid1D softmax = new WorkerGrid1D(batchSize * hp * BatchMmaKernels.GROUP);
            softmax.setLocalWork(BatchMmaKernels.GROUP, 1, 1);
            int8Grids.put(graph + pTask, softmax);
            String aTask = "attn_values_" + pass;
            layer.task(
                    aTask,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    probs,
                    valuesTF16,
                    attnF32,
                    batchSize,
                    hs,
                    npad,
                    hp * npad,
                    npad,
                    dim,
                    npad,
                    hs * npad,
                    hs,
                    state.workspace.batchStartPosHolder,
                    0,
                    1,
                    0,
                    z0,
                    kvMul,
                    z0);
            int8Grids.put(graph + aTask, mmaGrid(batchSize, hs, hp));
        }
        layer.task(
                "attn_out16",
                BatchMmaKernels::toHalf,
                context,
                attnF32,
                state.workspace.attnOutFP16,
                batchSize * dim);
        int8Grids.put(graph + "attn_out16", elementwise(batchSize * dim));
    }

    private static WorkerGrid mmaGrid(int m, int n, int batches) {
        WorkerGrid3D grid = new WorkerGrid3D((m / 128) * 256, n / 128, batches);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    /** Whether layer {@code l}'s projections all take the int8 GEMM: every one Q4_0. */
    private boolean int8(int l, TornadoTensor[]... tensors) {
        if (!int8) {
            return false;
        }
        for (TornadoTensor[] t : tensors) {
            if (t[l].dataType() != DataType.Q4_0) {
                return false;
            }
        }
        return true;
    }

    /** The chunk's {@code k}-wide FP16 rows in int8 blocks, for the GEMMs after. */
    private void quantize(TaskGraph layer, String name, HalfFloatArray x, int k) {
        int total = batchSize * k;
        layer.task(
                name,
                Int8GemmKernels::quantizeActivationsQ8WarpFP16,
                context,
                x,
                q8,
                q8Scales,
                total);
        int8Grids.put(layer.getTaskGraphName() + "." + name, elementwise(total));
    }

    private static final int SPLITS_MAX = 8;

    /** The streaming multiprocessors of the device this plan is built for; 0 when unknown. */
    private static int smCount() {
        return org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
    }

    /** K splits that best fill the device with 128 x 128 tiles, as the other int8 prefills. */
    private int gemmSplits(int outputs, int k) {
        int sms = smCount();
        if (sms <= 0) {
            return 1;
        }
        int tiles = (batchSize / Int8GemmKernels.I8_BM) * (outputs / Int8GemmKernels.I8_BN);
        int rounds = k / Int8GemmKernels.I8_BK;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= SPLITS_MAX && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return bestCost * 10 <= unsplit * 9 ? best : 1;
    }

    /**
     * {@code out[r * ldo + colOffset + c] = x8[r] . w[c]} over the chunk, the Q4_0 blocks read in
     * place; {@code split} lets a plain output cut K for fuller tiles, finished by a reduce.
     */
    private void int8Gemm(
            TaskGraph layer,
            String name,
            TornadoTensor w,
            FloatArray out,
            int n,
            int k,
            int ldo,
            int colOffset,
            boolean split) {
        int splits = split ? gemmSplits(n, k) : 1;
        FloatArray partial = splits > 1 ? splitPartial : out;
        layer.task(
                name,
                w.isPackedQ4()
                        ? Int8GemmKernels::gemmInt8Q4_0Packed
                        : Int8GemmKernels::gemmInt8Q4_0,
                context,
                q8,
                q8Scales,
                w.asByteArray(),
                out,
                out,
                batchSize,
                n,
                k,
                Int8GemmKernels.EPILOGUE_STORE,
                partial,
                splits,
                state.workspace.batchStartPosHolder,
                ldo,
                colOffset);
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        (batchSize / Int8GemmKernels.I8_BM) * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN * splits);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        String graph = layer.getTaskGraphName() + ".";
        int8Grids.put(graph + name, grid);
        if (splits > 1) {
            layer.task(
                    name + "Reduce",
                    Int8GemmKernels::reduceSplitsQ8_0,
                    context,
                    splitPartial,
                    out,
                    batchSize * n,
                    splits,
                    Int8GemmKernels.EPILOGUE_STORE,
                    state.workspace.batchStartPosHolder,
                    n);
            int8Grids.put(graph + name + "Reduce", elementwise(batchSize * n));
        }
    }

    /** Decodes {@code w} into the scratch at {@code offset}, by the tensor's own representation. */
    private void dequantize(TaskGraph layer, String name, TornadoTensor w, int offset) {
        if (w.isPackedQ4()) {
            // A packed weight has only the int8 GEMM.
            throw org.beehive.jitllm.backend.tornado.tensor.PackedWeights.noPackedKernel(
                    "llama batch-prefill task '" + name + "'");
        }
        DataType type = w.dataType();
        if (type == DataType.Q4_0) {
            layer.task(
                    name,
                    Gemma4BatchPrefillKernels::dequantizeQ4_0ToFP16,
                    context,
                    w.asByteArray(),
                    scratch,
                    offset);
        } else if (type == DataType.Q4_1) {
            layer.task(
                    name,
                    Gemma4BatchPrefillKernels::dequantizeQ4_1ToFP16,
                    context,
                    w.asByteArray(),
                    scratch,
                    offset);
        } else {
            throw new UnsupportedOperationException(
                    "the Llama Q4_0 batched prefill decodes Q4_0 and Q4_1 weights, not " + type);
        }
    }

    // @formatter:off
    /**
     * {@code out[rows][n] = x[rows][k] · Wᵀ} with W the decoded {@code [n][k]} weight in the
     * scratch. In cuBLAS's column-major terms that is {@code C = op_T(W) · X}: the weight is A,
     * transposed, and the activation B as it is.
     */
    // @formatter:on
    private void gemm(
            TaskGraph layer, String name, HalfFloatArray x, FloatArray out, int n, int k) {
        layer.libraryTask(
                name,
                CuBlas::cublasGemmExFP16FP32,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                batchSize,
                k,
                1.0f,
                scratch,
                k,
                x,
                k,
                0.0f,
                out,
                n);
    }

    private static WorkerGrid elementwise(int n) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSize = config.headSize();

        WorkerGrid rms =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid rmsApply = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid rope = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int attnLocal = Math.min(headSize, 128);
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        for (int i = firstLayer; i < endLayer; i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApply);
            if (!int8(i, weights.wqLayered, weights.wkLayered, weights.wvLayered)) {
                scheduler.addWorkerGrid(p + "qDequant", elementwise(dim * dim));
                scheduler.addWorkerGrid(p + "kDequant", elementwise(kvDim * dim));
                scheduler.addWorkerGrid(p + "vDequant", elementwise(kvDim * dim));
            }
            scheduler.addWorkerGrid(p + "batch_rope_kv", rope);
            if (!tcAttention) {
                scheduler.addWorkerGrid(p + "batch_attention", attention);
            }
            if (!int8(i, weights.woLayered)) {
                scheduler.addWorkerGrid(p + "woDequant", elementwise(dim * dim));
            }
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApply);
            if (!int8(i, weights.w1Layered, weights.w3Layered)) {
                scheduler.addWorkerGrid(p + "gateDequant", elementwise(hidDim * dim));
                scheduler.addWorkerGrid(p + "upDequant", elementwise(hidDim * dim));
            }
            scheduler.addWorkerGrid(p + "swiglu", elementwise(batchSize * hidDim));
            if (!int8(i, weights.w2Layered)) {
                scheduler.addWorkerGrid(p + "downDequant", elementwise(dim * hidDim));
            }
            scheduler.addWorkerGrid(p + "w2Resid", elementwise(batchSize * dim));
        }
        int8Grids.forEach(scheduler::addWorkerGrid);
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }
}
