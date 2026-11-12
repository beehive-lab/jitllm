package org.beehive.jitllm.backend.tornado.layers.type.fp16;

import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4AttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Gemma4FP16FFNLayers: FP16 transformer-layer task graphs for the Gemma 4 architecture.
 *
 * <p>Gemma 4's layers differ enough from the "Llama-like" models that nothing here is fused the way
 * {@code Qwen3FP16FFNLayers} is -- each layer carries its own Q/K-norm and a "sandwich" of pre/post
 * normalization around both attention and FFN, attention head dimensions and RoPE tables differ
 * between sliding-window and full-attention layers (and are baked into each layer's task graph as
 * compile-time constants -- see {@link Gemma4Configuration#headDim}), some layers reuse an earlier
 * layer's KV cache instead of computing their own, the FFN uses GeGLU, and every layer mixes in a
 * per-layer embedding (PLE) contribution. See {@link InferenceCore#forwardJavaGemma4} for the
 * reference computation each task mirrors.
 *
 * <p>Layer 0's task graph additionally carries one-time-per-token setup that the reference
 * implementation performs before the layer loop: scaling the token embedding by {@code sqrt(dim)},
 * and computing the per-layer-embedding inputs ({@code perLayerInputs}) from the per-layer model
 * projection and the (host-gathered) per-layer token embedding row -- see {@link
 * #appendPLESetupTasks}.
 */
public class Gemma4FP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Gemma4TornadoWeights, Gemma4Configuration> {

    /**
     * Local memory size for per-head Q/K/V-norm reductions; must evenly divide both head dimensions
     * (256, 512).
     */
    private static final int HEAD_NORM_LOCAL_SIZE = 64;

    /**
     * Lanes' worth of reduction scratch the attention kernel allocates.
     *
     * <p>The invariant is {@code ATTENTION_LOCAL_SIZE >= } the launched workgroup size, not
     * equality with it. It is passed as the kernel's {@code localMemSize} and the kernel derives
     * every bound from {@code context.localGroupSizeX}, so a value above the launch over-allocates
     * shared memory and a value below it corrupts the reduction. {@code createAttentionWorker}
     * picks {@code min(headDim, 64)} for the single-pass kernel, so 256 is slack, not a match — an
     * earlier comment here claimed they had to be equal, which would have invited someone to lower
     * this to 64 and under-size the scratch for the split kernel, which does launch at 256.
     */
    private static final int ATTENTION_LOCAL_SIZE = 256;

    private final Gemma4State gemma4State;
    private final int nHead;
    private final int nHeadKv;
    private final int kvMul;
    private final int dim;
    private final int nEmbdPerLayer;
    private final int perLayerTotal;
    private final float embedScale;
    private final float perLayerTokEmbedScale;
    private final float perLayerProjScale;
    private final float perLayerInputScale;
    private final boolean batchedPlan;
    private final boolean fp16KeyValue;

    public Gemma4FP16FFNLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            SchedulerType schedulerType) {
        this(state, weights, config, schedulerType, false);
    }

    /**
     * @param batchedPlan whether these are the decode layers of the batched prefill/decode plan:
     *     layer 0 then reads its activation and the key/value caches from {@code decodeActivation},
     *     and every layer the projections, RoPE tables and per-layer projection norm the
     *     batch-prefill graphs already uploaded
     */
    public Gemma4FP16FFNLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            SchedulerType schedulerType,
            boolean batchedPlan) {
        super(state, weights, config, schedulerType);
        this.batchedPlan = batchedPlan;
        this.gemma4State = state;
        this.fp16KeyValue = state.usesFp16KeyValueCache();
        this.nHead = config.numberOfHeads();
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.kvMul = config.kvMul();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerTokEmbedScale = (float) Math.sqrt(nEmbdPerLayer);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));
        setupFFNLayers();
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                  TASK GRAPH
    // ═══════════════════════════════════════════════════════════════════════════════════

    /** How many slices the window is cut into, or 1 to run the single-pass kernel. */
    private int attentionSplits() {
        return config.attentionSplits();
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var taskGraphName = GraphNames.layer(layerIndex);
        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int qDim = nHead * headDim;
        final int kvDim = nHeadKv * headDim;
        final int ffnLen = config.feedForwardLength(layerIndex);
        final int cacheBaseOffset = gemma4State.cacheLayerBaseOffset[layerIndex];
        final int windowSize = isSwa ? config.slidingWindowSize() : config.contextLength();
        final var freqCisReal =
                (isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull).asFloatArray();
        final var freqCisImag =
                (isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull).asFloatArray();
        final int peOffset = layerIndex * nEmbdPerLayer;

        var unifiedLayer = new TaskGraph(taskGraphName);
        if (batchedPlan) {
            // A graph that consumes anything by name takes nothing through an unnamed consume.
            unifiedLayer.consumeFromDevice(
                    layerIndex == 0
                            ? GraphNames.DECODE_ACTIVATION
                            : GraphNames.layer(layerIndex - 1),
                    gemma4State.workspace.wrapX);
            // The projections the batch-prefill graph of this layer already uploaded; K and V
            // only where the layer has its own, as there.
            java.util.List<Object> shared = new java.util.ArrayList<>();
            shared.add(weights.wqLayered[layerIndex].asHalfFloatArray());
            shared.add(weights.woLayered[layerIndex].asHalfFloatArray());
            shared.add(weights.w1Layered[layerIndex].asHalfFloatArray());
            shared.add(weights.w3Layered[layerIndex].asHalfFloatArray());
            shared.add(weights.w2Layered[layerIndex].asHalfFloatArray());
            if (hasOwnKv) {
                shared.add(weights.wkLayered[layerIndex].asHalfFloatArray());
                shared.add(weights.wvLayered[layerIndex].asHalfFloatArray());
            } else {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        weights.wkLayered[layerIndex].asHalfFloatArray(),
                        weights.wvLayered[layerIndex].asHalfFloatArray());
            }
            unifiedLayer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex), shared.toArray());
        } else {
            unifiedLayer.consumeFromDevice(gemma4State.workspace.wrapX);
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    weights.w2Layered[layerIndex].asHalfFloatArray());
        }
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.attnQNorm[layerIndex].asFloatArray(),
                weights.attnKNorm[layerIndex].asFloatArray(),
                weights.attnPostNorm[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                weights.perLayerInpGate[layerIndex].asHalfFloatArray(),
                weights.perLayerProj[layerIndex].asHalfFloatArray(),
                weights.perLayerPostNorm[layerIndex].asFloatArray());
        if (weights.layerOutputScale[layerIndex] != null) {
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        if (layerIndex == 0) {
            appendPLESetupTasks(unifiedLayer);
        }

        // ═══════════════════════════════════ ATTENTION ═══════════════════════════════════
        unifiedLayer.task(
                "attn_norm_reduce",
                rmsReduceKernel(),
                context,
                gemma4State.workspace.temp,
                gemma4State.workspace.wrapX,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "attn_norm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.workspace.temp,
                    dim,
                    config.rmsNormEps());
        }
        unifiedLayer.task(
                "attn_norm_apply",
                Gemma4Kernels::applyRmsNorm,
                context,
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                gemma4State.workspace.temp,
                dim);

        unifiedLayer.task(
                "q_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapQ,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                dim,
                qDim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task(
                "q_norm",
                Gemma4Kernels::rmsNormPerHead,
                context,
                gemma4State.workspace.wrapQ,
                weights.attnQNorm[layerIndex].asFloatArray(),
                nHead,
                headDim,
                HEAD_NORM_LOCAL_SIZE,
                config.rmsNormEps());

        if (hasOwnKv) {
            unifiedLayer.task(
                    "k_proj",
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapK,
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    dim,
                    kvDim,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
            unifiedLayer.task(
                    "k_norm",
                    Gemma4Kernels::rmsNormPerHead,
                    context,
                    gemma4State.workspace.wrapK,
                    weights.attnKNorm[layerIndex].asFloatArray(),
                    nHeadKv,
                    headDim,
                    HEAD_NORM_LOCAL_SIZE,
                    config.rmsNormEps());
            unifiedLayer.task(
                    "v_proj",
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapV,
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    dim,
                    kvDim,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
            unifiedLayer.task(
                    "v_norm",
                    Gemma4Kernels::rmsNormPerHeadNoWeight,
                    context,
                    gemma4State.workspace.wrapV,
                    nHeadKv,
                    headDim,
                    HEAD_NORM_LOCAL_SIZE,
                    config.rmsNormEps());
            if (fp16KeyValue) {
                unifiedLayer.task(
                        "rope_and_cache",
                        Gemma4AttentionKernels::ropeNeoxRotateAndCacheCopyFP16,
                        context,
                        gemma4State.workspace.positionHolder,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapK,
                        gemma4State.workspace.wrapV,
                        gemma4State.workspace.wrapKeyCacheFP16,
                        gemma4State.workspace.wrapValueCacheFP16,
                        freqCisReal,
                        freqCisImag,
                        nHeadKv,
                        headDim,
                        kvDim,
                        cacheBaseOffset);
            } else {
                unifiedLayer.task(
                        "rope_and_cache",
                        Gemma4Kernels::ropeNeoxRotateAndCacheCopy,
                        context,
                        gemma4State.workspace.positionHolder,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapK,
                        gemma4State.workspace.wrapV,
                        gemma4State.workspace.wrapKeyCache,
                        gemma4State.workspace.wrapValueCache,
                        freqCisReal,
                        freqCisImag,
                        nHeadKv,
                        headDim,
                        kvDim,
                        cacheBaseOffset);
            }
        } else {
            unifiedLayer.task(
                    "rope_q_only",
                    Gemma4Kernels::ropeNeoxRotateQOnly,
                    context,
                    gemma4State.workspace.positionHolder,
                    gemma4State.workspace.wrapQ,
                    freqCisReal,
                    freqCisImag,
                    headDim);
        }

        int splits = attentionSplits();
        if (fp16KeyValue) {
            unifiedLayer.task(
                    "attention_split",
                    Gemma4Kernels::attentionWithSlidingWindowSplitFP16,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCacheFP16,
                    gemma4State.workspace.wrapValueCacheFP16,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapAttSplit,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    splits,
                    ATTENTION_LOCAL_SIZE);
            unifiedLayer.task(
                    "attention_combine",
                    Gemma4Kernels::combineSplitKVAttentionPerElement,
                    context,
                    gemma4State.workspace.wrapAttSplit,
                    gemma4State.workspace.wrapXb,
                    nHead,
                    headDim,
                    splits);
        } else if (splits > 1) {
            unifiedLayer.task(
                    "attention_split",
                    Gemma4Kernels::attentionWithSlidingWindowSplit,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCache,
                    gemma4State.workspace.wrapValueCache,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapAttSplit,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    splits,
                    ATTENTION_LOCAL_SIZE);
            unifiedLayer.task(
                    "attention_combine",
                    Gemma4Kernels::combineSplitKVAttentionPerElement,
                    context,
                    gemma4State.workspace.wrapAttSplit,
                    gemma4State.workspace.wrapXb,
                    nHead,
                    headDim,
                    splits);
        } else {
            unifiedLayer.task(
                    "attention",
                    Gemma4Kernels::attentionWithSlidingWindowParallel,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCache,
                    gemma4State.workspace.wrapValueCache,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapAtt,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    ATTENTION_LOCAL_SIZE);
        }

        unifiedLayer.task(
                "wo_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapXb2,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                qDim,
                dim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task(
                "post_attn_reduce",
                rmsReduceKernel(),
                context,
                gemma4State.workspace.tempPostAttn,
                gemma4State.workspace.wrapXb2,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "post_attn_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.workspace.tempPostAttn,
                    dim,
                    config.rmsNormEps());
        }
        unifiedLayer.task(
                "post_attn_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapXb2,
                weights.attnPostNorm[layerIndex].asFloatArray(),
                gemma4State.workspace.tempPostAttn,
                dim);

        // ═══════════════════════════════════════ FFN ═════════════════════════════════════
        unifiedLayer.task(
                "ffn_norm_reduce",
                rmsReduceKernel(),
                context,
                gemma4State.workspace.tempFFN,
                gemma4State.workspace.wrapX,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_norm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.workspace.tempFFN,
                    dim,
                    config.rmsNormEps());
        }
        unifiedLayer.task(
                "ffn_norm_apply",
                Gemma4Kernels::applyRmsNorm,
                context,
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapX,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                gemma4State.workspace.tempFFN,
                dim);

        unifiedLayer.task(
                "ffn_gate_up",
                Gemma4Kernels::fusedGateUpGeGLU,
                context,
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapHb,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                dim,
                ffnLen,
                LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task(
                "ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapHb,
                gemma4State.workspace.wrapXb2,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                ffnLen,
                dim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task(
                "post_ffn_reduce",
                rmsReduceKernel(),
                context,
                gemma4State.workspace.tempPostFfn,
                gemma4State.workspace.wrapXb2,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "post_ffn_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.workspace.tempPostFfn,
                    dim,
                    config.rmsNormEps());
        }
        unifiedLayer.task(
                "post_ffn_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapXb2,
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                gemma4State.workspace.tempPostFfn,
                dim);

        // ═══════════════════════════ PER-LAYER EMBEDDING (PLE) ═══════════════════════════
        unifiedLayer.task(
                "ple_gate_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapPerLayerGate,
                weights.perLayerInpGate[layerIndex].asHalfFloatArray(),
                dim,
                nEmbdPerLayer,
                LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task(
                "ple_gate_gelu_mul",
                Gemma4Kernels::pleGateGeluMul,
                context,
                gemma4State.workspace.wrapPerLayerGate,
                gemma4State.workspace.wrapPerLayerInputs,
                peOffset,
                nEmbdPerLayer);
        unifiedLayer.task(
                "ple_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapPerLayerGate,
                gemma4State.workspace.wrapPerLayerOut,
                weights.perLayerProj[layerIndex].asHalfFloatArray(),
                nEmbdPerLayer,
                dim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task(
                "ple_post_reduce",
                rmsReduceKernel(),
                context,
                gemma4State.workspace.tempPostPle,
                gemma4State.workspace.wrapPerLayerOut,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ple_post_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.workspace.tempPostPle,
                    dim,
                    config.rmsNormEps());
        }
        unifiedLayer.task(
                "ple_post_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapPerLayerOut,
                weights.perLayerPostNorm[layerIndex].asFloatArray(),
                gemma4State.workspace.tempPostPle,
                dim);

        if (weights.layerOutputScale[layerIndex] != null) {
            unifiedLayer.task(
                    "layer_output_scale",
                    Gemma4Kernels::scaleInPlaceFromTensor,
                    context,
                    gemma4State.workspace.wrapX,
                    weights.layerOutputScale[layerIndex].asFloatArray(),
                    dim);
        }

        unifiedLayer.persistOnDevice(gemma4State.workspace.wrapX);
        return unifiedLayer;
    }

    /**
     * One-time-per-token setup tasks, prepended to layer 0's graph: scales the token embedding by
     * {@code sqrt(dim)} (Gemma4 scales embeddings on input -- the generic {@link Activation} task
     * graph that produced {@code wrapX} doesn't know about this), then computes the per-layer
     * embedding inputs from the per-layer model projection and the (host-gathered) per-token
     * per-layer-token-embedding row. Mirrors steps 1-2 of {@link InferenceCore#forwardJavaGemma4}.
     */
    private void appendPLESetupTasks(TaskGraph unifiedLayer) {
        unifiedLayer.task(
                "scale_embedding",
                TransformerComputeKernels::scaleInPlace,
                context,
                gemma4State.workspace.wrapX,
                embedScale,
                dim);

        unifiedLayer.task(
                "ple_model_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapPerLayerProjScratch,
                weights.perLayerModelProj.asHalfFloatArray(),
                dim,
                perLayerTotal,
                LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task(
                "ple_proj_scale_norm",
                Gemma4Kernels::pleProjScaleAndNormalize,
                context,
                gemma4State.workspace.wrapPerLayerProjScratch,
                weights.perLayerProjNorm.asFloatArray(),
                nEmbdPerLayer,
                HEAD_NORM_LOCAL_SIZE,
                perLayerProjScale,
                config.rmsNormEps());
        unifiedLayer.task(
                "ple_merge",
                Gemma4Kernels::addAndScale,
                context,
                gemma4State.workspace.wrapPerLayerInputs,
                gemma4State.workspace.wrapPerLayerProjScratch,
                gemma4State.workspace.wrapPerLayerTokenEmbedRow,
                perLayerInputScale,
                perLayerTotal);
    }

    /** The key cache in the representation the state allocated. */
    private Object keyCache() {
        return fp16KeyValue
                ? gemma4State.workspace.wrapKeyCacheFP16
                : gemma4State.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state allocated. */
    private Object valueCache() {
        return fp16KeyValue
                ? gemma4State.workspace.wrapValueCacheFP16
                : gemma4State.workspace.wrapValueCache;
    }

    /** The first layer that attends with a sliding window, or the first that attends fully. */
    private int firstLayerOfKind(boolean isSwa) {
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSwa(l) == isSwa) {
                return l;
            }
        }
        throw new IllegalStateException("no layer attends with isSwa=" + isSwa);
    }

    /** Configure data transfers for first and subsequent layers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    gemma4State.workspace.positionHolder,
                    gemma4State.workspace.wrapPerLayerTokenEmbedRow,
                    gemma4State.workspace.temp,
                    gemma4State.workspace.tempFFN,
                    gemma4State.workspace.tempPostAttn,
                    gemma4State.workspace.tempPostFfn,
                    gemma4State.workspace.tempPostPle);
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, weights.perLayerModelProj.asHalfFloatArray());
            if (batchedPlan) {
                // Uploaded by the batch-prefill graphs: the norm by the first, each RoPE pair by
                // the first layer of its kind; the cache the prefill filled, relayed by the
                // decode activation.
                unifiedLayer.consumeFromDevice(
                        "batchPrefillLayer_0", weights.perLayerProjNorm.asFloatArray());
                unifiedLayer.consumeFromDevice(
                        GraphNames.batchPrefillLayer(firstLayerOfKind(true)),
                        weights.freqCisRealSwa.asFloatArray(),
                        weights.freqCisImagSwa.asFloatArray());
                unifiedLayer.consumeFromDevice(
                        GraphNames.batchPrefillLayer(firstLayerOfKind(false)),
                        weights.freqCisRealFull.asFloatArray(),
                        weights.freqCisImagFull.asFloatArray());
                unifiedLayer.consumeFromDevice(
                        GraphNames.DECODE_ACTIVATION, keyCache(), valueCache());
            } else {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        weights.perLayerProjNorm.asFloatArray(),
                        weights.freqCisRealSwa.asFloatArray(),
                        weights.freqCisImagSwa.asFloatArray(),
                        weights.freqCisRealFull.asFloatArray(),
                        weights.freqCisImagFull.asFloatArray());
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, keyCache(), valueCache());
            }
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXb2,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapK,
                    gemma4State.workspace.wrapV,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapPerLayerInputs,
                    gemma4State.workspace.wrapPerLayerProjScratch,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut);
        } else if (batchedPlan) {
            unifiedLayer.consumeFromDevice(
                    GraphNames.layer(layerIndex - 1),
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXb2,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapK,
                    gemma4State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapPerLayerInputs,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut,
                    gemma4State.workspace.positionHolder);
        } else {
            unifiedLayer.consumeFromDevice(
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXb2,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapK,
                    gemma4State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapPerLayerInputs,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut,
                    gemma4State.workspace.positionHolder);
        }
        return unifiedLayer;
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                 GRID SCHEDULER
    // ═══════════════════════════════════════════════════════════════════════════════════

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker =
                WorkerGridFactory.createRmsNormWorker(dim, gemma4State.localSize);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);
        WorkerGrid dimElementWiseWorker =
                WorkerGridFactory.genericWorker(dim, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid woProjWorker =
                WorkerGridFactory.genericWorker(
                        dim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid pleGateProjWorker =
                WorkerGridFactory.genericWorker(
                        nEmbdPerLayer * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid pleGateGeluWorker =
                WorkerGridFactory.genericWorker(nEmbdPerLayer, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // === Layer-0 PLE setup ===
        gridScheduler.addWorkerGrid("layer_0.scale_embedding", dimElementWiseWorker);
        gridScheduler.addWorkerGrid(
                "layer_0.ple_model_proj",
                WorkerGridFactory.genericWorker(
                        perLayerTotal * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC));
        gridScheduler.addWorkerGrid(
                "layer_0.ple_proj_scale_norm",
                WorkerGridFactory.genericWorker(
                        config.numberOfLayers() * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE));
        gridScheduler.addWorkerGrid(
                "layer_0.ple_merge",
                WorkerGridFactory.genericWorker(perLayerTotal, LOCAL_WORK_GROUP_SIZE_ALLOC));

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String prefix = GraphNames.layer(i) + ".";
            int headDim = config.headDim(i);
            boolean hasOwnKv = config.hasOwnKv(i);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(i);

            WorkerGrid headNormWorker =
                    WorkerGridFactory.genericWorker(
                            nHead * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid kvHeadNormWorker =
                    WorkerGridFactory.genericWorker(
                            nHeadKv * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(nHead, headDim);
            WorkerGrid attentionWorker = WorkerGridFactory.createAttentionWorker(nHead, headDim);
            WorkerGrid qProjWorker =
                    WorkerGridFactory.genericWorker(
                            qDim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
            WorkerGrid kvProjWorker =
                    WorkerGridFactory.genericWorker(
                            kvDim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
            WorkerGrid ffnGateUpWorker =
                    WorkerGridFactory.genericWorker(
                            ffnLen * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);

            gridScheduler.addWorkerGrid(prefix + "attn_norm_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "attn_norm_apply", dimElementWiseWorker);
            gridScheduler.addWorkerGrid(prefix + "q_proj", qProjWorker);
            gridScheduler.addWorkerGrid(prefix + "q_norm", headNormWorker);
            if (hasOwnKv) {
                gridScheduler.addWorkerGrid(prefix + "k_proj", kvProjWorker);
                gridScheduler.addWorkerGrid(prefix + "k_norm", kvHeadNormWorker);
                gridScheduler.addWorkerGrid(prefix + "v_proj", kvProjWorker);
                gridScheduler.addWorkerGrid(prefix + "v_norm", kvHeadNormWorker);
                gridScheduler.addWorkerGrid(prefix + "rope_and_cache", ropeWorker);
            } else {
                gridScheduler.addWorkerGrid(prefix + "rope_q_only", ropeWorker);
            }
            if (fp16KeyValue || attentionSplits() > 1) {
                gridScheduler.addWorkerGrid(
                        prefix + "attention_split",
                        WorkerGridFactory.genericWorker(
                                nHead * attentionSplits() * ATTENTION_LOCAL_SIZE,
                                ATTENTION_LOCAL_SIZE));
                gridScheduler.addWorkerGrid(
                        prefix + "attention_combine",
                        WorkerGridFactory.genericWorker(nHead * headDim, 128));
            } else {
                gridScheduler.addWorkerGrid(prefix + "attention", attentionWorker);
            }
            gridScheduler.addWorkerGrid(prefix + "wo_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "post_attn_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "post_attn_apply", dimElementWiseWorker);

            gridScheduler.addWorkerGrid(prefix + "ffn_norm_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_norm_apply", dimElementWiseWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_gate_up", ffnGateUpWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_down_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "post_ffn_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "post_ffn_apply", dimElementWiseWorker);

            gridScheduler.addWorkerGrid(prefix + "ple_gate_proj", pleGateProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_gate_gelu_mul", pleGateGeluWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_apply", dimElementWiseWorker);

            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid(prefix + "attn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_attn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ffn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_ffn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ple_post_finalize", rmsNormWorker);
            }
            if (weights.layerOutputScale[i] != null) {
                gridScheduler.addWorkerGrid(prefix + "layer_output_scale", dimElementWiseWorker);
            }
        }
        return gridScheduler;
    }
}
