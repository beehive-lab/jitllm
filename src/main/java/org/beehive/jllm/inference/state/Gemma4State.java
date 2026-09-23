package org.beehive.jllm.inference.state;

import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.gemma4.Gemma4Configuration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Inference state for Gemma 4 models.
 *
 * <p>In addition to the common buffers, Gemma 4 needs scratch space for its per-layer embedding
 * (PLE) mechanism. Buffers that vary in size across layers (Q/K/V, attention output, FFN hidden
 * state) are sized to the maximum across all layers. The KV cache is allocated per "physical"
 * layer: layers that reuse an earlier layer's KV cache (Gemma4's "shared KV layers" feature) simply
 * alias that layer's cache arrays.
 *
 * <p>The TornadoVM (GPU) wrapper buffers mirror the same scheme: {@link #wrapKeyCache}/ {@link
 * #wrapValueCache} are laid out back-to-back only for layers that own a KV cache (see {@link
 * #cacheLayerBaseOffset}), and the per-layer-embedding scratch buffers are exposed as flat {@code
 * FloatArray}s for transfer to the GPU.
 */
public final class Gemma4State extends State {

    /** Per-layer projected input embeddings (PLE), laid out as [layer][embeddingLengthPerLayer]. */
    public final FloatTensor perLayerInputs;

    /**
     * Scratch buffer for the per-layer model projection output, same layout as {@link
     * #perLayerInputs}.
     */
    public final FloatTensor perLayerProjScratch;

    /** Scratch buffer for a single layer's gated per-layer-embedding contribution. */
    public final FloatTensor perLayerGate;

    /** Scratch buffer for a single layer's projected per-layer-embedding output (dim-sized). */
    public final FloatTensor perLayerOut;

    /**
     * For each layer {@code l}, the base element offset of its KV-cache slot inside {@link
     * #wrapKeyCache}/{@link #wrapValueCache} (GPU path) and {@link #keyCache}/{@link #valueCache}
     * (CPU path, where it doubles as the "physical" layer index used for cache aliasing). Layers
     * that reuse an earlier layer's cache share that layer's offset, so attention kernels can
     * address the (possibly shared) cache uniformly via {@code cacheLayerBaseOffset[l]} without
     * branching on reuse.
     */
    public final int[] cacheLayerBaseOffset;

    // GPU (TornadoVM) per-layer-embedding scratch buffers; mirror
    // perLayerInputs/perLayerProjScratch/perLayerGate/perLayerOut.
    /**
     * Holds the current token's per-layer-token-embedding row (gathered on the host each step, then
     * transferred to the GPU).
     */

    // Extra RMSNorm reduction scratch buffers (GPU path): Gemma4's "sandwich norm" pattern needs
    // five independent reductions per layer (attn-norm uses the inherited `temp`, FFN-norm
    // `tempFFN`); each of the others gets its own buffer so consecutive reduce/apply pairs never
    // alias.

    public Gemma4State(Configuration config, int batchsize) {
        super(config, batchsize);

        Gemma4Configuration gemma4config = (Gemma4Configuration) config;
        int perLayerTotal = gemma4config.numberOfLayers() * gemma4config.embeddingLengthPerLayer();
        this.perLayerInputs = ArrayFloatTensor.allocate(perLayerTotal);
        this.perLayerProjScratch = ArrayFloatTensor.allocate(perLayerTotal);
        this.perLayerGate = ArrayFloatTensor.allocate(gemma4config.embeddingLengthPerLayer());
        this.perLayerOut = ArrayFloatTensor.allocate(gemma4config.dim());

        this.cacheLayerBaseOffset = computeCacheLayerBaseOffsets(gemma4config);

        this.workspace.wrapPerLayerInputs = TornadoWorkspaces.floats(perLayerTotal);
        this.workspace.wrapPerLayerProjScratch = TornadoWorkspaces.floats(perLayerTotal);
        this.workspace.wrapPerLayerGate =
                TornadoWorkspaces.floats(gemma4config.embeddingLengthPerLayer());
        this.workspace.wrapPerLayerOut = TornadoWorkspaces.floats(gemma4config.dim());
        this.workspace.wrapPerLayerTokenEmbedRow = TornadoWorkspaces.floats(perLayerTotal);

        int tempSize = 1 + ((gemma4config.dim() + localSize - 1) / localSize);
        this.workspace.tempPostAttn = TornadoWorkspaces.floats(tempSize);
        this.workspace.tempPostFfn = TornadoWorkspaces.floats(tempSize);
        this.workspace.tempPostPle = TornadoWorkspaces.floats(tempSize);

        allocateBatchPrefillWorkspace(gemma4config, perLayerTotal);
    }

    // @formatter:off
    /**
     * The chunk-wide buffers the batched prefill graphs use, allocated only when this state was
     * built for a batched plan.
     *
     * <p>Sized like the generic ones the base class allocates, at the padded row count the
     * tensor-core GEMMs launch, with three additions this family needs and no other does: the
     * per-layer-embedding block a chunk wide, one FP16 carrier for the residual (the per-layer gate
     * projection reads it as a GEMM operand and the residual itself stays FP32), and the attention
     * score scratch.
     *
     * <p>The score scratch is the one allocation here that is not obviously small. A window may be
     * the whole context, so a (row, head) slice is sized at {@code contextLength}; at the widths
     * this plan is built for that is tens of megabytes, which is the price of computing each
     * query-key dot product once instead of twice.
     */
    // @formatter:on
    private void allocateBatchPrefillWorkspace(Gemma4Configuration config, int perLayerTotal) {
        int batch = prefillBatchWidth;
        if (batch <= 1) {
            return;
        }
        int padded = (batch + 127) & ~127;
        int segment = config.embeddingLengthPerLayer();

        this.workspace.wrapPerLayerInputsBatch = TornadoWorkspaces.floats(padded * perLayerTotal);
        this.workspace.wrapPerLayerProjScratchBatch =
                TornadoWorkspaces.floats(padded * perLayerTotal);
        this.workspace.wrapPerLayerGateBatch = TornadoWorkspaces.floats(padded * segment);
        this.workspace.wrapPerLayerGateFP16Batch = TornadoWorkspaces.halfFloats(padded * segment);
        this.workspace.wrapPerLayerOutBatch = TornadoWorkspaces.floats(padded * config.dim());
        this.workspace.wrapPerLayerTokenEmbedRowBatch =
                TornadoWorkspaces.floats(padded * perLayerTotal);
        this.workspace.wrapXFP16Batch = TornadoWorkspaces.halfFloats(padded * config.dim());
        this.workspace.branchScaleBatch = TornadoWorkspaces.floats(padded);
        this.workspace.attnScoresBatch =
                TornadoWorkspaces.floats(padded * config.numberOfHeads() * config.contextLength());
        // The depth slices of the two narrow projections, before they are summed. One buffer for
        // both: they are sequential in a layer's graph, so the second overwrites what the first has
        // already been reduced out of.
        // A projection's weights decoded into FP16, so its GEMM stages operands it does not have to
        // convert. One buffer for every projection of every layer: the widest is what it has to
        // hold -- the gate/up pair -- and everything else uses a prefix. They are sequential within
        // a layer's graph and across layers, so nothing needs its own.
        this.workspace.weightsF16Scratch =
                TornadoWorkspaces.halfFloats(2 * config.maxFeedForwardLength() * config.dim());
        this.workspace.splitKPartialBatch =
                TornadoWorkspaces.floats(
                        org.beehive.jllm.backend.tornado.layers.Gemma4BatchPrefillLayers
                                        .SPLIT_K_SLICES
                                * padded
                                * config.dim());
    }

    /** This family's widest head is what one shared query buffer has to hold. */
    @Override
    protected int batchQDim(Configuration configuration) {
        Gemma4Configuration config = (Gemma4Configuration) configuration;
        return config.numberOfHeads() * config.maxHeadDim();
    }

    @Override
    protected int batchKvDim(Configuration configuration) {
        Gemma4Configuration config = (Gemma4Configuration) configuration;
        return config.numberOfKeyValueHeads() * config.maxHeadDim();
    }

    // @formatter:off
    /**
     * The widest feed-forward across the layers, because {@code hiddenDim()} on this configuration
     * refuses to answer: blocks 0-14 are 6144 wide and blocks 15-34 are 12288, and one buffer
     * shared by every layer's graph has to hold the larger.
     */
    // @formatter:on
    @Override
    protected int batchHiddenDim(Configuration configuration) {
        return ((Gemma4Configuration) configuration).maxFeedForwardLength();
    }

    /**
     * Computes, for each layer, the base element offset of its KV-cache slot in a flat buffer that
     * back-to-back concatenates only the caches of layers that own one ({@link
     * Gemma4Configuration#hasOwnKv}). Reusing layers inherit their source layer's offset (and -- by
     * construction -- its head dimension, since {@link Gemma4Configuration#kvReuseLayer} only ever
     * points to a layer with the same {@code isSwa}-ness).
     */
    private static int[] computeCacheLayerBaseOffsets(Gemma4Configuration config) {
        int nHeadKv = config.numberOfKeyValueHeads();
        int[] offsets = new int[config.numberOfLayers()];
        int running = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int reuse = config.kvReuseLayer(l);
            if (reuse < 0) {
                offsets[l] = running;
                running += config.contextLength() * (nHeadKv * config.headDim(l));
            } else {
                offsets[l] = offsets[reuse];
            }
        }
        return offsets;
    }

    /** Total number of elements needed for the (deduplicated) flat KV cache buffer. */
    private static int totalCacheElements(Gemma4Configuration config, int[] cacheLayerBaseOffset) {
        int nHeadKv = config.numberOfKeyValueHeads();
        int total = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.hasOwnKv(l)) {
                total =
                        Math.max(
                                total,
                                cacheLayerBaseOffset[l]
                                        + config.contextLength() * (nHeadKv * config.headDim(l)));
            }
        }
        return total;
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Gemma4Configuration config = (Gemma4Configuration) configuration;

        int dim = config.dim();
        int nHead = config.numberOfHeads();
        int nHeadKv = config.numberOfKeyValueHeads();
        int maxHeadDim = config.maxHeadDim();
        int maxFFN = config.maxFeedForwardLength();

        int qSize = nHead * maxHeadDim;
        int kvSize = nHeadKv * maxHeadDim;

        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(Math.max(dim, qSize));
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(maxFFN);
        fields.hb2 = ArrayFloatTensor.allocate(maxFFN);
        fields.q = ArrayFloatTensor.allocate(qSize);
        fields.k = ArrayFloatTensor.allocate(kvSize);
        fields.v = ArrayFloatTensor.allocate(kvSize);
        fields.att = ArrayFloatTensor.allocate(nHead, config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // KV cache: layers that own their KV get a fresh cache; layers that reuse an earlier
        // layer's KV (Gemma4's "shared KV layers") alias that layer's arrays directly.
        FloatTensor[] keyCache = new FloatTensor[config.numberOfLayers()];
        FloatTensor[] valueCache = new FloatTensor[config.numberOfLayers()];
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int reuse = config.kvReuseLayer(l);
            if (reuse < 0) {
                int layerKvDim = config.headDim(l) * nHeadKv;
                keyCache[l] = allocateKeyValue(config.contextLength(), layerKvDim);
                valueCache[l] = allocateKeyValue(config.contextLength(), layerKvDim);
            } else {
                keyCache[l] = keyCache[reuse];
                valueCache[l] = valueCache[reuse];
            }
        }
        fields.keyCache = keyCache;
        fields.valueCache = valueCache;

        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, dim);
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, dim);
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }

        workspace.wrapX = TornadoWorkspaces.floats(dim);
        workspace.wrapXb = TornadoWorkspaces.floats(Math.max(dim, qSize));
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(Math.max(dim, qSize));
        workspace.wrapXb2 = TornadoWorkspaces.floats(dim);
        workspace.wrapHb = TornadoWorkspaces.floats(maxFFN);
        workspace.wrapHb2 = TornadoWorkspaces.floats(maxFFN);
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(qSize);
        workspace.wrapK = TornadoWorkspaces.floats(kvSize);
        workspace.wrapV = TornadoWorkspaces.floats(kvSize);

        // Flat GPU KV cache: back-to-back slots only for layers that own a cache (see
        // cacheLayerBaseOffset).
        int[] gpuCacheLayerBaseOffset = computeCacheLayerBaseOffsets(config);
        int totalCacheElements = Math.max(1, totalCacheElements(config, gpuCacheLayerBaseOffset));
        workspace.wrapKeyCache = TornadoWorkspaces.floats(totalCacheElements);
        workspace.wrapValueCache = TornadoWorkspaces.floats(totalCacheElements);
        TornadoWorkspaces.zeroKeyValue(workspace);
        // An activation in Q8 blocks, for the packed-integer projections: four quants per int, one
        // scale and one sum of quants per block of 32. Sized for the widest activation any of them
        // reads and used as a prefix by the narrower ones. This family's feed-forward width differs
        // by layer, so the widest is the maximum over layers rather than a single hiddenDim.
        int widest = Math.max(config.dim(), config.maxFeedForwardLength());
        workspace.wrapXbQuants = TornadoWorkspaces.ints(widest / 4);
        workspace.wrapXbScales = TornadoWorkspaces.floats(widest / 32);
        workspace.wrapXbSums = TornadoWorkspaces.ints(widest / 32);

        workspace.wrapAtt = TornadoWorkspaces.floats(nHead * config.contextLength());
        // Split-KV partials: per head, SPLIT_KV numerators of headDim, then SPLIT_KV maxima and
        // SPLIT_KV sums. Sized at the widest head because this family's head width differs by
        // layer -- 256 on the sliding-window layers, 512 on the full ones -- while the buffer is
        // one allocation shared by every layer's graph.
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        nHead * config.attentionSplits() * (config.maxHeadDim() + 2));
        workspace.positionHolder = TornadoWorkspaces.ints(1);

        workspace.temp = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempLogits = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));

        return fields;
    }
}
