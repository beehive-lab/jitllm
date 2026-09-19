package org.beehive.jllm.model.qwen35;

import org.beehive.jllm.model.Configuration;

/**
 * Configuration for the {@code qwen35} architecture — the hybrid stack shipped by the Qwen3.5,
 * Qwen3.6 and Qwen3.8 releases, verified here against {@code Qwen3.8-27B-Q4_0.gguf}.
 *
 * <p>Named after the architecture the GGUF declares rather than after any one model, because the
 * declared architecture is what recognition keys on and several releases share it.
 *
 * <h2>Two kinds of layer</h2>
 *
 * <p>Only every {@link #fullAttentionInterval()}-th layer is attention. The rest are Gated Delta
 * Net linear-attention layers holding a recurrent state instead of a key/value cache — 48 of the
 * 27B's 64 trunk layers. {@link #isRecurrentLayer(int)} is the single place that decision is made.
 *
 * <p>{@link #numberOfLayers()} counts the trunk only. The MTP/NextN blocks that follow it are
 * {@link #numberOfNextnLayers()}, addressed at indices {@code numberOfLayers() .. numberOfLayers()
 * + numberOfNextnLayers() - 1}; they are loaded and driven by speculative decoding, never by the
 * trunk's own forward pass.
 *
 * <h2>Dimensions that are stated, not derived</h2>
 *
 * <p>The attention head dimension is {@code attention.key_length}, which is 256 while {@code dim /
 * heads} is 213. Deriving it silently mis-addresses every head. The delta-net dimensions are the
 * mirror case — they are <i>only</i> derivable, from the SSM metadata block, which is what {@link
 * #headKeyDim()} and friends do once so no caller repeats the arithmetic.
 */
// @formatter:off
public record Qwen35Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfNextnLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int numberOfHeadsKey,
        int numberOfHeadsValue,
        int fullAttentionInterval,
        int ssmConvKernel,
        int ssmStateSize,
        int ssmGroupCount,
        int ssmTimeStepRank,
        int ssmInnerSize,
        int ropeDimensionCount,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    /**
     * The attention head dimension, stated by the file; {@code dim / heads} is a different number.
     */
    @Override
    public int headSize() {
        return numberOfHeadsKey;
    }

    @Override
    public int kvDim() {
        return numberOfHeadsValue * numberOfKeyValueHeads;
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }

    // ---- layer kinds -------------------------------------------------------

    /**
     * Whether layer {@code l} mixes with a delta-net recurrence rather than with attention.
     *
     * <p>The interval counts from one, so with an interval of four the attention layers are 3, 7,
     * 11 … and everything else recurs. Trunk layers only: an MTP block is a full attention block
     * and is never recurrent, which is why the index check comes first.
     */
    public boolean isRecurrentLayer(int l) {
        return l < numberOfLayers && (l + 1) % fullAttentionInterval != 0;
    }

    /** How many trunk layers hold a key/value cache — the ones that are not recurrent. */
    public int numberOfAttentionLayers() {
        int count = 0;
        for (int l = 0; l < numberOfLayers; l++) {
            if (!isRecurrentLayer(l)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Where layer {@code l} addresses its key/value entries, or -1 if it holds none.
     *
     * <p>A <b>dense</b> index over the layers that attend, not the layer's own. Only one trunk
     * layer in four attends, so a store sized and addressed by the layer index would be four times
     * larger than the model uses — gigabytes at any useful context. Every MTP block attends and
     * follows the trunk's attention layers in this numbering.
     */
    public int keyValueLayerIndex(int l) {
        if (isRecurrentLayer(l)) {
            return -1;
        }
        int index = 0;
        for (int layer = 0; layer < l; layer++) {
            if (!isRecurrentLayer(layer)) {
                index++;
            }
        }
        return index;
    }

    /** How many blocks hold key/value entries: the attending trunk layers plus the MTP blocks. */
    @Override
    public int keyValueLayerCount() {
        return numberOfAttentionLayers() + numberOfNextnLayers;
    }

    /**
     * Where layer {@code l} addresses its recurrent state, or -1 if it holds none.
     *
     * <p>The counterpart of {@link #keyValueLayerIndex}: a dense index over the layers that recur,
     * because their convolution windows and delta-net matrices share one allocation each.
     */
    public int recurrentLayerIndex(int l) {
        if (!isRecurrentLayer(l)) {
            return -1;
        }
        int index = 0;
        for (int layer = 0; layer < l; layer++) {
            if (isRecurrentLayer(layer)) {
                index++;
            }
        }
        return index;
    }

    /** How many layers hold recurrent state. */
    public int recurrentLayerCount() {
        return numberOfLayers - numberOfAttentionLayers();
    }

    /** Total blocks the file carries: the trunk plus its MTP blocks. */
    public int numberOfBlocks() {
        return numberOfLayers + numberOfNextnLayers;
    }

    // ---- attention geometry -------------------------------------------------

    /**
     * Width of one attention layer's {@code attn_q} output.
     *
     * <p>Twice what a plain query projection would be: each head emits its query and its output
     * gate side by side, the gate applied as {@code sigmoid(gate) * attention} before the output
     * projection.
     */
    public int queryGateDim() {
        return numberOfHeadsKey * numberOfHeads * 2;
    }

    /** Width of the attention branch feeding {@code attn_output}: one query slice per head. */
    public int attentionOutputInputDim() {
        return numberOfHeadsKey * numberOfHeads;
    }

    // ---- delta-net geometry -------------------------------------------------

    /** Key/query head width in a recurrent layer: the SSM state size. */
    public int headKeyDim() {
        return ssmStateSize;
    }

    /** Key/query heads in a recurrent layer: the SSM group count. */
    public int numberOfKeyHeads() {
        return ssmGroupCount;
    }

    /** Value heads in a recurrent layer: the SSM time-step rank, one decay and one beta each. */
    public int numberOfValueHeads() {
        return ssmTimeStepRank;
    }

    /** Value head width in a recurrent layer. Derived, and required to divide exactly. */
    public int headValueDim() {
        return ssmInnerSize / ssmTimeStepRank;
    }

    /** Width of the packed key half of {@code attn_qkv}; there are two such halves, q then k. */
    public int deltaNetKeyDim() {
        return headKeyDim() * numberOfKeyHeads();
    }

    /** Width of the value part of {@code attn_qkv}, and of the {@code attn_gate} projection. */
    public int deltaNetValueDim() {
        return ssmInnerSize;
    }

    /**
     * Channels the depthwise convolution runs over: {@code q ‖ k ‖ v}, and {@code attn_qkv}'s
     * width.
     */
    public int deltaNetConvDim() {
        return 2 * deltaNetKeyDim() + deltaNetValueDim();
    }

    /**
     * How many value heads there are per key head.
     *
     * <p>Only a divisibility fact: it is <b>not</b> how a value head finds its key head. The
     * mapping is {@code h % numberOfKeyHeads()}, because the reference repeats the key heads by
     * tiling.
     */
    public int valueHeadsPerKeyHead() {
        return numberOfValueHeads() / numberOfKeyHeads();
    }

    /** Elements of recurrent delta-net state one layer holds, per sequence. */
    public int deltaNetStateSize() {
        return numberOfValueHeads() * headValueDim() * headValueDim();
    }

    /**
     * Elements of convolution history one layer holds, per sequence: the kernel minus this step.
     */
    public int convStateSize() {
        return (ssmConvKernel - 1) * deltaNetConvDim();
    }

    /** The convolution windows and delta-net matrices, one set per recurrent layer. */
    @Override
    public long recurrentStateBytes() {
        return (long) recurrentLayerCount() * (convStateSize() + deltaNetStateSize()) * Float.BYTES;
    }

    // @formatter:off
    /**
     * One, whatever the layout says.
     *
     * <p>This family's batched decode graphs consume the weights the batch-prefill graphs uploaded,
     * so a plan holds the model once however many families it lays out. At 14.944 GiB of weights
     * the difference is not a refinement: predicted twice, the 27B is refused on a device it runs
     * on.
     */
    // @formatter:on
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

    /** The chunk-wide scratch {@code Qwen35State} allocates for batched prefill. */
    @Override
    public long additionalBatchWorkspaceBytes(int batchSize) {
        if (batchSize <= 1) {
            return 0L;
        }
        long perRow =
                (long) dim() // the normalized activation
                        + queryGateDim() // the fused query/gate projection
                        + 2L * attentionOutputInputDim() // its two halves
                        + 2L * deltaNetConvDim() // the fused qkv and its convolution
                        + 2L * deltaNetValueDim() // the z gate and the readout
                        + 2L * numberOfValueHeads() // decay and beta
                        + 2L * deltaNetKeyDim() // the split queries and keys
                        + deltaNetValueDim(); // the split values
        // The attention scores of the batched FP16 key/value path, a span per (row, head) over
        // the context capacity rounded up to whole key tiles. Counted whether or not that path is
        // built: the prediction has to hold for the widest state this width can allocate.
        perRow += (long) numberOfHeads() * attentionScoreKeys(contextLength());
        long bytes = perRow * batchSize * Float.BYTES;
        // The tensor-core attention's FP16 staging, per (16-query tile, head).
        bytes += 2L * attentionStageHalves(batchSize, numberOfHeads());
        // The dequantize-then-GEMM scratch: one FP16 copy of the largest projection matrix that
        // takes the pair (gate/up and the Q4_1 ffn_down; the Q5_K ssm_out shares it), at the widths
        // that take that path.
        if (dequantGemmWidth(batchSize)) {
            bytes += 2L * hiddenDim() * dim();
        }
        return bytes;
    }

    /** Queries one tile of the tensor-core batched attention covers. */
    public static final int ATTENTION_TILE_ROWS = 16;

    /** Halves of FP16 staging one (16-query tile, head) workgroup of that attention owns. */
    public static final int ATTENTION_STAGE_HALVES_PER_TILE = 16 * 256 + 16 * 32;

    /**
     * Halves of the tensor-core attention's staging scratch for a width, or zero where the width is
     * not whole 16-query tiles (and the attention falls back to the warp kernel).
     */
    public static long attentionStageHalves(int batchSize, int heads) {
        if (batchSize <= 1 || batchSize % ATTENTION_TILE_ROWS != 0) {
            return 0L;
        }
        return (long) (batchSize / ATTENTION_TILE_ROWS) * heads * ATTENTION_STAGE_HALVES_PER_TILE;
    }

    /** Keys one tile of the tensor-core batched attention stages. */
    public static final int ATTENTION_TILE_KEYS = 32;

    /**
     * Keys the score scratch holds per (row, head): the context capacity rounded up to whole key
     * tiles, the tensor-core kernels' transposed regions being padded to them.
     */
    public static int attentionScoreKeys(int contextLength) {
        return (contextLength + ATTENTION_TILE_KEYS - 1)
                / ATTENTION_TILE_KEYS
                * ATTENTION_TILE_KEYS;
    }

    /**
     * Key/value splits of this family's decode attention: one warp per (head, split), each split a
     * contiguous slice of the positions. Thirty-two: at 24 heads that is 768 warps, which fills the
     * device the kernel was measured on; screened 8/16/32 at depths 512 and 2048, 32 fastest at
     * both (kernel + combine 28 / 44 us against 51 / 124 at eight). Sizes the split scratch.
     */
    public static final int DECODE_ATTENTION_SPLITS = 32;

    /** Rows one tile of the batched FP16 GEMM covers; a width has to be a whole number of them. */
    public static final int DEQUANT_GEMM_ROWS = 128;

    /**
     * Whether a prefill width takes the dequantize-then-GEMM path for its Q4_0 projections: the
     * chunk has to fill whole 128-row GEMM tiles. The state allocates the scratch, the plan
     * dispatches, and the memory model accounts, all from this one answer.
     */
    public static boolean dequantGemmWidth(int batchSize) {
        return batchSize > 0 && batchSize % DEQUANT_GEMM_ROWS == 0;
    }
}
