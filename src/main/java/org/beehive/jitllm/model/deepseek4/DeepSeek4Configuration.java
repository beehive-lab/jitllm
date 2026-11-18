package org.beehive.jitllm.model.deepseek4;

import org.beehive.jitllm.model.Configuration;

/**
 * The geometry of a {@code deepseek4} model (DeepSeek-V4).
 *
 * <p>Every block reads and writes {@link #hcCount()} residual streams through manifold-constrained
 * hyper-connections, as GLM-5.3 does, and every block is attention followed by a mixture of experts
 * with one shared expert. Attention has one key/value head of {@link #headDim()} that is both the
 * key and the value of all {@link #numberOfHeads()} query heads, and a learned sink per head. A
 * query sees the last {@link #slidingWindow()} positions, plus compressed blocks when its layer's
 * {@link #compressRatios() compression ratio} is not zero: every complete block of 128 positions
 * (ratio 128), or the blocks of 4 positions its indexer scores highest (ratio 4).
 *
 * @param numberOfLayers the trunk; the MTP layers the file may append are counted in {@link
 *     #nextnLayers()}
 * @param compressRatios per layer, trunk then MTP: 0, 4 or 128
 * @param ropeDim the rotated tail of each head; the first {@code headDim - ropeDim} values are not
 *     rotated
 * @param ropeTheta the RoPE base of the layers without compression, which also use no YaRN
 * @param compressRopeTheta the RoPE base of the compressed layers, with YaRN
 * @param hashLayerCount the leading layers that route by token through a fixed table
 * @param expertChunkSize how many experts each loaded expert stack holds: the file's expert tensors
 *     are split so that every stack stays under 2³¹ bytes
 */
// @formatter:off
public record DeepSeek4Configuration(
        String quantization,
        int dim,
        int numberOfLayers,
        int nextnLayers,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        int hcCount,
        int hcSinkhornIterations,
        float hcEps,
        int numberOfHeads,
        int headDim,
        int ropeDim,
        int qLoraRank,
        int outputGroups,
        int outputLoraRank,
        int slidingWindow,
        int[] compressRatios,
        float ropeTheta,
        float compressRopeTheta,
        float ropeScalingFactor,
        int ropeOriginalContext,
        float yarnBetaFast,
        float yarnBetaSlow,
        int indexerHeads,
        int indexerHeadDim,
        int indexerTopK,
        int expertCount,
        int expertsUsed,
        int expertHiddenDim,
        int sharedExperts,
        int expertChunkSize,
        float expertWeightsScale,
        boolean expertWeightsNorm,
        float swigluLimit,
        int hashLayerCount)
        implements Configuration {
    // @formatter:on

    /** The block length of the layers whose indexer selects blocks. */
    public static final int INDEXED_RATIO = 4;

    /** The block length of the layers that attend to every complete block. */
    public static final int DENSE_RATIO = 128;

    /** The mHC expansion's sigmoid scale: fixed by the model. */
    public static final float HC_POST_SCALE = 2.0f;

    @Override
    public String quantization() {
        return quantization;
    }

    /** Layer {@code l}'s compression ratio: 0, {@link #INDEXED_RATIO} or {@link #DENSE_RATIO}. */
    public int ratio(int l) {
        return compressRatios[l];
    }

    public boolean hasIndexer(int l) {
        return compressRatios[l] == INDEXED_RATIO;
    }

    /** Whether layer {@code l} routes by token: the leading trunk layers, never an MTP layer. */
    public boolean isHashRouted(int l) {
        return l < hashLayerCount && l < numberOfLayers;
    }

    /**
     * The overlapping compressors of ratio 4 pool two blocks' worth of entries; their projections
     * are twice as wide, the first half feeding the next block.
     */
    public static int overlap(int ratio) {
        return ratio == INDEXED_RATIO ? 2 : 1;
    }

    /** Compressed blocks a cache of {@link #contextLength()} positions can hold for ratio r. */
    public int maxBlocks(int ratio) {
        return ratio == 0 ? 0 : contextLength / ratio + 1;
    }

    /** Trunk and MTP layers together: the index range of the weights. */
    public int allLayers() {
        return numberOfLayers + nextnLayers;
    }

    /** {@code (2 + n) * n} mixing coefficients: collapse, expand, then the n-by-n mix. */
    public int hcMixCount() {
        return (2 + hcCount) * hcCount;
    }

    public int queryDim() {
        return numberOfHeads * headDim;
    }

    public int nopeDim() {
        return headDim - ropeDim;
    }

    /** The input width of one output group: its heads' outputs side by side. */
    public int groupInputDim() {
        return queryDim() / outputGroups;
    }

    public int groupedOutputDim() {
        return outputGroups * outputLoraRank;
    }

    public float attentionScale() {
        return (float) (1.0 / Math.sqrt(headDim));
    }

    /** The most keys one query attends to: the window and every compressed block it may see. */
    public int maxAttended() {
        int blocks = 0;
        for (int r : compressRatios) {
            if (r == INDEXED_RATIO) {
                blocks = Math.max(blocks, Math.min(indexerTopK, maxBlocks(r)));
            } else if (r != 0) {
                blocks = Math.max(blocks, maxBlocks(r));
            }
        }
        return slidingWindow + blocks;
    }

    public int sharedHiddenDim() {
        return sharedExperts * expertHiddenDim;
    }

    /**
     * The rotation frequency of each pair of the rotated tail, {@code ropeDim / 2} of them. The
     * layers without compression use {@link #ropeTheta()} as is; the compressed layers use {@link
     * #compressRopeTheta()} with YaRN: below the correction range a frequency is divided by {@link
     * #ropeScalingFactor()}, above it kept, and blended linearly in between, as the reference's
     * {@code precompute_freqs_cis} does. YaRN does not scale the magnitude here.
     */
    public float[] ropeFrequencies(boolean compressed) {
        int half = ropeDim / 2;
        double base = compressed ? compressRopeTheta : ropeTheta;
        float[] freqs = new float[half];
        for (int i = 0; i < half; i++) {
            freqs[i] = (float) (1.0 / Math.pow(base, (2.0 * i) / ropeDim));
        }
        if (!compressed || ropeOriginalContext <= 0) {
            return freqs;
        }
        double low = Math.max(Math.floor(correctionDim(yarnBetaFast, base)), 0);
        double high = Math.min(Math.ceil(correctionDim(yarnBetaSlow, base)), ropeDim - 1);
        if (low == high) {
            high += 0.001;
        }
        for (int i = 0; i < half; i++) {
            double ramp = Math.min(1.0, Math.max(0.0, (i - low) / (high - low)));
            double smooth = 1.0 - ramp;
            freqs[i] = (float) (freqs[i] / ropeScalingFactor * (1 - smooth) + freqs[i] * smooth);
        }
        return freqs;
    }

    private double correctionDim(double rotations, double base) {
        return ropeDim
                * Math.log(ropeOriginalContext / (rotations * 2 * Math.PI))
                / (2 * Math.log(base));
    }

    @Override
    public int numberOfKeyValueHeads() {
        return 1;
    }

    @Override
    public int numberOfHeadsKey() {
        return headDim;
    }

    @Override
    public int headSize() {
        return headDim;
    }

    @Override
    public int kvDim() {
        return headDim;
    }

    @Override
    public int kvMul() {
        return numberOfHeads;
    }

    @Override
    public int hiddenDim() {
        return expertHiddenDim;
    }

    @Override
    public long recurrentStateBytes() {
        return 0;
    }
}
