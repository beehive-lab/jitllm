package org.beehive.jitllm.backend.tornado.workspace;

import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The chunk-wide device arrays of the {@code glm5next} batched prefill, row {@code b} of each
 * holding the chunk's token {@code b}, over the chunk padded to whole 128-row GEMM tiles. The
 * embedding arrives in the workspace's {@code wrapXBatch}, which then carries each sublayer's
 * collapsed input.
 */
public final class Glm5NextBatchArrays {

    public final FloatArray streams;
    public final FloatArray residual;
    public final FloatArray mixes;
    public final FloatArray coeffs;
    public final FloatArray normed;
    public final IntArray quants;
    public final FloatArray scales;
    public final IntArray sums;

    /**
     * The same activation in the int8 GEMMs' form — one byte per element and a scale per 32 — over
     * {@code paddedBatch} rows, the GEMMs' whole 128-row tiles.
     */
    public final ByteArray a8;

    public final FloatArray a8Scales;

    public final FloatArray branch;

    public final FloatArray kdaQ;
    public final FloatArray kdaK;
    public final FloatArray kdaV;
    public final FloatArray kdaConvQ;
    public final FloatArray kdaConvK;
    public final FloatArray kdaConvV;
    public final FloatArray kdaDecayLow;
    public final FloatArray kdaGateLow;
    public final FloatArray kdaDecay;
    public final FloatArray kdaBeta;
    public final FloatArray kdaGate;
    public final FloatArray kdaOut;

    public final FloatArray queryLatent;
    public final FloatArray kvLatent;
    public final FloatArray query;
    public final FloatArray absorbed;
    public final FloatArray latentOut;
    public final FloatArray attnOut;
    public final FloatArray indexRawKey;
    public final FloatArray indexGate;
    public final FloatArray indexQuery;
    public final FloatArray indexHeadWeights;
    public final FloatArray poolScores;
    public final IntArray selected;

    public final FloatArray hidden;
    public final FloatArray hidden2;
    public final FloatArray routerLogits;
    public final IntArray expertIds;
    public final FloatArray expertWeights;
    public final FloatArray moeHidden;
    public final IntArray moeQuants;
    public final FloatArray moeScales;
    public final IntArray moeSums;

    /** The grouped experts: each assignment's sorted row, and the per-expert tiles of rows. */
    public final IntArray sortedToken;

    public final IntArray position;
    public final IntArray tiles;
    public final int maxTiles;

    /** The routed experts' hidden activations and outputs, one row per sorted assignment. */
    public final FloatArray routedHidden;

    public final ByteArray routedHidden8;
    public final FloatArray routedHiddenScales;
    public final FloatArray routedOut;

    /** The shared expert over every row of the chunk. */
    public final FloatArray sharedGate;

    public final FloatArray sharedUp;
    public final FloatArray sharedOut;

    // @formatter:off
    /**
     * The sparse attention on FP16 tensor cores, null without it: the queries and the absorbed
     * queries in FP16, the layer's latents up to the chunk in FP16 as rows and as columns, the
     * scores of every (token, head) row against every padded position, their masked softmax, and
     * the attended latents in FP16 for the decompression.
     */
    // @formatter:on
    public final HalfFloatArray queryF16;

    public final HalfFloatArray absorbedF16;
    public final HalfFloatArray keysF16;
    public final HalfFloatArray latentTF16;
    public final FloatArray scores;
    public final HalfFloatArray probsF16;
    public final HalfFloatArray latentF16;

    /** The context rounded up to whole 128-column tiles: the score GEMM's width. */
    public final int paddedPositions;

    /** The expert kernels' shape: dim, expert hidden, shared hidden, used, split, batch. */
    public final IntArray expertShape;

    /** The widths the arrays are sized from, per row; {@code batch} rows of each. */
    public record Sizes(
            int batch,
            int paddedBatch,
            int dim,
            int streams,
            int mixes,
            int widest,
            int kdaHeads,
            int kdaHeadDim,
            int kdaWidth,
            int qLora,
            int rank,
            int queryDim,
            int absorbed,
            int attnOut,
            int indexHeads,
            int indexHeadDim,
            int maxPools,
            int hidden,
            int experts,
            int used,
            int moeHidden,
            int expertHidden,
            int sharedHidden,
            boolean tensorAttention,
            int heads,
            int context) {}

    private Glm5NextBatchArrays(Sizes s) {
        // Every chunk-wide array holds whole 128-row tiles: an int8 GEMM stores every row of a
        // tile it computes, real or padding.
        int b = s.paddedBatch;
        streams = TornadoWorkspaces.floats(b * s.streams);
        residual = TornadoWorkspaces.floats(b * s.streams);
        mixes = TornadoWorkspaces.floats(b * s.mixes);
        coeffs = TornadoWorkspaces.floats(b * s.mixes);
        normed = TornadoWorkspaces.floats(b * s.dim);
        quants = TornadoWorkspaces.ints(b * s.widest / 4);
        scales = TornadoWorkspaces.floats(b * s.widest / 32);
        sums = TornadoWorkspaces.ints(b * s.widest / 32);
        a8 = TornadoWorkspaces.bytes(s.paddedBatch * s.widest);
        a8Scales = TornadoWorkspaces.floats(s.paddedBatch * s.widest / 32);
        branch = TornadoWorkspaces.floats(b * s.dim);
        kdaQ = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaK = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaV = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaConvQ = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaConvK = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaConvV = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaDecayLow = TornadoWorkspaces.floats(b * s.kdaHeadDim);
        kdaGateLow = TornadoWorkspaces.floats(b * s.kdaHeadDim);
        kdaDecay = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaBeta = TornadoWorkspaces.floats(b * s.kdaHeads);
        kdaGate = TornadoWorkspaces.floats(b * s.kdaWidth);
        kdaOut = TornadoWorkspaces.floats(b * s.kdaWidth);
        queryLatent = TornadoWorkspaces.floats(b * s.qLora);
        kvLatent = TornadoWorkspaces.floats(b * s.rank);
        query = TornadoWorkspaces.floats(b * s.queryDim);
        absorbed = TornadoWorkspaces.floats(b * s.absorbed);
        latentOut = TornadoWorkspaces.floats(b * s.absorbed);
        attnOut = TornadoWorkspaces.floats(b * s.attnOut);
        indexRawKey = TornadoWorkspaces.floats(b * s.indexHeadDim);
        indexGate = TornadoWorkspaces.floats(b * s.indexHeadDim);
        indexQuery = TornadoWorkspaces.floats(b * s.indexHeads * s.indexHeadDim);
        indexHeadWeights = TornadoWorkspaces.floats(b * s.indexHeads);
        poolScores = TornadoWorkspaces.floats(b * s.maxPools);
        selected = TornadoWorkspaces.ints(b * s.maxPools);
        hidden = TornadoWorkspaces.floats(b * s.hidden);
        hidden2 = TornadoWorkspaces.floats(b * s.hidden);
        routerLogits = TornadoWorkspaces.floats(b * Math.max(1, s.experts));
        expertIds = TornadoWorkspaces.ints(b * Math.max(1, s.used));
        expertWeights = TornadoWorkspaces.floats(b * Math.max(1, s.used));
        moeHidden = TornadoWorkspaces.floats(b * s.moeHidden);
        moeQuants = TornadoWorkspaces.ints(b * s.moeHidden / 4);
        moeScales = TornadoWorkspaces.floats(b * s.moeHidden / 32);
        moeSums = TornadoWorkspaces.ints(b * s.moeHidden / 32);
        expertShape = TornadoWorkspaces.ints(6);
        int assignments = b * Math.max(1, s.used);
        sortedToken = TornadoWorkspaces.ints(assignments);
        position = TornadoWorkspaces.ints(assignments);
        maxTiles =
                org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels.maxTiles(
                        assignments, Math.max(1, s.experts));
        tiles = TornadoWorkspaces.ints(1 + 3 * maxTiles);
        int hidden = Math.max(32, s.expertHidden);
        routedHidden = TornadoWorkspaces.floats(assignments * hidden);
        routedHidden8 = TornadoWorkspaces.bytes(assignments * hidden);
        routedHiddenScales = TornadoWorkspaces.floats(assignments * hidden / 32);
        routedOut = TornadoWorkspaces.floats(assignments * s.dim);
        int shared = Math.max(32, s.sharedHidden);
        sharedGate = TornadoWorkspaces.floats(b * shared);
        sharedUp = TornadoWorkspaces.floats(b * shared);
        sharedOut = TornadoWorkspaces.floats(b * s.dim);
        paddedPositions = (s.context + 127) & ~127;
        if (s.tensorAttention) {
            int rows = b * s.heads;
            queryF16 = TornadoWorkspaces.halfFloats(b * s.queryDim);
            absorbedF16 = TornadoWorkspaces.halfFloats(b * s.absorbed);
            keysF16 = TornadoWorkspaces.halfFloats(paddedPositions * s.rank);
            latentTF16 = TornadoWorkspaces.halfFloats(s.rank * paddedPositions);
            scores = TornadoWorkspaces.floats(rows * paddedPositions);
            probsF16 = TornadoWorkspaces.halfFloats(rows * paddedPositions);
            latentF16 = TornadoWorkspaces.halfFloats(b * s.absorbed);
        } else {
            queryF16 = null;
            absorbedF16 = null;
            keysF16 = null;
            latentTF16 = null;
            scores = null;
            probsF16 = null;
            latentF16 = null;
        }
    }

    public static Glm5NextBatchArrays allocate(Sizes sizes) {
        return new Glm5NextBatchArrays(sizes);
    }
}
