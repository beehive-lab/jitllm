package org.beehive.jitllm.backend.tornado.workspace;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The chunk-wide device arrays of the {@code glm5next} batched prefill, row {@code b} of each
 * holding the chunk's token {@code b}. The embedding arrives in the workspace's {@code wrapXBatch},
 * which then carries each sublayer's collapsed input.
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

    /** The widths the arrays are sized from, per row; {@code batch} rows of each. */
    public record Sizes(
            int batch,
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
            int moeHidden) {}

    private Glm5NextBatchArrays(Sizes s) {
        int b = s.batch;
        streams = TornadoWorkspaces.floats(b * s.streams);
        residual = TornadoWorkspaces.floats(b * s.streams);
        mixes = TornadoWorkspaces.floats(b * s.mixes);
        coeffs = TornadoWorkspaces.floats(b * s.mixes);
        normed = TornadoWorkspaces.floats(b * s.dim);
        quants = TornadoWorkspaces.ints(b * s.widest / 4);
        scales = TornadoWorkspaces.floats(b * s.widest / 32);
        sums = TornadoWorkspaces.ints(b * s.widest / 32);
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
    }

    public static Glm5NextBatchArrays allocate(Sizes sizes) {
        return new Glm5NextBatchArrays(sizes);
    }
}
