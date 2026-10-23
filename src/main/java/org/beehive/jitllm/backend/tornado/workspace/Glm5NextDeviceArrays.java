package org.beehive.jitllm.backend.tornado.workspace;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The device arrays a {@code glm5next} decode plan adds to the shared workspace: the residual
 * streams and their hyper-connection coefficients, the KDA branch's activations, and the indexer's
 * caches and selection.
 *
 * <p>The latent cache is the workspace's {@code wrapKeyCache}, the KDA windows and matrices its
 * {@code wrapConvState} and {@code wrapDeltaState}, so a sequence reset clears them as it does for
 * qwen35.
 */
public final class Glm5NextDeviceArrays {

    /** The residual streams, stream-major. */
    public final FloatArray streams;

    /** The streams as they entered the current sublayer. */
    public final FloatArray residual;

    public final FloatArray mixes;

    /** {@code pre[n]}, {@code post[n]}, then the {@code n × n} mix. */
    public final FloatArray coeffs;

    /** A sublayer's output, before the expansion back into the streams. */
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

    public final FloatArray attnOut;
    public final FloatArray indexRawKey;
    public final FloatArray indexGate;
    public final FloatArray indexQuery;
    public final FloatArray indexHeadWeights;
    public final FloatArray indexKeyCache;
    public final FloatArray poolGateCache;
    public final FloatArray poolKeys;
    public final FloatArray poolScores;
    public final IntArray selected;

    /** Stands in for the RoPE tables the deepseek2 cache kernel reads; there is no rotation. */
    public final FloatArray noRope;

    private Glm5NextDeviceArrays(Sizes s) {
        streams = TornadoWorkspaces.floats(s.streams);
        residual = TornadoWorkspaces.floats(s.streams);
        mixes = TornadoWorkspaces.floats(s.mixes);
        coeffs = TornadoWorkspaces.floats(s.mixes);
        branch = TornadoWorkspaces.floats(s.dim);
        kdaQ = TornadoWorkspaces.floats(s.kdaWidth);
        kdaK = TornadoWorkspaces.floats(s.kdaWidth);
        kdaV = TornadoWorkspaces.floats(s.kdaWidth);
        kdaConvQ = TornadoWorkspaces.floats(s.kdaWidth);
        kdaConvK = TornadoWorkspaces.floats(s.kdaWidth);
        kdaConvV = TornadoWorkspaces.floats(s.kdaWidth);
        kdaDecayLow = TornadoWorkspaces.floats(s.kdaHeadDim);
        kdaGateLow = TornadoWorkspaces.floats(s.kdaHeadDim);
        kdaDecay = TornadoWorkspaces.floats(s.kdaWidth);
        kdaBeta = TornadoWorkspaces.floats(s.kdaHeads);
        kdaGate = TornadoWorkspaces.floats(s.kdaWidth);
        kdaOut = TornadoWorkspaces.floats(s.kdaWidth);
        attnOut = TornadoWorkspaces.floats(s.attnOut);
        indexRawKey = TornadoWorkspaces.floats(s.indexHeadDim);
        indexGate = TornadoWorkspaces.floats(s.indexHeadDim);
        indexQuery = TornadoWorkspaces.floats(s.indexHeads * s.indexHeadDim);
        indexHeadWeights = TornadoWorkspaces.floats(s.indexHeads);
        int indexerCache = Math.max(1, s.sparseLayers * s.indexerCacheStride);
        indexKeyCache = TornadoWorkspaces.floats(indexerCache);
        poolGateCache = TornadoWorkspaces.floats(indexerCache);
        poolKeys =
                TornadoWorkspaces.floats(Math.max(1, s.sparseLayers * s.maxPools * s.indexHeadDim));
        poolScores = TornadoWorkspaces.floats(s.maxPools);
        selected = TornadoWorkspaces.ints(s.maxPools);
        noRope = TornadoWorkspaces.floats(1);
    }

    /** The widths the arrays are sized from. */
    public record Sizes(
            int dim,
            int streams,
            int mixes,
            int kdaHeads,
            int kdaHeadDim,
            int kdaWidth,
            int attnOut,
            int indexHeads,
            int indexHeadDim,
            int sparseLayers,
            int indexerCacheStride,
            int maxPools) {}

    public static Glm5NextDeviceArrays allocate(Sizes sizes) {
        return new Glm5NextDeviceArrays(sizes);
    }
}
