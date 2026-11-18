package org.beehive.jitllm.backend.tornado.workspace;

import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The device arrays a {@code deepseek4} decode plan adds to the shared workspace.
 *
 * <ul>
 *   <li>the residual streams and their hyper-connection coefficients;
 *   <li>the attention window of every layer, {@code [layer][slot][headDim]} in FP32, slot {@code
 *       position % window};
 *   <li>per compressed layer, the compressor's pending entries and the compressed keys; per ratio-4
 *       layer, the indexer's own entries, keys and selection;
 *   <li>the hash-routed layers' experts for the current token, staged from the host.
 * </ul>
 *
 * <p>The pending entries need no clearing for a new sequence: the compressor kernel ignores the
 * previous block's entries while it pools the first block, and every other entry it reads was
 * written earlier in the same sequence.
 */
public final class DeepSeek4DeviceArrays {

    /** The residual streams, stream-major. */
    public final FloatArray streams;

    /** The streams as they entered the current sublayer. */
    public final FloatArray residual;

    public final FloatArray mixes;
    public final FloatArray mixPartials;

    /** {@code pre[n]}, {@code post[n]}, then the {@code n × n} mix. */
    public final FloatArray coeffs;

    /** A sublayer's output, before the expansion back into the streams. */
    public final FloatArray branch;

    /** The rotation frequencies of the uncompressed layers, then of the compressed ones. */
    public final FloatArray ropeFrequencies;

    public final FloatArray compressRopeFrequencies;

    public final FloatArray keyValue;
    public final FloatArray attnOut;
    public final FloatArray grouped;
    public final FloatArray window;

    public final FloatArray compressorKv;
    public final FloatArray compressorGate;
    public final FloatArray pendingKv;
    public final FloatArray pendingScore;
    public final FloatArray compressedKeys;

    public final FloatArray indexQuery;
    public final FloatArray indexHeadWeights;
    public final FloatArray indexBlockScores;

    /** The blocks the current query attends to, and their count at {@code [0]} of the count. */
    public final IntArray selected;

    public final IntArray selectedCount;

    /**
     * The experts of the hash-routed layers for the current token, {@code used} per layer. A stage
     * of a model split across devices reads the session's ({@link #readHashExpertsFrom}).
     */
    public IntArray hashExperts;

    /** Stands in for an absent tensor (the router bias of a hash-routed layer). */
    public final FloatArray empty;

    private DeepSeek4DeviceArrays(Sizes s, float[] ropeFrequencies, float[] compressFrequencies) {
        streams = TornadoWorkspaces.floats(s.streams);
        residual = TornadoWorkspaces.floats(s.streams);
        mixes = TornadoWorkspaces.floats(s.mixes);
        mixPartials = TornadoWorkspaces.floats(2 * s.mixes * Glm5NextKernels.HC_CHUNKS);
        coeffs = TornadoWorkspaces.floats(s.mixes);
        branch = TornadoWorkspaces.floats(s.dim);
        this.ropeFrequencies = TornadoWorkspaces.floats(ropeFrequencies.length);
        this.compressRopeFrequencies = TornadoWorkspaces.floats(compressFrequencies.length);
        for (int i = 0; i < ropeFrequencies.length; i++) {
            this.ropeFrequencies.set(i, ropeFrequencies[i]);
            this.compressRopeFrequencies.set(i, compressFrequencies[i]);
        }
        keyValue = TornadoWorkspaces.floats(s.headDim);
        attnOut = TornadoWorkspaces.floats(s.queryDim);
        grouped = TornadoWorkspaces.floats(s.groupedDim);
        window = TornadoWorkspaces.floats(s.windowSize);
        compressorKv = TornadoWorkspaces.floats(s.compressorRow);
        compressorGate = TornadoWorkspaces.floats(s.compressorRow);
        pendingKv = TornadoWorkspaces.floats(Math.max(1, s.pendingSize));
        pendingScore = TornadoWorkspaces.floats(Math.max(1, s.pendingSize));
        compressedKeys = TornadoWorkspaces.floats(Math.max(1, s.compressedSize));
        indexQuery = TornadoWorkspaces.floats(Math.max(1, s.indexQuery));
        indexHeadWeights = TornadoWorkspaces.floats(Math.max(1, s.indexHeads));
        indexBlockScores = TornadoWorkspaces.floats(Math.max(1, s.maxBlocks));
        selected = TornadoWorkspaces.ints(Math.max(1, s.maxSelected));
        selectedCount = TornadoWorkspaces.ints(1);
        hashExperts = TornadoWorkspaces.ints(Math.max(1, s.hashExperts));
        empty = TornadoWorkspaces.floats(1);
    }

    /** Stages entry {@code index} ({@code layer * used + k}) of the hash-routed experts. */
    public void setHashExpert(int index, int expert) {
        hashExperts.set(index, expert);
    }

    /** Reads the hash-routed experts the host stages in {@code session}'s arrays. */
    public void readHashExpertsFrom(DeepSeek4DeviceArrays session) {
        hashExperts = session.hashExperts;
    }

    // @formatter:off
    /**
     * The widths the arrays are sized from: {@code windowSize}, {@code pendingSize} and {@code
     * compressedSize} cover every layer's part, {@code compressorRow} the widest compressor
     * projection, {@code maxBlocks} the most compressed blocks an indexer scores and {@code
     * maxSelected} the most a query attends to.
     */
    // @formatter:on
    public record Sizes(
            int dim,
            int streams,
            int mixes,
            int headDim,
            int queryDim,
            int groupedDim,
            int windowSize,
            int compressorRow,
            int pendingSize,
            int compressedSize,
            int indexQuery,
            int indexHeads,
            int maxBlocks,
            int maxSelected,
            int hashExperts) {}

    public static DeepSeek4DeviceArrays allocate(
            Sizes sizes, float[] ropeFrequencies, float[] compressFrequencies) {
        return new DeepSeek4DeviceArrays(sizes, ropeFrequencies, compressFrequencies);
    }
}
