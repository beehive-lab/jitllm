package org.beehive.jitllm.backend.tornado.workspace;

import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The chunk-wide device arrays of the batched {@code deepseek4} prefill: row {@code b} holds the
 * chunk's token {@code b}. Every array holds whole 128-row tiles, as an int8 GEMM stores every row
 * of a tile it computes. The caches are the decode plan's ({@link DeepSeek4DeviceArrays}).
 */
public final class DeepSeek4BatchArrays {

    public final FloatArray streams;
    public final FloatArray residual;
    public final FloatArray mixes;
    public final FloatArray mixPartials;
    public final FloatArray coeffs;
    public final FloatArray normed;

    /** Every row quantized per block, packed for the DP4A kernels. */
    public final IntArray quants;

    public final FloatArray scales;
    public final IntArray sums;

    /** Every row quantized per block, bytewise for the int8 GEMMs. */
    public final ByteArray a8;

    public final FloatArray a8Scales;
    public final FloatArray branch;

    public final FloatArray queryLatent;

    /** The key/value rows, normalized and then rotated: the chunk's window keys. */
    public final FloatArray keyValue;

    public final FloatArray query;
    public final FloatArray attnOut;
    public final FloatArray grouped;

    /**
     * The attention output group after group ({@code [group][row][groupIn]}), quantized for the
     * grouped GEMM, and that GEMM's output in the same order.
     */
    public final FloatArray groupMajor;

    public final ByteArray groupMajor8;
    public final FloatArray groupMajorScales;
    public final FloatArray groupMajorOut;

    /** Every (row, head)'s attention scores, {@code maxAttended} apart. */
    public final FloatArray scores;

    public final FloatArray compressorKv;
    public final FloatArray compressorGate;

    public final FloatArray indexQuery;
    public final FloatArray indexHeadWeights;
    public final FloatArray indexBlockScores;
    public final IntArray selected;
    public final IntArray selectedCount;

    public final FloatArray routerLogits;
    public final IntArray expertIds;
    public final FloatArray expertWeights;
    public final FloatArray moeHidden;
    public final IntArray moeQuants;
    public final FloatArray moeScales;
    public final IntArray moeSums;

    /** {@code dim, hiddenDim, sharedHiddenDim, used, split, batch} for the expert kernels. */
    public final IntArray expertShape;

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

    /** The shared expert's gate, up and output, over every row. */
    public final FloatArray sharedGate;

    public final FloatArray sharedUp;
    public final FloatArray sharedOut;

    /**
     * The hash-routed layers' experts for every row, {@code hashStride} per row. A stage of a model
     * split across devices reads the session's ({@link #readHashExpertsFrom}).
     */
    public IntArray hashExperts;

    public final int hashStride;

    // @formatter:off
    /**
     * The widths: {@code widest} the widest quantized activation, {@code compressorRow} the widest
     * compressor projection, {@code maxBlocks} the most blocks an indexer scores, {@code capacity}
     * the most a query attends to.
     */
    // @formatter:on
    public record Sizes(
            int paddedBatch,
            int dim,
            int streams,
            int mixes,
            int widest,
            int qLora,
            int headDim,
            int queryDim,
            int groupedDim,
            int heads,
            int maxAttended,
            int compressorRow,
            int indexQuery,
            int indexHeads,
            int maxBlocks,
            int capacity,
            int experts,
            int used,
            int moeHidden,
            int expertHidden,
            int sharedHidden,
            int hashStride) {}

    private DeepSeek4BatchArrays(Sizes s) {
        int b = s.paddedBatch;
        streams = TornadoWorkspaces.floats(b * s.streams);
        residual = TornadoWorkspaces.floats(b * s.streams);
        mixes = TornadoWorkspaces.floats(b * s.mixes);
        mixPartials = TornadoWorkspaces.floats(2 * b * s.mixes * Glm5NextKernels.HC_CHUNKS);
        coeffs = TornadoWorkspaces.floats(b * s.mixes);
        normed = TornadoWorkspaces.floats(b * s.dim);
        quants = TornadoWorkspaces.ints(b * s.widest / 4);
        scales = TornadoWorkspaces.floats(b * s.widest / 32);
        sums = TornadoWorkspaces.ints(b * s.widest / 32);
        a8 = TornadoWorkspaces.bytes(b * s.widest);
        a8Scales = TornadoWorkspaces.floats(b * s.widest / 32);
        branch = TornadoWorkspaces.floats(b * s.dim);
        queryLatent = TornadoWorkspaces.floats(b * s.qLora);
        keyValue = TornadoWorkspaces.floats(b * s.headDim);
        query = TornadoWorkspaces.floats(b * s.queryDim);
        attnOut = TornadoWorkspaces.floats(b * s.queryDim);
        grouped = TornadoWorkspaces.floats(b * s.groupedDim);
        groupMajor = TornadoWorkspaces.floats(b * s.queryDim);
        groupMajor8 = TornadoWorkspaces.bytes(b * s.queryDim);
        groupMajorScales = TornadoWorkspaces.floats(b * s.queryDim / 32);
        groupMajorOut = TornadoWorkspaces.floats(b * s.groupedDim);
        scores = TornadoWorkspaces.floats(b * s.heads * s.maxAttended);
        compressorKv = TornadoWorkspaces.floats(b * s.compressorRow);
        compressorGate = TornadoWorkspaces.floats(b * s.compressorRow);
        indexQuery = TornadoWorkspaces.floats(b * Math.max(1, s.indexQuery));
        indexHeadWeights = TornadoWorkspaces.floats(b * Math.max(1, s.indexHeads));
        indexBlockScores = TornadoWorkspaces.floats(b * s.maxBlocks);
        selected = TornadoWorkspaces.ints(b * s.capacity);
        selectedCount = TornadoWorkspaces.ints(b);
        routerLogits = TornadoWorkspaces.floats(b * s.experts);
        expertIds = TornadoWorkspaces.ints(b * s.used);
        expertWeights = TornadoWorkspaces.floats(b * s.used);
        moeHidden = TornadoWorkspaces.floats(b * s.moeHidden);
        moeQuants = TornadoWorkspaces.ints(b * s.moeHidden / 4);
        moeScales = TornadoWorkspaces.floats(b * s.moeHidden / 32);
        moeSums = TornadoWorkspaces.ints(b * s.moeHidden / 32);
        expertShape = TornadoWorkspaces.ints(6);
        int assignments = b * Math.max(1, s.used);
        sortedToken = TornadoWorkspaces.ints(assignments);
        position = TornadoWorkspaces.ints(assignments);
        maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, Math.max(1, s.experts));
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
        hashStride = Math.max(1, s.hashStride);
        hashExperts = TornadoWorkspaces.ints(b * hashStride);
    }

    /**
     * Stages entry {@code index} ({@code layer * used + k}) of the hash-routed experts for {@code
     * row}.
     */
    public void setHashExpert(int row, int index, int expert) {
        hashExperts.set(row * hashStride + index, expert);
    }

    /** Reads the hash-routed experts the host stages in {@code session}'s arrays. */
    public void readHashExpertsFrom(DeepSeek4BatchArrays session) {
        hashExperts = session.hashExperts;
    }

    public static DeepSeek4BatchArrays allocate(Sizes sizes) {
        return new DeepSeek4BatchArrays(sizes);
    }
}
