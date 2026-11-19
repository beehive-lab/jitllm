package org.beehive.jitllm.backend.tornado.workspace;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The chunk-wide device arrays of a KDA layer in a batched prefill, a row per token. {@code
 * betaWide} receives {@code b_proj} padded to whole GEMM tiles; the decay kernel compacts it into
 * {@code beta}.
 */
public final class KdaBatchArrays {

    /** The output columns {@code b_proj} is padded to for the int8 GEMM. */
    public static final int BETA_PADDED = 128;

    public final FloatArray q;
    public final FloatArray k;
    public final FloatArray v;
    public final FloatArray convQ;
    public final FloatArray convK;
    public final FloatArray convV;
    public final FloatArray decayLow;
    public final FloatArray gateLow;
    public final FloatArray decay;
    public final FloatArray betaWide;
    public final FloatArray beta;
    public final FloatArray gate;
    public final FloatArray out;

    private KdaBatchArrays(int rows, int heads, int headDim) {
        int width = heads * headDim;
        q = TornadoWorkspaces.floats(rows * width);
        k = TornadoWorkspaces.floats(rows * width);
        v = TornadoWorkspaces.floats(rows * width);
        convQ = TornadoWorkspaces.floats(rows * width);
        convK = TornadoWorkspaces.floats(rows * width);
        convV = TornadoWorkspaces.floats(rows * width);
        decayLow = TornadoWorkspaces.floats(rows * headDim);
        gateLow = TornadoWorkspaces.floats(rows * headDim);
        decay = TornadoWorkspaces.floats(rows * width);
        betaWide = TornadoWorkspaces.floats(rows * Math.max(BETA_PADDED, heads));
        beta = TornadoWorkspaces.floats(rows * heads);
        gate = TornadoWorkspaces.floats(rows * width);
        out = TornadoWorkspaces.floats(rows * width);
    }

    /** Arrays for {@code rows} rows (the chunk padded to whole GEMM tiles). */
    public static KdaBatchArrays allocate(int rows, int heads, int headDim) {
        return new KdaBatchArrays(rows, heads, headDim);
    }
}
