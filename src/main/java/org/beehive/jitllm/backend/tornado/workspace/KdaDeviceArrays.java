package org.beehive.jitllm.backend.tornado.workspace;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The per-token device arrays of a KDA decode layer, and the {@code branch} a sublayer writes
 * before it is added to the residual. The windows and matrices are the workspace's {@code
 * wrapConvState} and {@code wrapDeltaState}, so a sequence reset clears them.
 */
public final class KdaDeviceArrays {

    public final FloatArray q;
    public final FloatArray k;
    public final FloatArray v;
    public final FloatArray convQ;
    public final FloatArray convK;
    public final FloatArray convV;
    public final FloatArray decayLow;
    public final FloatArray gateLow;
    public final FloatArray decay;
    public final FloatArray beta;
    public final FloatArray gate;
    public final FloatArray out;

    /** A sublayer's output before the residual add. */
    public final FloatArray branch;

    private KdaDeviceArrays(int dim, int heads, int headDim) {
        int width = heads * headDim;
        q = TornadoWorkspaces.floats(width);
        k = TornadoWorkspaces.floats(width);
        v = TornadoWorkspaces.floats(width);
        convQ = TornadoWorkspaces.floats(width);
        convK = TornadoWorkspaces.floats(width);
        convV = TornadoWorkspaces.floats(width);
        decayLow = TornadoWorkspaces.floats(headDim);
        gateLow = TornadoWorkspaces.floats(headDim);
        decay = TornadoWorkspaces.floats(width);
        beta = TornadoWorkspaces.floats(heads);
        gate = TornadoWorkspaces.floats(width);
        out = TornadoWorkspaces.floats(width);
        branch = TornadoWorkspaces.floats(dim);
    }

    public static KdaDeviceArrays allocate(int dim, int heads, int headDim) {
        return new KdaDeviceArrays(dim, heads, headDim);
    }
}
