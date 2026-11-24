package org.beehive.jitllm.inference.state;

import java.util.function.IntPredicate;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host memory of Kimi delta attention (KDA): per KDA layer its three convolution windows and
 * its delta-rule matrices, and the per-token buffers every KDA layer shares.
 *
 * <p>Shared by every family with KDA layers. The per-layer arrays are indexed by layer and are null
 * on the layers that mix tokens otherwise.
 */
public final class KdaHostState {

    public final FloatTensor[] convWindowQ;
    public final FloatTensor[] convWindowK;
    public final FloatTensor[] convWindowV;

    /** Per head a {@code headDim × headDim} matrix, row = key channel, column = value channel. */
    public final FloatTensor[] deltaState;

    public final FloatTensor q;
    public final FloatTensor k;
    public final FloatTensor v;
    public final FloatTensor convQ;
    public final FloatTensor convK;
    public final FloatTensor convV;
    public final FloatTensor lowRank;
    public final FloatTensor decay;
    public final FloatTensor beta;
    public final FloatTensor gate;
    public final FloatTensor out;

    /**
     * @param layers the layer count the per-layer arrays are indexed by
     * @param isKda which of those layers are KDA layers
     */
    public KdaHostState(int layers, IntPredicate isKda, int heads, int headDim, int convKernel) {
        int width = heads * headDim;
        int window = width * (convKernel - 1);
        this.convWindowQ = new FloatTensor[layers];
        this.convWindowK = new FloatTensor[layers];
        this.convWindowV = new FloatTensor[layers];
        this.deltaState = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            if (isKda.test(l)) {
                convWindowQ[l] = ArrayFloatTensor.allocate(window);
                convWindowK[l] = ArrayFloatTensor.allocate(window);
                convWindowV[l] = ArrayFloatTensor.allocate(window);
                deltaState[l] = ArrayFloatTensor.allocate(heads * headDim * headDim);
            }
        }
        this.q = ArrayFloatTensor.allocate(width);
        this.k = ArrayFloatTensor.allocate(width);
        this.v = ArrayFloatTensor.allocate(width);
        this.convQ = ArrayFloatTensor.allocate(width);
        this.convK = ArrayFloatTensor.allocate(width);
        this.convV = ArrayFloatTensor.allocate(width);
        this.lowRank = ArrayFloatTensor.allocate(headDim);
        this.decay = ArrayFloatTensor.allocate(width);
        this.beta = ArrayFloatTensor.allocate(heads);
        this.gate = ArrayFloatTensor.allocate(width);
        this.out = ArrayFloatTensor.allocate(width);
    }

    /** Clears the convolution windows and the delta-rule matrices. */
    public void reset() {
        for (int l = 0; l < deltaState.length; l++) {
            if (deltaState[l] != null) {
                convWindowQ[l].fillInPlace(0, convWindowQ[l].size(), 0f);
                convWindowK[l].fillInPlace(0, convWindowK[l].size(), 0f);
                convWindowV[l].fillInPlace(0, convWindowV[l].size(), 0f);
                deltaState[l].fillInPlace(0, deltaState[l].size(), 0f);
            }
        }
    }
}
