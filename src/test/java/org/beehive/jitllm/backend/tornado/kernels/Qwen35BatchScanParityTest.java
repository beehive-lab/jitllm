package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The two scanned recurrent kernels against the single-token kernels they have to reproduce.
 *
 * <p>The claim a batched recurrent layer rests on is that scanning a chunk inside one kernel
 * performs exactly the updates the single-token kernel performs one invocation at a time. That is
 * checkable without a device: both bodies are lane methods, so the same lanes can be walked on the
 * host, once over a chunk and once per token, and the state compared afterwards.
 *
 * <p>Comparison is <b>exact</b>. This is not an arithmetic reformulation — it is the same
 * expressions in the same order — so any difference at all is an indexing defect rather than
 * floating-point noise.
 */
// @formatter:on
public class Qwen35BatchScanParityTest {

    private static final int CONV_DIM = 640;
    private static final int CONV_KERNEL = 4;

    private static final int STATE_DIM = 32;
    private static final int VALUE_HEADS = 6;
    private static final int KEY_HEADS = 2;

    private final Random random = new Random(4242L);

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private static FloatArray toDevice(float[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, v[i]);
        }
        return a;
    }

    /** A chunk through the scan equals the same tokens through the single-token convolution. */
    @Test
    public void theConvolutionScanEqualsTheSingleTokenKernel() {
        for (int rows : new int[] {1, 2, 3, 7, 8}) {
            float[] input = noise(rows * CONV_DIM, 1.0f);
            float[] weight = noise(CONV_DIM * CONV_KERNEL, 0.5f);
            float[] window = noise(CONV_DIM * (CONV_KERNEL - 1), 1.0f);

            FloatArray scanWindow = toDevice(window);
            FloatArray scanOut = new FloatArray(rows * CONV_DIM);
            FloatArray weightArray = toDevice(weight);
            FloatArray inputArray = toDevice(input);
            for (int channel = 0; channel < CONV_DIM; channel++) {
                Qwen35BatchKernels.causalConv1dScanLane(
                        inputArray,
                        weightArray,
                        scanWindow,
                        scanOut,
                        CONV_DIM,
                        CONV_KERNEL,
                        0,
                        rows,
                        channel);
            }

            FloatArray stepWindow = toDevice(window);
            FloatArray stepOut = new FloatArray(rows * CONV_DIM);
            for (int row = 0; row < rows; row++) {
                FloatArray rowInput = new FloatArray(CONV_DIM);
                FloatArray rowOutput = new FloatArray(CONV_DIM);
                for (int c = 0; c < CONV_DIM; c++) {
                    rowInput.set(c, input[row * CONV_DIM + c]);
                }
                for (int channel = 0; channel < CONV_DIM; channel++) {
                    Qwen35DeltaNetKernels.causalConv1dLane(
                            rowInput, weightArray, stepWindow, rowOutput, CONV_KERNEL, 0, channel);
                }
                for (int c = 0; c < CONV_DIM; c++) {
                    stepOut.set(row * CONV_DIM + c, rowOutput.get(c));
                }
            }

            for (int i = 0; i < rows * CONV_DIM; i++) {
                assertEquals(
                        "rows=" + rows + " output[" + i + "]",
                        stepOut.get(i),
                        scanOut.get(i),
                        0.0f);
            }
            for (int i = 0; i < window.length; i++) {
                assertEquals(
                        "rows=" + rows + " window[" + i + "] after the chunk",
                        stepWindow.get(i),
                        scanWindow.get(i),
                        0.0f);
            }
        }
    }

    /** The same, for the delta rule: readouts and the retained state after the chunk. */
    @Test
    public void theDeltaRuleScanEqualsTheSingleTokenKernel() {
        int keyDim = KEY_HEADS * STATE_DIM;
        int valueDim = VALUE_HEADS * STATE_DIM;
        int stateSize = VALUE_HEADS * STATE_DIM * STATE_DIM;

        for (int rows : new int[] {1, 2, 3, 7, 8}) {
            float[] q = noise(rows * keyDim, 0.5f);
            float[] k = noise(rows * keyDim, 0.5f);
            float[] v = noise(rows * valueDim, 1.0f);
            float[] decay = new float[rows * VALUE_HEADS];
            float[] beta = new float[rows * VALUE_HEADS];
            for (int i = 0; i < decay.length; i++) {
                decay[i] = (float) Math.exp(-Math.abs(random.nextGaussian()));
                beta[i] = (float) (1.0 / (1.0 + Math.exp(-random.nextGaussian())));
            }
            float[] initialState = noise(stateSize, 0.1f);

            FloatArray scanState = toDevice(initialState);
            FloatArray scanOut = new FloatArray(rows * valueDim);
            FloatArray qArray = toDevice(q);
            FloatArray kArray = toDevice(k);
            FloatArray vArray = toDevice(v);
            FloatArray decayArray = toDevice(decay);
            FloatArray betaArray = toDevice(beta);
            for (int lane = 0; lane < VALUE_HEADS * STATE_DIM; lane++) {
                Qwen35BatchKernels.deltaRuleScanLane(
                        qArray,
                        kArray,
                        vArray,
                        decayArray,
                        betaArray,
                        scanState,
                        scanOut,
                        VALUE_HEADS,
                        KEY_HEADS,
                        STATE_DIM,
                        0,
                        rows,
                        lane);
            }

            FloatArray stepState = toDevice(initialState);
            FloatArray stepOut = new FloatArray(rows * valueDim);
            for (int row = 0; row < rows; row++) {
                FloatArray rowQ = slice(q, row * keyDim, keyDim);
                FloatArray rowK = slice(k, row * keyDim, keyDim);
                FloatArray rowV = slice(v, row * valueDim, valueDim);
                FloatArray rowDecay = slice(decay, row * VALUE_HEADS, VALUE_HEADS);
                FloatArray rowBeta = slice(beta, row * VALUE_HEADS, VALUE_HEADS);
                FloatArray rowOut = new FloatArray(valueDim);
                for (int lane = 0; lane < VALUE_HEADS * STATE_DIM; lane++) {
                    Qwen35DeltaNetKernels.deltaRuleLane(
                            rowQ, rowK, rowV, rowDecay, rowBeta, stepState, rowOut, KEY_HEADS,
                            STATE_DIM, 0, lane);
                }
                for (int i = 0; i < valueDim; i++) {
                    stepOut.set(row * valueDim + i, rowOut.get(i));
                }
            }

            for (int i = 0; i < rows * valueDim; i++) {
                assertEquals(
                        "rows=" + rows + " readout[" + i + "]",
                        stepOut.get(i),
                        scanOut.get(i),
                        0.0f);
            }
            for (int i = 0; i < stateSize; i++) {
                assertEquals(
                        "rows=" + rows + " state[" + i + "] after the chunk",
                        stepState.get(i),
                        scanState.get(i),
                        0.0f);
            }
        }
    }

    private static FloatArray slice(float[] source, int offset, int length) {
        FloatArray out = new FloatArray(length);
        for (int i = 0; i < length; i++) {
            out.set(i, source[offset + i]);
        }
        return out;
    }
}
