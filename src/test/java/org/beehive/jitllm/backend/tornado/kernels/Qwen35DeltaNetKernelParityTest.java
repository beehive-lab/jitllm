package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The Gated Delta Net device kernels against the host operations, lane by lane, on the same inputs.
 *
 * <p>Every kernel in {@code Qwen35DeltaNetKernels} is a two-line wrapper around a static method
 * that takes its lane index, which is what makes this test possible at all: a body written directly
 * against {@code KernelContext} could only be exercised by running a model on a device, and an
 * indexing mistake in it would surface as slightly wrong text rather than as a failure.
 *
 * <p>The dimensions are Qwen3.8-27B's own, not toy ones. Two of them matter:
 *
 * <ul>
 *   <li><b>48 value heads against 16 key heads.</b> A value head reads key head {@code h % 16}, and
 *       the ratio being a whole number 3 means the wrong mapping — {@code h / 3} — is equally
 *       well-formed and equally silent. Only running both against a reference separates them.
 *   <li><b>10240 convolution channels with a kernel of 4.</b> The window is state, and a lane that
 *       advanced a neighbour's slice of it would still produce plausible output.
 * </ul>
 *
 * <p>This is a host test. It settles the arithmetic and the addressing; that the kernels also
 * compile and run on a device is a separate question, and a different gate.
 */
public class Qwen35DeltaNetKernelParityTest {

    // Qwen3.8-27B's delta-net geometry.
    private static final int CONV_DIM = 10240;
    private static final int CONV_KERNEL = 4;
    private static final int STATE_DIM = 128;
    private static final int VALUE_HEADS = 48;
    private static final int KEY_HEADS = 16;
    private static final int KEY_DIM = KEY_HEADS * STATE_DIM;
    private static final int VALUE_DIM = VALUE_HEADS * STATE_DIM;
    private static final float EPS = 1e-6f;

    /** Bit-exact: the two implementations perform the same operations in the same order. */
    private static final float EXACT = 0f;

    private final Random random = new Random(20260908L);

    private float[] noise(int n, float scale) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = (float) random.nextGaussian() * scale;
        }
        return values;
    }

    private static FloatArray toDevice(float[] values) {
        FloatArray array = new FloatArray(values.length);
        for (int i = 0; i < values.length; i++) {
            array.set(i, values[i]);
        }
        return array;
    }

    private static void assertSame(String what, FloatTensor host, FloatArray device) {
        for (int i = 0; i < host.size(); i++) {
            assertEquals(what + "[" + i + "]", host.getFloat(i), device.get(i), EXACT);
        }
    }

    /**
     * Equality to float rounding, for the one comparison that cannot be exact.
     *
     * <p>Used only where the two sides genuinely evaluate the same expression at different
     * precisions — the host through {@code Math.sqrt} and {@code Math.exp}, which are double, and
     * the device through {@code TornadoMath}, which is float. Everywhere else this test asserts bit
     * equality, because everywhere else the operations and their order are identical and a
     * tolerance would hide a real difference.
     */
    private static void assertSameToRounding(String what, FloatTensor host, FloatArray device) {
        for (int i = 0; i < host.size(); i++) {
            float expected = host.getFloat(i);
            assertEquals(
                    what + "[" + i + "]",
                    expected,
                    device.get(i),
                    Math.max(1e-9f, Math.abs(expected) * 1e-6f));
        }
    }

    @Test
    public void theConvolutionAndItsWindowMatchTheHost() {
        float[] input = noise(CONV_DIM, 1.0f);
        float[] weight = noise(CONV_DIM * CONV_KERNEL, 0.5f);
        float[] window = noise(CONV_DIM * (CONV_KERNEL - 1), 1.0f);

        FloatTensor hostInput = new ArrayFloatTensor(input.clone());
        FloatTensor hostWeight = new ArrayFloatTensor(weight.clone());
        FloatTensor hostWindow = new ArrayFloatTensor(window.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(CONV_DIM);
        CpuOperations.causalConv1d(
                hostInput, hostWeight, hostWindow, hostOut, CONV_DIM, CONV_KERNEL);

        FloatArray deviceInput = toDevice(input);
        FloatArray deviceWeight = toDevice(weight);
        FloatArray deviceWindow = toDevice(window);
        FloatArray deviceOut = new FloatArray(CONV_DIM);
        for (int channel = 0; channel < CONV_DIM; channel++) {
            Qwen35DeltaNetKernels.causalConv1dLane(
                    deviceInput, deviceWeight, deviceWindow, deviceOut, CONV_KERNEL, 0, channel);
        }

        assertSame("conv out", hostOut, deviceOut);
        // The window is state: a lane that advanced the wrong slice would still produce a correct
        // output this step and a wrong one on every step after.
        assertSame("conv window", hostWindow, deviceWindow);
    }

    @Test
    public void theL2NormMatchesTheHost() {
        float[] values = noise(KEY_DIM, 1.0f);

        FloatTensor host = new ArrayFloatTensor(values.clone());
        for (int head = 0; head < KEY_HEADS; head++) {
            CpuOperations.l2Norm(host, head * STATE_DIM, STATE_DIM, EPS);
        }

        FloatArray device = toDevice(values);
        for (int head = 0; head < KEY_HEADS; head++) {
            Qwen35DeltaNetKernels.l2NormLane(device, STATE_DIM, EPS, head);
        }

        assertSame("l2", host, device);
    }

    /**
     * The state after the update matters as much as the readout: a wrong state is invisible for one
     * token and compounds thereafter.
     */
    @Test
    public void theDeltaRuleAndItsStateMatchTheHost() {
        float[] q = noise(KEY_DIM, 0.1f);
        float[] k = noise(KEY_DIM, 0.1f);
        float[] v = noise(VALUE_DIM, 1.0f);
        float[] state = noise(VALUE_HEADS * STATE_DIM * STATE_DIM, 0.05f);

        float[] decay = new float[VALUE_HEADS];
        float[] beta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            decay[h] = 0.5f + 0.5f * random.nextFloat(); // exp of a negative log decay
            beta[h] = random.nextFloat();
        }

        FloatTensor hostState = new ArrayFloatTensor(state.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(VALUE_DIM);
        CpuOperations.deltaRuleUpdate(
                new ArrayFloatTensor(q.clone()),
                new ArrayFloatTensor(k.clone()),
                new ArrayFloatTensor(v.clone()),
                new ArrayFloatTensor(decay.clone()),
                new ArrayFloatTensor(beta.clone()),
                hostState,
                hostOut,
                VALUE_HEADS,
                KEY_HEADS,
                STATE_DIM);

        FloatArray deviceState = toDevice(state);
        FloatArray deviceOut = new FloatArray(VALUE_DIM);
        FloatArray deviceQ = toDevice(q);
        FloatArray deviceK = toDevice(k);
        FloatArray deviceV = toDevice(v);
        FloatArray deviceDecay = toDevice(decay);
        FloatArray deviceBeta = toDevice(beta);
        for (int lane = 0; lane < VALUE_HEADS * STATE_DIM; lane++) {
            Qwen35DeltaNetKernels.deltaRuleLane(
                    deviceQ,
                    deviceK,
                    deviceV,
                    deviceDecay,
                    deviceBeta,
                    deviceState,
                    deviceOut,
                    KEY_HEADS,
                    STATE_DIM,
                    0,
                    lane);
        }

        assertSame("delta readout", hostOut, deviceOut);
        assertSame("delta state", hostState, deviceState);
    }

    /**
     * The key-head mapping is tiling, not blocking, and the wrong one is well-formed.
     *
     * <p>Asserted directly rather than left to the comparison above, because both sides could drift
     * to the same wrong mapping in a future edit — which is exactly how this defect survived a
     * passing test once already.
     */
    @Test
    public void aValueHeadReadsTheKeyHeadItsIndexModuloTheKeyHeadCount() {
        // A state that is zero everywhere, a beta of one and a decay of one, so the readout is
        // q·(k ⊗ v) for whichever key head the lane picked — and each key head is given a
        // distinctive value.
        int valueHeads = 6;
        int keyHeads = 2;
        int stateDim = 2;

        float[] k = new float[keyHeads * stateDim];
        float[] q = new float[keyHeads * stateDim];
        for (int head = 0; head < keyHeads; head++) {
            k[head * stateDim] = head + 1; // key head 0 -> 1, key head 1 -> 2
            q[head * stateDim] = 1.0f;
        }
        float[] v = new float[valueHeads * stateDim];
        for (int head = 0; head < valueHeads; head++) {
            v[head * stateDim] = 1.0f;
        }
        float[] decay = new float[valueHeads];
        float[] beta = new float[valueHeads];
        java.util.Arrays.fill(decay, 1.0f);
        java.util.Arrays.fill(beta, 1.0f);

        FloatArray state = new FloatArray(valueHeads * stateDim * stateDim);
        FloatArray out = new FloatArray(valueHeads * stateDim);
        for (int lane = 0; lane < valueHeads * stateDim; lane++) {
            Qwen35DeltaNetKernels.deltaRuleLane(
                    toDevice(q),
                    toDevice(k),
                    toDevice(v),
                    toDevice(decay),
                    toDevice(beta),
                    state,
                    out,
                    keyHeads,
                    stateDim,
                    0,
                    lane);
        }

        // With S = 0, decay = 1 and beta = 1: correction = v, S becomes k ⊗ v, and the readout is
        // (q·k) * v = k[0] for column 0. So value head h reports its key head's marker, 1 or 2.
        for (int head = 0; head < valueHeads; head++) {
            float expected = (head % keyHeads) + 1;
            assertEquals(
                    "value head " + head + " read the wrong key head",
                    expected,
                    out.get(head * stateDim),
                    1e-6f);
        }
    }

    /**
     * Two layers sharing one state array do not disturb each other.
     *
     * <p>Every recurrent layer's state lives in one allocation, addressed by an offset, so a lane
     * that dropped the offset — or applied it to one of its two passes and not the other — would
     * read layer 0's state while writing layer 1's. Nothing about the output would look wrong for a
     * while.
     *
     * <p>The check is that a layer at a non-zero offset produces exactly what the same layer
     * produces alone at offset zero, and that its neighbour is untouched.
     */
    @Test
    public void aLayerOffsetIsolatesOneLayersState() {
        int heads = 4;
        int keyHeads = 2;
        int dim = 8;
        int perLayer = heads * dim * dim;

        float[] q = noise(keyHeads * dim, 0.3f);
        float[] k = noise(keyHeads * dim, 0.3f);
        float[] v = noise(heads * dim, 1.0f);
        float[] decay = new float[heads];
        float[] beta = new float[heads];
        for (int h = 0; h < heads; h++) {
            decay[h] = 0.5f + 0.5f * random.nextFloat();
            beta[h] = random.nextFloat();
        }
        float[] layerState = noise(perLayer, 0.05f);
        float[] neighbour = noise(perLayer, 0.05f);

        // Alone, at offset zero.
        FloatArray alone = toDevice(layerState);
        FloatArray aloneOut = new FloatArray(heads * dim);
        for (int lane = 0; lane < heads * dim; lane++) {
            Qwen35DeltaNetKernels.deltaRuleLane(
                    toDevice(q),
                    toDevice(k),
                    toDevice(v),
                    toDevice(decay),
                    toDevice(beta),
                    alone,
                    aloneOut,
                    keyHeads,
                    dim,
                    0,
                    lane);
        }

        // The same layer as the second of two, with a neighbour in front of it.
        float[] both = new float[2 * perLayer];
        System.arraycopy(neighbour, 0, both, 0, perLayer);
        System.arraycopy(layerState, 0, both, perLayer, perLayer);
        FloatArray shared = toDevice(both);
        FloatArray sharedOut = new FloatArray(heads * dim);
        for (int lane = 0; lane < heads * dim; lane++) {
            Qwen35DeltaNetKernels.deltaRuleLane(
                    toDevice(q),
                    toDevice(k),
                    toDevice(v),
                    toDevice(decay),
                    toDevice(beta),
                    shared,
                    sharedOut,
                    keyHeads,
                    dim,
                    perLayer,
                    lane);
        }

        for (int i = 0; i < heads * dim; i++) {
            assertEquals("readout[" + i + "]", aloneOut.get(i), sharedOut.get(i), EXACT);
        }
        for (int i = 0; i < perLayer; i++) {
            assertEquals("offset state[" + i + "]", alone.get(i), shared.get(perLayer + i), EXACT);
            assertEquals("neighbour disturbed at " + i, neighbour[i], shared.get(i), EXACT);
        }
    }

    /** The same, for the convolution's window. */
    @Test
    public void aWindowOffsetIsolatesOneLayersWindow() {
        int channels = 32;
        int kernel = 4;
        int perLayer = channels * (kernel - 1);

        float[] input = noise(channels, 1.0f);
        float[] weight = noise(channels * kernel, 0.5f);
        float[] layerWindow = noise(perLayer, 1.0f);
        float[] neighbour = noise(perLayer, 1.0f);

        FloatArray alone = toDevice(layerWindow);
        FloatArray aloneOut = new FloatArray(channels);
        for (int channel = 0; channel < channels; channel++) {
            Qwen35DeltaNetKernels.causalConv1dLane(
                    toDevice(input), toDevice(weight), alone, aloneOut, kernel, 0, channel);
        }

        float[] both = new float[2 * perLayer];
        System.arraycopy(neighbour, 0, both, 0, perLayer);
        System.arraycopy(layerWindow, 0, both, perLayer, perLayer);
        FloatArray shared = toDevice(both);
        FloatArray sharedOut = new FloatArray(channels);
        for (int channel = 0; channel < channels; channel++) {
            Qwen35DeltaNetKernels.causalConv1dLane(
                    toDevice(input),
                    toDevice(weight),
                    shared,
                    sharedOut,
                    kernel,
                    perLayer,
                    channel);
        }

        for (int i = 0; i < channels; i++) {
            assertEquals("conv out[" + i + "]", aloneOut.get(i), sharedOut.get(i), EXACT);
        }
        for (int i = 0; i < perLayer; i++) {
            assertEquals("offset window[" + i + "]", alone.get(i), shared.get(perLayer + i), EXACT);
            assertEquals("neighbour disturbed at " + i, neighbour[i], shared.get(i), EXACT);
        }
    }

    @Test
    public void theGatedNormMatchesTheHost() {
        float[] values = noise(VALUE_DIM, 1.0f);
        float[] gate = noise(VALUE_DIM, 1.0f);
        float[] weight = noise(STATE_DIM, 1.0f);

        FloatTensor host = new ArrayFloatTensor(values.clone());
        CpuOperations.gatedNorm(
                host,
                new ArrayFloatTensor(gate.clone()),
                new ArrayFloatTensor(weight.clone()),
                VALUE_HEADS,
                STATE_DIM,
                EPS);

        FloatArray device = toDevice(values);
        for (int head = 0; head < VALUE_HEADS; head++) {
            Qwen35DeltaNetKernels.gatedNormLane(
                    device, toDevice(gate), toDevice(weight), STATE_DIM, EPS, head);
        }

        // Not bit-exact, and the reason is in the arithmetic rather than in the addressing: the
        // host takes its reciprocal square root and its logistic in double and narrows once at the
        // end, the device works in float throughout. A single ULP at the output, no more.
        assertSameToRounding("gated norm", host, device);
    }

    /**
     * The decay and write strength, against the host path's own arithmetic.
     *
     * <p>The host computes these inline in its forward pass rather than in a named operation, so
     * the reference here is {@code CpuOperations.logistic} and {@code CpuOperations.softplus}
     * composed the way that forward pass composes them.
     */
    @Test
    public void theDecayAndWriteStrengthMatchTheHost() {
        float[] alpha = noise(VALUE_HEADS, 1.0f);
        float[] beta = noise(VALUE_HEADS, 1.0f);
        float[] dtBias = noise(VALUE_HEADS, 0.5f);
        float[] a = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            a[h] = -(0.5f + random.nextFloat()); // -exp(A_log)
        }

        float[] expectedDecay = new float[VALUE_HEADS];
        float[] expectedBeta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            expectedBeta[h] = CpuOperations.logistic(beta[h]);
            expectedDecay[h] =
                    (float) Math.exp(a[h] * CpuOperations.softplus(alpha[h] + dtBias[h]));
        }

        FloatArray deviceAlpha = toDevice(alpha);
        FloatArray deviceBeta = toDevice(beta);
        for (int h = 0; h < VALUE_HEADS; h++) {
            Qwen35DeltaNetKernels.decayAndBetaLane(
                    deviceAlpha, deviceBeta, toDevice(dtBias), toDevice(a), h);
        }

        for (int h = 0; h < VALUE_HEADS; h++) {
            // The exponential and the logarithm are the platform's on one side and TornadoMath's on
            // the other, so these agree to float rounding rather than bit-exactly.
            assertEquals("beta[" + h + "]", expectedBeta[h], deviceBeta.get(h), 1e-6f);
            assertEquals("decay[" + h + "]", expectedDecay[h], deviceAlpha.get(h), 1e-6f);
        }
    }
}
