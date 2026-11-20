package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;

import java.util.Random;
import org.beehive.jitllm.backend.cpu.Lfm2Forward;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.standard.Lfm2StandardWeights;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.model.loader.RopeFrequencies;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Pins the {@code lfm2moe} forward pass against a second implementation of the same arithmetic,
 * over a synthetic model — no GGUF, no device.
 *
 * <p>The reference is written from llama.cpp's {@code llama_model_lfm2::graph} and {@code
 * build_moe_ffn} in plain {@code float[]} loops, and shares no code with the production path, so
 * what it checks is the decomposition: which weights feed which operation, in what order, with what
 * strides.
 *
 * <p>The fixture exercises every combination the architecture has: layer 0 is a convolution layer
 * with a dense feed-forward, layer 1 a convolution layer with routed experts, layer 2 attends with
 * routed experts, and layer 3 convolves again after it. Seven positions run, so the convolution's
 * two-step window fills and rolls, and attention sees several keys.
 */
public class Lfm2CpuOperationEquivalenceTest {

    private static String previousFp32Property;

    @BeforeClass
    public static void pinFp32KeyValueCache() {
        previousFp32Property =
                System.getProperty(org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY);
        System.setProperty(org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY, "true");
    }

    @AfterClass
    public static void restoreKeyValueCache() {
        if (previousFp32Property == null) {
            System.clearProperty(org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY);
        } else {
            System.setProperty(
                    org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY,
                    previousFp32Property);
        }
    }

    private static final int DIM = 16;
    private static final int HIDDEN = 24;
    private static final int LAYERS = 4;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD = DIM / HEADS; // 4
    private static final int KV_DIM = KV_HEADS * HEAD; // 8
    private static final int[] HEAD_COUNT_KV = {0, 0, KV_HEADS, 0};
    private static final int CACHE = 3;
    private static final int EXPERTS = 6;
    private static final int USED = 2;
    private static final int EXPERT_HIDDEN = 8;
    private static final int DENSE_LEAD = 1;
    private static final int VOCAB = 32;
    private static final int CONTEXT = 16;
    private static final float EPS = 1e-5f;
    private static final float THETA = 10000f;

    private static Lfm2Configuration config() {
        return new Lfm2Configuration(
                "Q8_0",
                DIM,
                HIDDEN,
                LAYERS,
                HEADS,
                HEAD_COUNT_KV.clone(),
                CACHE,
                EXPERTS,
                USED,
                EXPERT_HIDDEN,
                DENSE_LEAD,
                VOCAB,
                CONTEXT,
                CONTEXT,
                EPS,
                THETA);
    }

    // ---- the fixture --------------------------------------------------------

    private final Random random = new Random(20261014L);

    private float[] noise(int n, float scale) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = (float) random.nextGaussian() * scale;
        }
        return values;
    }

    private float[] normWeights(int n) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = 0.8f + 0.4f * random.nextFloat();
        }
        return values;
    }

    private static final class Fixture {
        float[] tokenEmbd, outputNorm, output, freqReal, freqImag;
        float[][] attnNorm = new float[LAYERS][], ffnNorm = new float[LAYERS][];
        float[][] ffnGate = new float[LAYERS][], ffnUp = new float[LAYERS][];
        float[][] ffnDown = new float[LAYERS][];
        float[][] router = new float[LAYERS][], bias = new float[LAYERS][];
        float[][] gateExps = new float[LAYERS][], upExps = new float[LAYERS][];
        float[][] downExps = new float[LAYERS][];
        float[][] wq = new float[LAYERS][], wk = new float[LAYERS][], wv = new float[LAYERS][];
        float[][] wo = new float[LAYERS][], qNorm = new float[LAYERS][];
        float[][] kNorm = new float[LAYERS][];
        float[][] inProj = new float[LAYERS][], conv = new float[LAYERS][];
        float[][] outProj = new float[LAYERS][];
    }

    private Fixture fixture() {
        Lfm2Configuration c = config();
        Fixture f = new Fixture();
        f.tokenEmbd = noise(VOCAB * DIM, 0.5f);
        for (int l = 0; l < LAYERS; l++) {
            f.attnNorm[l] = normWeights(DIM);
            f.ffnNorm[l] = normWeights(DIM);
            if (c.isMoeLayer(l)) {
                f.router[l] = noise(EXPERTS * DIM, 0.5f);
                // Large enough to change which experts win, so selection and weighting differ.
                f.bias[l] = noise(EXPERTS, 0.5f);
                f.gateExps[l] = noise(EXPERTS * EXPERT_HIDDEN * DIM, 0.2f);
                f.upExps[l] = noise(EXPERTS * EXPERT_HIDDEN * DIM, 0.2f);
                f.downExps[l] = noise(EXPERTS * DIM * EXPERT_HIDDEN, 0.2f);
            } else {
                f.ffnGate[l] = noise(HIDDEN * DIM, 0.2f);
                f.ffnUp[l] = noise(HIDDEN * DIM, 0.2f);
                f.ffnDown[l] = noise(DIM * HIDDEN, 0.2f);
            }
            if (c.isAttentionLayer(l)) {
                f.wq[l] = noise(DIM * DIM, 0.3f);
                f.wk[l] = noise(KV_DIM * DIM, 0.3f);
                f.wv[l] = noise(KV_DIM * DIM, 0.3f);
                f.wo[l] = noise(DIM * DIM, 0.2f);
                f.qNorm[l] = normWeights(HEAD);
                f.kNorm[l] = normWeights(HEAD);
            } else {
                f.inProj[l] = noise(3 * DIM * DIM, 0.3f);
                f.conv[l] = noise(DIM * CACHE, 0.5f);
                f.outProj[l] = noise(DIM * DIM, 0.2f);
            }
        }
        f.outputNorm = normWeights(DIM);
        f.output = noise(VOCAB * DIM, 0.2f);
        var freqs = RopeFrequencies.precomputeFreqsCis(CONTEXT, HEAD, THETA, false, 0, 0, 0, 0);
        f.freqReal = freqs.first();
        f.freqImag = freqs.second();
        return f;
    }

    private static FloatTensor t(float[] values) {
        return values == null ? null : new ArrayFloatTensor(values);
    }

    private static FloatTensor[] t(float[][] values) {
        FloatTensor[] tensors = new FloatTensor[values.length];
        for (int i = 0; i < values.length; i++) {
            tensors[i] = t(values[i]);
        }
        return tensors;
    }

    private static Lfm2StandardWeights weights(Fixture f) {
        return new Lfm2StandardWeights(
                t(f.tokenEmbd),
                t(f.attnNorm),
                t(f.ffnNorm),
                t(f.ffnGate),
                t(f.ffnUp),
                t(f.ffnDown),
                t(f.router),
                t(f.bias),
                t(f.gateExps),
                t(f.upExps),
                t(f.downExps),
                t(f.wq),
                t(f.wk),
                t(f.wv),
                t(f.wo),
                t(f.qNorm),
                t(f.kNorm),
                t(f.inProj),
                t(f.conv),
                t(f.outProj),
                t(f.outputNorm),
                t(f.output),
                t(f.freqReal),
                t(f.freqImag),
                DataType.F32);
    }

    // ---- the independent reference ------------------------------------------

    private static float[] rmsNorm(float[] x, int offset, int n, float[] w) {
        double ss = 0;
        for (int i = 0; i < n; i++) {
            ss += x[offset + i] * x[offset + i];
        }
        float inv = (float) (1.0 / Math.sqrt(ss / n + EPS));
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = w[i] * x[offset + i] * inv;
        }
        return out;
    }

    private static float[] matVec(float[] w, int base, float[] x, int rows, int cols) {
        float[] out = new float[rows];
        for (int r = 0; r < rows; r++) {
            double sum = 0;
            for (int c = 0; c < cols; c++) {
                sum += w[base + r * cols + c] * x[c];
            }
            out[r] = (float) sum;
        }
        return out;
    }

    private static float silu(float v) {
        return v / (1f + (float) Math.exp(-v));
    }

    private static float[] swiGluFfn(
            float[] x, float[] gate, float[] up, float[] down, int base, int downBase, int hidden) {
        float[] g = matVec(gate, base, x, hidden, DIM);
        float[] u = matVec(up, base, x, hidden, DIM);
        for (int i = 0; i < hidden; i++) {
            g[i] = silu(g[i]) * u[i];
        }
        return matVec(down, downBase, g, DIM, hidden);
    }

    /** NEOX rotation of one head: element i pairs with i + head/2. */
    private static void rope(float[] v, int offset, int position) {
        int half = HEAD / 2;
        for (int i = 0; i < half; i++) {
            double freq = 1.0 / Math.pow(THETA, (2.0 * i) / HEAD);
            double angle = position * freq;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float a = v[offset + i];
            float b = v[offset + i + half];
            v[offset + i] = a * cos - b * sin;
            v[offset + i + half] = a * sin + b * cos;
        }
    }

    /** The reference model's mutable state: a key/value history and a convolution window. */
    private static final class Reference {
        final Fixture f;
        final float[][][] keys = new float[LAYERS][CONTEXT][];
        final float[][][] values = new float[LAYERS][CONTEXT][];
        final float[][][] window = new float[LAYERS][CACHE - 1][DIM];

        Reference(Fixture f) {
            this.f = f;
        }

        float[] forward(int token, int position) {
            Lfm2Configuration c = config();
            float[] x = new float[DIM];
            System.arraycopy(f.tokenEmbd, token * DIM, x, 0, DIM);
            for (int l = 0; l < LAYERS; l++) {
                float[] h = rmsNorm(x, 0, DIM, f.attnNorm[l]);
                float[] mixed = c.isAttentionLayer(l) ? attend(h, l, position) : convolve(h, l);
                for (int i = 0; i < DIM; i++) {
                    x[i] += mixed[i];
                }
                float[] n = rmsNorm(x, 0, DIM, f.ffnNorm[l]);
                float[] ffn =
                        c.isMoeLayer(l)
                                ? experts(n, l)
                                : swiGluFfn(
                                        n, f.ffnGate[l], f.ffnUp[l], f.ffnDown[l], 0, 0, HIDDEN);
                for (int i = 0; i < DIM; i++) {
                    x[i] += ffn[i];
                }
            }
            return matVec(f.output, 0, rmsNorm(x, 0, DIM, f.outputNorm), VOCAB, DIM);
        }

        float[] convolve(float[] h, int l) {
            float[] bcx = matVec(f.inProj[l], 0, h, 3 * DIM, DIM);
            float[] bx = new float[DIM];
            for (int i = 0; i < DIM; i++) {
                bx[i] = bcx[i] * bcx[2 * DIM + i];
            }
            float[] y = new float[DIM];
            for (int ch = 0; ch < DIM; ch++) {
                float sum =
                        f.conv[l][ch * CACHE] * window[l][0][ch]
                                + f.conv[l][ch * CACHE + 1] * window[l][1][ch]
                                + f.conv[l][ch * CACHE + 2] * bx[ch];
                y[ch] = bcx[DIM + ch] * sum;
            }
            window[l][0] = window[l][1];
            window[l][1] = bx;
            return matVec(f.outProj[l], 0, y, DIM, DIM);
        }

        float[] attend(float[] h, int l, int position) {
            float[] q = matVec(f.wq[l], 0, h, DIM, DIM);
            float[] k = matVec(f.wk[l], 0, h, KV_DIM, DIM);
            float[] v = matVec(f.wv[l], 0, h, KV_DIM, DIM);
            for (int head = 0; head < HEADS; head++) {
                System.arraycopy(
                        rmsNorm(q, head * HEAD, HEAD, f.qNorm[l]), 0, q, head * HEAD, HEAD);
                rope(q, head * HEAD, position);
            }
            for (int head = 0; head < KV_HEADS; head++) {
                System.arraycopy(
                        rmsNorm(k, head * HEAD, HEAD, f.kNorm[l]), 0, k, head * HEAD, HEAD);
                rope(k, head * HEAD, position);
            }
            keys[l][position] = k;
            values[l][position] = v;
            float[] out = new float[DIM];
            for (int head = 0; head < HEADS; head++) {
                int kvHead = head / (HEADS / KV_HEADS);
                float[] scores = new float[position + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int p = 0; p <= position; p++) {
                    float dot = 0;
                    for (int i = 0; i < HEAD; i++) {
                        dot += q[head * HEAD + i] * keys[l][p][kvHead * HEAD + i];
                    }
                    scores[p] = dot / (float) Math.sqrt(HEAD);
                    max = Math.max(max, scores[p]);
                }
                float sum = 0;
                for (int p = 0; p <= position; p++) {
                    scores[p] = (float) Math.exp(scores[p] - max);
                    sum += scores[p];
                }
                for (int p = 0; p <= position; p++) {
                    for (int i = 0; i < HEAD; i++) {
                        out[head * HEAD + i] += scores[p] / sum * values[l][p][kvHead * HEAD + i];
                    }
                }
            }
            return matVec(f.wo[l], 0, out, DIM, DIM);
        }

        float[] experts(float[] n, int l) {
            float[] probs = matVec(f.router[l], 0, n, EXPERTS, DIM);
            for (int e = 0; e < EXPERTS; e++) {
                probs[e] = 1f / (1f + (float) Math.exp(-probs[e]));
            }
            boolean[] taken = new boolean[EXPERTS];
            int[] chosen = new int[USED];
            float total = 0;
            for (int j = 0; j < USED; j++) {
                int best = -1;
                for (int e = 0; e < EXPERTS; e++) {
                    if (!taken[e]
                            && (best < 0
                                    || probs[e] + f.bias[l][e] > probs[best] + f.bias[l][best])) {
                        best = e;
                    }
                }
                taken[best] = true;
                chosen[j] = best;
                total += probs[best];
            }
            float[] out = new float[DIM];
            for (int j = 0; j < USED; j++) {
                int e = chosen[j];
                float[] y =
                        swiGluFfn(
                                n,
                                f.gateExps[l],
                                f.upExps[l],
                                f.downExps[l],
                                e * EXPERT_HIDDEN * DIM,
                                e * DIM * EXPERT_HIDDEN,
                                EXPERT_HIDDEN);
                for (int i = 0; i < DIM; i++) {
                    out[i] += probs[e] / total * y[i];
                }
            }
            return out;
        }
    }

    // ---- the tests ------------------------------------------------------------

    private static final int[] TOKENS = {3, 17, 5, 29, 11, 0, 23};

    private static float[] toArray(FloatTensor tensor, int n) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = tensor.getFloat(i);
        }
        return values;
    }

    @Test
    public void forwardMatchesTheIndependentReferenceAtEveryPosition() {
        Fixture f = fixture();
        Lfm2StandardWeights weights = weights(f);
        Lfm2State state = new Lfm2State(config(), -1);
        Reference reference = new Reference(f);
        for (int position = 0; position < TOKENS.length; position++) {
            float[] expected = reference.forward(TOKENS[position], position);
            float[] actual =
                    toArray(
                            Lfm2Forward.forward(
                                    config(), weights, state, TOKENS[position], position),
                            VOCAB);
            assertArrayEquals("logits at position " + position, expected, actual, 1e-4f);
        }
    }

    @Test
    public void resetClearsTheConvolutionWindow() {
        Fixture f = fixture();
        Lfm2StandardWeights weights = weights(f);
        Lfm2State state = new Lfm2State(config(), -1);
        float[] first = null;
        for (int position = 0; position < 3; position++) {
            float[] logits =
                    toArray(
                            Lfm2Forward.forward(
                                    config(), weights, state, TOKENS[position], position),
                            VOCAB);
            if (position == 0) {
                first = logits;
            }
        }
        state.resetSequenceState();
        float[] again = toArray(Lfm2Forward.forward(config(), weights, state, TOKENS[0], 0), VOCAB);
        assertArrayEquals(first, again, 0f);
    }

    @Test
    public void routerSelectsOnBiasedScoresButWeighsByUnbiasedOnes() {
        // Expert 0 has the highest score; a bias on expert 2 makes it win selection.
        float[] logits = {2f, 1f, 0f, -1f};
        float[] bias = {0f, 0f, 3f, 0f};
        FloatTensor identity = ArrayFloatTensor.allocate(4 * 4);
        for (int i = 0; i < 4; i++) {
            identity.setFloat(i * 4 + i, 1f);
        }
        int[] ids = new int[2];
        float[] weights = new float[2];
        CpuOperations.moeRouterSigmoid(
                new ArrayFloatTensor(logits),
                identity,
                new ArrayFloatTensor(bias),
                ArrayFloatTensor.allocate(4),
                ids,
                weights,
                4,
                2,
                4);
        assertEquals(2, ids[0]);
        assertEquals(0, ids[1]);
        float s2 = 0.5f;
        float s0 = 1f / (1f + (float) Math.exp(-2f));
        assertEquals(s2 / (s0 + s2), weights[0], 1e-6f);
        assertEquals(s0 / (s0 + s2), weights[1], 1e-6f);
        assertNotEquals(weights[0], weights[1], 0f);
    }

    @Test
    public void onlyAttentionLayersHoldKeyValueStorage() {
        Lfm2State state = new Lfm2State(config(), -1);
        for (int l = 0; l < LAYERS; l++) {
            if (config().isAttentionLayer(l)) {
                assertNull(state.convState[l]);
                assertEquals(CONTEXT * KV_DIM, state.keyCache[l].size());
            } else {
                assertNull(state.keyCache[l]);
                assertEquals((CACHE - 1) * DIM, state.convState[l].size());
            }
        }
        assertEquals(1, config().keyValueLayerCount());
        assertEquals(0, config().keyValueLayerIndex(2));
        assertEquals(2, config().convLayerIndex(3));
    }
}
