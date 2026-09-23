package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jitllm.backend.cpu.Qwen35Forward;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.jitllm.model.loader.RopeFrequencies;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Pins the {@code qwen35} forward pass against a second implementation of the same arithmetic, over
 * a synthetic model small enough to reason about — no GGUF, no device.
 *
 * <p>The reference below is written from the architecture, in plain {@code float[]} loops, and
 * shares no code with the production path. That is the point: a test that re-called the same
 * operations would confirm they are deterministic and nothing else. What is actually at risk here
 * is the <i>decomposition</i> — which weights feed which operation, in what order, with what
 * strides — and only an independently written second version can see a mistake in it.
 *
 * <p>The fixture is deliberately hostile to the assumptions this architecture breaks:
 *
 * <ul>
 *   <li>{@code dim / heads} is 4 while the head width is 8, so anything deriving the head dimension
 *       mis-addresses every head.
 *   <li>{@code fullAttentionInterval = 4} over 4 layers puts three recurrent layers before one
 *       attention layer, so both branches run and the recurrent one runs first.
 *   <li>The rotary width is 4 against an 8-wide head, so half of every head must pass through
 *       unrotated.
 *   <li>There are twice as many value heads as key heads, so the delta rule must read each key head
 *       twice rather than walking off the end of the buffer.
 * </ul>
 */
public class Qwen35CpuOperationEquivalenceTest {

    // These compare the CPU forward against independent FP32 implementations, so they pin the
    // FP32 key/value cache: FP16 is the default, and its rounding is not what they measure.
    private static String previousFp32Property;

    @org.junit.BeforeClass
    public static void pinFp32KeyValueCache() {
        previousFp32Property =
                System.getProperty(org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY);
        System.setProperty(org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY, "true");
    }

    @org.junit.AfterClass
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
    private static final int HEAD_DIM = 8;
    private static final int INTERVAL = 4;
    private static final int CONV_KERNEL = 4;
    private static final int STATE_SIZE = 4;
    private static final int GROUPS = 2;
    private static final int TIME_STEP_RANK = 4;
    private static final int INNER = 16;
    private static final int ROPE_DIM = 4;
    private static final int VOCAB = 32;
    private static final int CONTEXT = 16;
    private static final float EPS = 1e-6f;
    private static final float THETA = 10000f;

    private static final int HEAD_V = INNER / TIME_STEP_RANK; // 4
    private static final int KEY_DIM = STATE_SIZE * GROUPS; // 8
    private static final int CONV_DIM = 2 * KEY_DIM + INNER; // 32
    private static final int KV_DIM = HEAD_DIM * KV_HEADS; // 16
    private static final int Q_GATE_DIM = HEAD_DIM * HEADS * 2; // 64
    private static final int ATTN_DIM = HEAD_DIM * HEADS; // 32

    private static Qwen35Configuration config() {
        return new Qwen35Configuration(
                "Q8_0",
                DIM,
                HIDDEN,
                LAYERS,
                0,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                INTERVAL,
                CONV_KERNEL,
                STATE_SIZE,
                GROUPS,
                TIME_STEP_RANK,
                INNER,
                ROPE_DIM,
                VOCAB,
                CONTEXT,
                CONTEXT,
                EPS,
                THETA);
    }

    // ---- the fixture --------------------------------------------------------

    private final Random random = new Random(20260908L);

    private float[] noise(int n, float scale) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = (float) random.nextGaussian() * scale;
        }
        return values;
    }

    /** Positive-ish weights for the norms, as trained scales are. */
    private float[] normWeights(int n) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = 0.8f + 0.4f * random.nextFloat();
        }
        return values;
    }

    /** Every weight the fixture needs, kept as raw arrays so the reference can index them. */
    private static final class Fixture {
        float[] tokenEmbd;
        float[][] attnNorm, ffnNorm, ffnGate, ffnDown, ffnUp;
        float[] outputNorm, output, freqReal, freqImag;
        float[][] wq, wk, wv, wo, qNorm, kNorm;
        float[][] ssmQkv, ssmGate, ssmConv, ssmAlpha, ssmBeta, ssmDt, ssmA, ssmNorm, ssmOut;
    }

    private Fixture fixture() {
        Qwen35Configuration c = config();
        Fixture f = new Fixture();
        f.tokenEmbd = noise(VOCAB * DIM, 0.5f);
        f.attnNorm = new float[LAYERS][];
        f.ffnNorm = new float[LAYERS][];
        f.ffnGate = new float[LAYERS][];
        f.ffnDown = new float[LAYERS][];
        f.ffnUp = new float[LAYERS][];
        f.wq = new float[LAYERS][];
        f.wk = new float[LAYERS][];
        f.wv = new float[LAYERS][];
        f.wo = new float[LAYERS][];
        f.qNorm = new float[LAYERS][];
        f.kNorm = new float[LAYERS][];
        f.ssmQkv = new float[LAYERS][];
        f.ssmGate = new float[LAYERS][];
        f.ssmConv = new float[LAYERS][];
        f.ssmAlpha = new float[LAYERS][];
        f.ssmBeta = new float[LAYERS][];
        f.ssmDt = new float[LAYERS][];
        f.ssmA = new float[LAYERS][];
        f.ssmNorm = new float[LAYERS][];
        f.ssmOut = new float[LAYERS][];

        for (int l = 0; l < LAYERS; l++) {
            f.attnNorm[l] = normWeights(DIM);
            f.ffnNorm[l] = normWeights(DIM);
            f.ffnGate[l] = noise(HIDDEN * DIM, 0.2f);
            f.ffnDown[l] = noise(DIM * HIDDEN, 0.2f);
            f.ffnUp[l] = noise(HIDDEN * DIM, 0.2f);
            if (c.isRecurrentLayer(l)) {
                f.ssmQkv[l] = noise(CONV_DIM * DIM, 0.2f);
                f.ssmGate[l] = noise(INNER * DIM, 0.2f);
                f.ssmConv[l] = noise(CONV_DIM * CONV_KERNEL, 0.5f);
                f.ssmAlpha[l] = noise(TIME_STEP_RANK * DIM, 0.2f);
                f.ssmBeta[l] = noise(TIME_STEP_RANK * DIM, 0.2f);
                f.ssmDt[l] = noise(TIME_STEP_RANK, 0.5f);
                // -exp(A_log): strictly negative, so the decay stays inside the unit interval.
                f.ssmA[l] = new float[TIME_STEP_RANK];
                for (int h = 0; h < TIME_STEP_RANK; h++) {
                    f.ssmA[l][h] = -(0.5f + random.nextFloat());
                }
                f.ssmNorm[l] = normWeights(HEAD_V);
                f.ssmOut[l] = noise(DIM * INNER, 0.2f);
            } else {
                f.wq[l] = noise(Q_GATE_DIM * DIM, 0.2f);
                f.wk[l] = noise(KV_DIM * DIM, 0.2f);
                f.wv[l] = noise(KV_DIM * DIM, 0.2f);
                f.wo[l] = noise(DIM * ATTN_DIM, 0.2f);
                f.qNorm[l] = normWeights(HEAD_DIM);
                f.kNorm[l] = normWeights(HEAD_DIM);
            }
        }
        f.outputNorm = normWeights(DIM);
        f.output = noise(VOCAB * DIM, 0.2f);

        var freqs = RopeFrequencies.precomputeFreqsCis(CONTEXT, ROPE_DIM, THETA, false, 0, 0, 0, 0);
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

    private Qwen35StandardWeights weights(Fixture f) {
        return new Qwen35StandardWeights(
                LAYERS,
                t(f.tokenEmbd),
                t(f.attnNorm),
                t(f.ffnNorm),
                t(f.ffnGate),
                t(f.ffnDown),
                t(f.ffnUp),
                t(f.outputNorm),
                t(f.output),
                t(f.freqReal),
                t(f.freqImag),
                t(f.wq),
                t(f.wk),
                t(f.wv),
                t(f.wo),
                t(f.qNorm),
                t(f.kNorm),
                t(f.ssmQkv),
                t(f.ssmGate),
                t(f.ssmConv),
                t(f.ssmAlpha),
                t(f.ssmBeta),
                t(f.ssmDt),
                t(f.ssmA),
                t(f.ssmNorm),
                t(f.ssmOut),
                new FloatTensor[LAYERS],
                new FloatTensor[LAYERS],
                new FloatTensor[LAYERS],
                new FloatTensor[LAYERS],
                DataType.F32);
    }

    // ---- the independent reference ------------------------------------------

    /**
     * A second implementation of the same forward pass, kept deliberately naive.
     *
     * <p>It carries its own key/value cache and its own recurrent state across calls, exactly as a
     * session does, so a sequence of steps exercises the recurrence rather than one isolated
     * update.
     */
    private static final class Reference {
        final Fixture f;
        final float[][] keyCache = new float[LAYERS][CONTEXT * KV_DIM];
        final float[][] valueCache = new float[LAYERS][CONTEXT * KV_DIM];
        final float[][] convState = new float[LAYERS][CONV_DIM * (CONV_KERNEL - 1)];
        final float[][] deltaState = new float[LAYERS][TIME_STEP_RANK * HEAD_V * HEAD_V];

        Reference(Fixture f) {
            this.f = f;
        }

        static float[] matVec(float[] w, float[] in, int rows, int cols) {
            float[] out = new float[rows];
            for (int r = 0; r < rows; r++) {
                float sum = 0f;
                for (int c = 0; c < cols; c++) {
                    sum += w[r * cols + c] * in[c];
                }
                out[r] = sum;
            }
            return out;
        }

        static void rmsNorm(float[] x, int offset, int size, float[] w) {
            float ss = 0f;
            for (int i = 0; i < size; i++) {
                ss += x[offset + i] * x[offset + i];
            }
            float inv = (float) (1.0 / Math.sqrt(ss / size + EPS));
            for (int i = 0; i < size; i++) {
                x[offset + i] = w[i] * inv * x[offset + i];
            }
        }

        static float[] rmsNormOf(float[] x, float[] w) {
            float[] out = x.clone();
            rmsNorm(out, 0, x.length, w);
            return out;
        }

        static float silu(float v) {
            return v / (float) (1.0 + Math.exp(-v));
        }

        static float sigmoid(float v) {
            return 1f / (1f + (float) Math.exp(-v));
        }

        /** Partial NeoX rotation: pairs (i, i + rot/2) below rot, everything above untouched. */
        static void rope(float[] vec, int heads, int position) {
            int half = ROPE_DIM / 2;
            for (int h = 0; h < heads; h++) {
                int base = h * HEAD_DIM;
                for (int ic = 0; ic < half; ic++) {
                    double freq = 1.0 / Math.pow(THETA, (2.0 * ic) / ROPE_DIM);
                    float cos = (float) Math.cos(position * freq);
                    float sin = (float) Math.sin(position * freq);
                    float v0 = vec[base + ic];
                    float v1 = vec[base + ic + half];
                    vec[base + ic] = v0 * cos - v1 * sin;
                    vec[base + ic + half] = v0 * sin + v1 * cos;
                }
            }
        }

        float[] forward(int token, int position) {
            Qwen35Configuration c = config();
            float[] x = new float[DIM];
            System.arraycopy(f.tokenEmbd, token * DIM, x, 0, DIM);

            for (int l = 0; l < LAYERS; l++) {
                float[] normed = rmsNormOf(x, f.attnNorm[l]);
                float[] branch =
                        c.isRecurrentLayer(l)
                                ? deltaNet(normed, l)
                                : attention(normed, l, position);
                for (int i = 0; i < DIM; i++) {
                    x[i] += branch[i];
                }

                float[] ffnIn = rmsNormOf(x, f.ffnNorm[l]);
                float[] gate = matVec(f.ffnGate[l], ffnIn, HIDDEN, DIM);
                float[] up = matVec(f.ffnUp[l], ffnIn, HIDDEN, DIM);
                float[] hidden = new float[HIDDEN];
                for (int i = 0; i < HIDDEN; i++) {
                    hidden[i] = silu(gate[i]) * up[i];
                }
                float[] down = matVec(f.ffnDown[l], hidden, DIM, HIDDEN);
                for (int i = 0; i < DIM; i++) {
                    x[i] += down[i];
                }
            }

            float[] finalNorm = rmsNormOf(x, f.outputNorm);
            return matVec(f.output, finalNorm, VOCAB, DIM);
        }

        float[] attention(float[] normed, int l, int position) {
            float[] fused = matVec(f.wq[l], normed, Q_GATE_DIM, DIM);
            float[] k = matVec(f.wk[l], normed, KV_DIM, DIM);
            float[] v = matVec(f.wv[l], normed, KV_DIM, DIM);

            float[] q = new float[ATTN_DIM];
            float[] gate = new float[ATTN_DIM];
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(fused, h * 2 * HEAD_DIM, q, h * HEAD_DIM, HEAD_DIM);
                System.arraycopy(fused, h * 2 * HEAD_DIM + HEAD_DIM, gate, h * HEAD_DIM, HEAD_DIM);
            }
            for (int h = 0; h < HEADS; h++) {
                rmsNorm(q, h * HEAD_DIM, HEAD_DIM, f.qNorm[l]);
            }
            for (int h = 0; h < KV_HEADS; h++) {
                rmsNorm(k, h * HEAD_DIM, HEAD_DIM, f.kNorm[l]);
            }
            rope(q, HEADS, position);
            rope(k, KV_HEADS, position);

            System.arraycopy(k, 0, keyCache[l], position * KV_DIM, KV_DIM);
            System.arraycopy(v, 0, valueCache[l], position * KV_DIM, KV_DIM);

            int kvMul = HEADS / KV_HEADS;
            float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
            float[] out = new float[ATTN_DIM];
            for (int h = 0; h < HEADS; h++) {
                int kvBase = (h / kvMul) * HEAD_DIM;
                float[] scores = new float[position + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int tt = 0; tt <= position; tt++) {
                    float dot = 0f;
                    for (int i = 0; i < HEAD_DIM; i++) {
                        dot += q[h * HEAD_DIM + i] * keyCache[l][tt * KV_DIM + kvBase + i];
                    }
                    scores[tt] = dot * scale;
                    max = Math.max(max, scores[tt]);
                }
                float sum = 0f;
                for (int tt = 0; tt <= position; tt++) {
                    scores[tt] = (float) Math.exp(scores[tt] - max);
                    sum += scores[tt];
                }
                for (int tt = 0; tt <= position; tt++) {
                    float a = scores[tt] / sum;
                    for (int i = 0; i < HEAD_DIM; i++) {
                        out[h * HEAD_DIM + i] += a * valueCache[l][tt * KV_DIM + kvBase + i];
                    }
                }
            }
            for (int i = 0; i < ATTN_DIM; i++) {
                out[i] *= sigmoid(gate[i]);
            }
            return matVec(f.wo[l], out, DIM, ATTN_DIM);
        }

        float[] deltaNet(float[] normed, int l) {
            float[] qkv = matVec(f.ssmQkv[l], normed, CONV_DIM, DIM);
            float[] z = matVec(f.ssmGate[l], normed, INNER, DIM);
            float[] betaRaw = matVec(f.ssmBeta[l], normed, TIME_STEP_RANK, DIM);
            float[] alphaRaw = matVec(f.ssmAlpha[l], normed, TIME_STEP_RANK, DIM);

            float[] beta = new float[TIME_STEP_RANK];
            float[] decay = new float[TIME_STEP_RANK];
            for (int h = 0; h < TIME_STEP_RANK; h++) {
                beta[h] = sigmoid(betaRaw[h]);
                double sp = Math.log1p(Math.exp(alphaRaw[h] + f.ssmDt[l][h]));
                decay[h] = (float) Math.exp(f.ssmA[l][h] * sp);
            }

            // Depthwise causal convolution, channel-major taps, oldest first.
            float[] conv = new float[CONV_DIM];
            for (int ch = 0; ch < CONV_DIM; ch++) {
                float sum = 0f;
                for (int tap = 0; tap < CONV_KERNEL - 1; tap++) {
                    sum +=
                            f.ssmConv[l][ch * CONV_KERNEL + tap]
                                    * convState[l][ch * (CONV_KERNEL - 1) + tap];
                }
                sum += f.ssmConv[l][ch * CONV_KERNEL + CONV_KERNEL - 1] * qkv[ch];
                conv[ch] = silu(sum);
            }
            for (int ch = 0; ch < CONV_DIM; ch++) {
                int base = ch * (CONV_KERNEL - 1);
                for (int tap = 0; tap + 1 < CONV_KERNEL - 1; tap++) {
                    convState[l][base + tap] = convState[l][base + tap + 1];
                }
                convState[l][base + CONV_KERNEL - 2] = qkv[ch];
            }

            float[] q = new float[KEY_DIM];
            float[] k = new float[KEY_DIM];
            float[] v = new float[INNER];
            System.arraycopy(conv, 0, q, 0, KEY_DIM);
            System.arraycopy(conv, KEY_DIM, k, 0, KEY_DIM);
            System.arraycopy(conv, 2 * KEY_DIM, v, 0, INNER);

            for (int h = 0; h < GROUPS; h++) {
                l2(q, h * STATE_SIZE);
                l2(k, h * STATE_SIZE);
            }
            float qScale = (float) (1.0 / Math.sqrt(STATE_SIZE));
            for (int i = 0; i < KEY_DIM; i++) {
                q[i] *= qScale;
            }

            float[] out = new float[INNER];
            for (int h = 0; h < TIME_STEP_RANK; h++) {
                int sBase = h * HEAD_V * HEAD_V;
                // Tiling, not blocking: ggml_repeat cycles the key heads, and the fused kernel
                // reads iv1 % nek1. Blocking here produces fluent output that decays with length.
                int kvBase = (h % GROUPS) * STATE_SIZE;
                float[] predicted = new float[HEAD_V];
                for (int i = 0; i < HEAD_V; i++) {
                    for (int j = 0; j < HEAD_V; j++) {
                        deltaState[l][sBase + i * HEAD_V + j] *= decay[h];
                        predicted[j] += deltaState[l][sBase + i * HEAD_V + j] * k[kvBase + i];
                    }
                }
                float[] correction = new float[HEAD_V];
                for (int j = 0; j < HEAD_V; j++) {
                    correction[j] = (v[h * HEAD_V + j] - predicted[j]) * beta[h];
                }
                for (int i = 0; i < HEAD_V; i++) {
                    for (int j = 0; j < HEAD_V; j++) {
                        deltaState[l][sBase + i * HEAD_V + j] += k[kvBase + i] * correction[j];
                    }
                }
                for (int i = 0; i < HEAD_V; i++) {
                    for (int j = 0; j < HEAD_V; j++) {
                        out[h * HEAD_V + j] +=
                                deltaState[l][sBase + i * HEAD_V + j] * q[kvBase + i];
                    }
                }
            }

            for (int h = 0; h < TIME_STEP_RANK; h++) {
                int base = h * HEAD_V;
                float ss = 0f;
                for (int i = 0; i < HEAD_V; i++) {
                    ss += out[base + i] * out[base + i];
                }
                float inv = (float) (1.0 / Math.sqrt(ss / HEAD_V + EPS));
                for (int i = 0; i < HEAD_V; i++) {
                    out[base + i] = f.ssmNorm[l][i] * (inv * out[base + i]) * silu(z[base + i]);
                }
            }
            return matVec(f.ssmOut[l], out, DIM, INNER);
        }

        static void l2(float[] values, int offset) {
            float ss = 0f;
            for (int i = 0; i < STATE_SIZE; i++) {
                ss += values[offset + i] * values[offset + i];
            }
            float inv = 1.0f / Math.max((float) Math.sqrt(ss), EPS);
            for (int i = 0; i < STATE_SIZE; i++) {
                values[offset + i] *= inv;
            }
        }
    }

    // ---- the checks ---------------------------------------------------------

    @Test
    public void matchesAnIndependentImplementationAcrossASequence() {
        Fixture f = fixture();
        Qwen35Configuration config = config();
        Qwen35StandardWeights w = weights(f);
        Qwen35State state = new Qwen35State(config, -1);
        Reference reference = new Reference(f);

        int[] tokens = {3, 11, 7, 7, 25, 0, 19};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor logits =
                    Qwen35Forward.forward(config, w, state, tokens[position], position);
            float[] expected = reference.forward(tokens[position], position);
            for (int i = 0; i < VOCAB; i++) {
                assertEquals(
                        "logit " + i + " at position " + position,
                        expected[i],
                        logits.getFloat(i),
                        2e-4f);
            }
        }
    }

    /**
     * The recurrence must actually carry state.
     *
     * <p>Feeding the same token twice at successive positions has to give different logits — if it
     * does not, the delta-net state is being discarded and the previous check would still pass,
     * because the reference would be discarding it too only if it shared the bug. It does not, so
     * this asserts the property directly.
     */
    @Test
    public void recurrentStateCarriesBetweenSteps() {
        Fixture f = fixture();
        Qwen35Configuration config = config();
        Qwen35StandardWeights w = weights(f);
        Qwen35State state = new Qwen35State(config, -1);

        float[] first = new float[VOCAB];
        Qwen35Forward.forward(config, w, state, 5, 0)
                .copyTo(0, new ArrayFloatTensor(first), 0, VOCAB);
        float[] second = new float[VOCAB];
        Qwen35Forward.forward(config, w, state, 5, 1)
                .copyTo(0, new ArrayFloatTensor(second), 0, VOCAB);

        boolean anyDifferent = false;
        for (int i = 0; i < VOCAB; i++) {
            anyDifferent |= Math.abs(first[i] - second[i]) > 1e-5f;
        }
        assertTrue("the same token at two positions produced identical logits", anyDifferent);
    }

    /**
     * A reset must return the recurrent state to where a fresh session starts, or a reused session
     * silently continues the previous sequence.
     */
    @Test
    public void resetClearsTheRecurrentState() {
        Fixture f = fixture();
        Qwen35Configuration config = config();
        Qwen35StandardWeights w = weights(f);
        Qwen35State state = new Qwen35State(config, -1);

        FloatTensor fresh = Qwen35Forward.forward(config, w, state, 9, 0);
        float[] before = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            before[i] = fresh.getFloat(i);
        }

        Qwen35Forward.forward(config, w, state, 4, 1);
        Qwen35Forward.forward(config, w, state, 12, 2);
        state.resetSequenceState();

        FloatTensor again = Qwen35Forward.forward(config, w, state, 9, 0);
        for (int i = 0; i < VOCAB; i++) {
            assertEquals("logit " + i + " after reset", before[i], again.getFloat(i), 1e-5f);
        }
    }

    /** Key/value storage exists only where a layer attends. */
    @Test
    public void keyValueCachesAreAllocatedOnlyForAttentionLayers() {
        Qwen35Configuration config = config();
        Qwen35State state = new Qwen35State(config, -1);
        for (int l = 0; l < LAYERS; l++) {
            if (config.isRecurrentLayer(l)) {
                assertNull("layer " + l + " recurs and needs no key cache", state.keyCache[l]);
                assertNotEquals(null, state.convState[l]);
                assertNotEquals(null, state.deltaState[l]);
            } else {
                assertNotEquals(null, state.keyCache[l]);
                assertNull("layer " + l + " attends and needs no conv state", state.convState[l]);
            }
        }
    }

    /** The interval counts from one: with an interval of four, layer three attends. */
    @Test
    public void layerKindsFollowTheInterval() {
        Qwen35Configuration config = config();
        assertTrue(config.isRecurrentLayer(0));
        assertTrue(config.isRecurrentLayer(1));
        assertTrue(config.isRecurrentLayer(2));
        assertTrue(!config.isRecurrentLayer(3));
        assertEquals(1, config.numberOfAttentionLayers());
    }
}
