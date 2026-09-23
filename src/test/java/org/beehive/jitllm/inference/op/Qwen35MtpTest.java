package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertEquals;
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
 * The MTP (NextN) draft head, over a synthetic model, against a second implementation.
 *
 * <p>Three things about this block are easy to get subtly wrong and impossible to see in the
 * output, because the trunk's token is what gets emitted whatever the head says:
 *
 * <ul>
 *   <li><b>Which hidden state it consumes.</b> It is the trunk's state <i>after</i> the final norm
 *       and <i>before</i> the vocabulary projection, taken at the position that produced the token
 *       being drafted from — not the current step's.
 *   <li><b>The concatenation order.</b> The normalized token embedding comes first and the
 *       normalized hidden state second. Swapping them halves the projection's meaning without
 *       changing a shape.
 *   <li><b>That it leaves the trunk alone.</b> A draft is a question about the future; asking it
 *       must not advance the recurrence or overwrite the residual stream.
 * </ul>
 */
public class Qwen35MtpTest {

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
    private static final int TRUNK = 4;
    private static final int NEXTN = 1;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int VOCAB = 32;
    private static final int CONTEXT = 16;
    private static final float EPS = 1e-6f;
    private static final float THETA = 10000f;
    private static final int KV_DIM = HEAD_DIM * KV_HEADS;
    private static final int Q_GATE_DIM = HEAD_DIM * HEADS * 2;
    private static final int ATTN_DIM = HEAD_DIM * HEADS;
    private static final int BLOCK = TRUNK; // the MTP block's index

    private static Qwen35Configuration config() {
        return new Qwen35Configuration(
                "Q8_0",
                DIM,
                HIDDEN,
                TRUNK,
                NEXTN,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                /* fullAttentionInterval */ 4,
                /* ssmConvKernel */ 4,
                /* ssmStateSize */ 4,
                /* ssmGroupCount */ 2,
                /* ssmTimeStepRank */ 4,
                /* ssmInnerSize */ 16,
                /* ropeDimensionCount */ 4,
                VOCAB,
                CONTEXT,
                CONTEXT,
                EPS,
                THETA);
    }

    private final Random random = new Random(77L);

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private float[] normWeights(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = 0.8f + 0.4f * random.nextFloat();
        }
        return v;
    }

    private static FloatTensor t(float[] v) {
        return v == null ? null : new ArrayFloatTensor(v);
    }

    // Only the MTP block's weights are exercised here; the trunk's are present but unused, so the
    // recurrent layers get placeholder tensors of the right shape rather than real ones.
    private float[] tokenEmbd;
    private float[] eNorm, hNorm, ehProj, sharedHeadNorm;
    private float[] attnNorm, ffnNorm, ffnGate, ffnDown, ffnUp;
    private float[] wq, wk, wv, wo, qNorm, kNorm;
    private float[] output, outputNorm;
    private float[] freqReal, freqImag;

    private Qwen35StandardWeights weights() {
        Qwen35Configuration c = config();
        int blocks = c.numberOfBlocks(); // 5
        int convDim = c.deltaNetConvDim();

        tokenEmbd = noise(VOCAB * DIM, 0.5f);
        eNorm = normWeights(DIM);
        hNorm = normWeights(DIM);
        ehProj = noise(DIM * 2 * DIM, 0.2f);
        sharedHeadNorm = normWeights(DIM);
        attnNorm = normWeights(DIM);
        ffnNorm = normWeights(DIM);
        ffnGate = noise(HIDDEN * DIM, 0.2f);
        ffnDown = noise(DIM * HIDDEN, 0.2f);
        ffnUp = noise(HIDDEN * DIM, 0.2f);
        wq = noise(Q_GATE_DIM * DIM, 0.2f);
        wk = noise(KV_DIM * DIM, 0.2f);
        wv = noise(KV_DIM * DIM, 0.2f);
        wo = noise(DIM * ATTN_DIM, 0.2f);
        qNorm = normWeights(HEAD_DIM);
        kNorm = normWeights(HEAD_DIM);
        output = noise(VOCAB * DIM, 0.2f);
        outputNorm = normWeights(DIM);

        var freqs = RopeFrequencies.precomputeFreqsCis(CONTEXT, 4, THETA, false, 0, 0, 0, 0);
        freqReal = freqs.first();
        freqImag = freqs.second();

        FloatTensor[] blockAttnNorm = new FloatTensor[blocks];
        FloatTensor[] blockFfnNorm = new FloatTensor[blocks];
        FloatTensor[] blockGate = new FloatTensor[blocks];
        FloatTensor[] blockDown = new FloatTensor[blocks];
        FloatTensor[] blockUp = new FloatTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            blockAttnNorm[l] = t(normWeights(DIM));
            blockFfnNorm[l] = t(normWeights(DIM));
            blockGate[l] = t(noise(HIDDEN * DIM, 0.2f));
            blockDown[l] = t(noise(DIM * HIDDEN, 0.2f));
            blockUp[l] = t(noise(HIDDEN * DIM, 0.2f));
        }
        blockAttnNorm[BLOCK] = t(attnNorm);
        blockFfnNorm[BLOCK] = t(ffnNorm);
        blockGate[BLOCK] = t(ffnGate);
        blockDown[BLOCK] = t(ffnDown);
        blockUp[BLOCK] = t(ffnUp);

        FloatTensor[] q = new FloatTensor[blocks];
        FloatTensor[] k = new FloatTensor[blocks];
        FloatTensor[] v = new FloatTensor[blocks];
        FloatTensor[] o = new FloatTensor[blocks];
        FloatTensor[] qn = new FloatTensor[blocks];
        FloatTensor[] kn = new FloatTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            if (!c.isRecurrentLayer(l)) {
                q[l] = t(l == BLOCK ? wq : noise(Q_GATE_DIM * DIM, 0.2f));
                k[l] = t(l == BLOCK ? wk : noise(KV_DIM * DIM, 0.2f));
                v[l] = t(l == BLOCK ? wv : noise(KV_DIM * DIM, 0.2f));
                o[l] = t(l == BLOCK ? wo : noise(DIM * ATTN_DIM, 0.2f));
                qn[l] = t(l == BLOCK ? qNorm : normWeights(HEAD_DIM));
                kn[l] = t(l == BLOCK ? kNorm : normWeights(HEAD_DIM));
            }
        }

        FloatTensor[] ssmQkv = new FloatTensor[TRUNK];
        FloatTensor[] ssmGate = new FloatTensor[TRUNK];
        FloatTensor[] ssmConv = new FloatTensor[TRUNK];
        FloatTensor[] ssmAlpha = new FloatTensor[TRUNK];
        FloatTensor[] ssmBeta = new FloatTensor[TRUNK];
        FloatTensor[] ssmDt = new FloatTensor[TRUNK];
        FloatTensor[] ssmA = new FloatTensor[TRUNK];
        FloatTensor[] ssmNorm = new FloatTensor[TRUNK];
        FloatTensor[] ssmOut = new FloatTensor[TRUNK];
        for (int l = 0; l < TRUNK; l++) {
            if (c.isRecurrentLayer(l)) {
                ssmQkv[l] = t(noise(convDim * DIM, 0.2f));
                ssmGate[l] = t(noise(c.deltaNetValueDim() * DIM, 0.2f));
                ssmConv[l] = t(noise(convDim * c.ssmConvKernel(), 0.3f));
                ssmAlpha[l] = t(noise(c.numberOfValueHeads() * DIM, 0.2f));
                ssmBeta[l] = t(noise(c.numberOfValueHeads() * DIM, 0.2f));
                ssmDt[l] = t(noise(c.numberOfValueHeads(), 0.3f));
                float[] a = new float[c.numberOfValueHeads()];
                for (int h = 0; h < a.length; h++) {
                    a[h] = -(0.5f + random.nextFloat());
                }
                ssmA[l] = t(a);
                ssmNorm[l] = t(normWeights(c.headValueDim()));
                ssmOut[l] = t(noise(DIM * c.deltaNetValueDim(), 0.2f));
            }
        }

        FloatTensor[] en = new FloatTensor[blocks];
        FloatTensor[] hn = new FloatTensor[blocks];
        FloatTensor[] eh = new FloatTensor[blocks];
        FloatTensor[] shn = new FloatTensor[blocks];
        en[BLOCK] = t(eNorm);
        hn[BLOCK] = t(hNorm);
        eh[BLOCK] = t(ehProj);
        shn[BLOCK] = t(sharedHeadNorm);

        return new Qwen35StandardWeights(
                blocks,
                t(tokenEmbd),
                blockAttnNorm,
                blockFfnNorm,
                blockGate,
                blockDown,
                blockUp,
                t(outputNorm),
                t(output),
                t(freqReal),
                t(freqImag),
                q,
                k,
                v,
                o,
                qn,
                kn,
                ssmQkv,
                ssmGate,
                ssmConv,
                ssmAlpha,
                ssmBeta,
                ssmDt,
                ssmA,
                ssmNorm,
                ssmOut,
                en,
                hn,
                eh,
                shn,
                DataType.F32);
    }

    // ---- reference ----------------------------------------------------------

    private static float[] matVec(float[] w, float[] in, int rows, int cols) {
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

    private static void rmsNorm(float[] x, int offset, int size, float[] w) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            ss += x[offset + i] * x[offset + i];
        }
        float inv = (float) (1.0 / Math.sqrt(ss / size + EPS));
        for (int i = 0; i < size; i++) {
            x[offset + i] = w[i] * inv * x[offset + i];
        }
    }

    private static float silu(float v) {
        return v / (float) (1.0 + Math.exp(-v));
    }

    private static float sigmoid(float v) {
        return 1f / (1f + (float) Math.exp(-v));
    }

    /** The MTP block, written out. Its key/value history is whatever the caller supplies. */
    private float[] referenceMtp(
            float[] hidden, int token, int position, float[] keyCache, float[] valueCache) {
        float[] concat = new float[2 * DIM];
        System.arraycopy(tokenEmbd, token * DIM, concat, 0, DIM);
        rmsNorm(concat, 0, DIM, eNorm);
        System.arraycopy(hidden, 0, concat, DIM, DIM);
        rmsNorm(concat, DIM, DIM, hNorm);

        float[] x = matVec(ehProj, concat, DIM, 2 * DIM);

        float[] normed = x.clone();
        rmsNorm(normed, 0, DIM, attnNorm);

        float[] fused = matVec(wq, normed, Q_GATE_DIM, DIM);
        float[] k = matVec(wk, normed, KV_DIM, DIM);
        float[] v = matVec(wv, normed, KV_DIM, DIM);
        float[] q = new float[ATTN_DIM];
        float[] gate = new float[ATTN_DIM];
        for (int h = 0; h < HEADS; h++) {
            System.arraycopy(fused, h * 2 * HEAD_DIM, q, h * HEAD_DIM, HEAD_DIM);
            System.arraycopy(fused, h * 2 * HEAD_DIM + HEAD_DIM, gate, h * HEAD_DIM, HEAD_DIM);
        }
        for (int h = 0; h < HEADS; h++) {
            rmsNorm(q, h * HEAD_DIM, HEAD_DIM, qNorm);
        }
        for (int h = 0; h < KV_HEADS; h++) {
            rmsNorm(k, h * HEAD_DIM, HEAD_DIM, kNorm);
        }
        rope(q, HEADS, position);
        rope(k, KV_HEADS, position);
        System.arraycopy(k, 0, keyCache, position * KV_DIM, KV_DIM);
        System.arraycopy(v, 0, valueCache, position * KV_DIM, KV_DIM);

        int kvMul = HEADS / KV_HEADS;
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[] attn = new float[ATTN_DIM];
        for (int h = 0; h < HEADS; h++) {
            int kvBase = (h / kvMul) * HEAD_DIM;
            float[] scores = new float[position + 1];
            float max = Float.NEGATIVE_INFINITY;
            for (int tt = 0; tt <= position; tt++) {
                float dot = 0f;
                for (int i = 0; i < HEAD_DIM; i++) {
                    dot += q[h * HEAD_DIM + i] * keyCache[tt * KV_DIM + kvBase + i];
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
                    attn[h * HEAD_DIM + i] += a * valueCache[tt * KV_DIM + kvBase + i];
                }
            }
        }
        for (int i = 0; i < ATTN_DIM; i++) {
            attn[i] *= sigmoid(gate[i]);
        }
        float[] projected = matVec(wo, attn, DIM, ATTN_DIM);
        for (int i = 0; i < DIM; i++) {
            x[i] += projected[i];
        }

        float[] ffnIn = x.clone();
        rmsNorm(ffnIn, 0, DIM, ffnNorm);
        float[] g = matVec(ffnGate, ffnIn, HIDDEN, DIM);
        float[] u = matVec(ffnUp, ffnIn, HIDDEN, DIM);
        float[] hid = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) {
            hid[i] = silu(g[i]) * u[i];
        }
        float[] d = matVec(ffnDown, hid, DIM, HIDDEN);
        for (int i = 0; i < DIM; i++) {
            x[i] += d[i];
        }

        rmsNorm(x, 0, DIM, sharedHeadNorm);
        return matVec(output, x, VOCAB, DIM);
    }

    private static void rope(float[] vec, int heads, int position) {
        int rot = 4;
        int half = rot / 2;
        for (int h = 0; h < heads; h++) {
            int base = h * HEAD_DIM;
            for (int ic = 0; ic < half; ic++) {
                double freq = 1.0 / Math.pow(THETA, (2.0 * ic) / rot);
                float cos = (float) Math.cos(position * freq);
                float sin = (float) Math.sin(position * freq);
                float v0 = vec[base + ic];
                float v1 = vec[base + ic + half];
                vec[base + ic] = v0 * cos - v1 * sin;
                vec[base + ic + half] = v0 * sin + v1 * cos;
            }
        }
    }

    // ---- checks -------------------------------------------------------------

    @Test
    public void draftHeadMatchesAnIndependentImplementation() {
        Qwen35Configuration c = config();
        Qwen35StandardWeights w = weights();
        Qwen35State state = new Qwen35State(c, -1);

        float[] keyCache = new float[CONTEXT * KV_DIM];
        float[] valueCache = new float[CONTEXT * KV_DIM];

        int[] tokens = {6, 2, 30, 14};
        for (int position = 0; position < tokens.length; position++) {
            // A hidden state the trunk might have produced. Written into the state exactly where
            // the trunk writes it, so the head reads it the same way it would in a real step.
            float[] hidden = noise(DIM, 1.0f);
            for (int i = 0; i < DIM; i++) {
                state.hNextn.setFloat(i, hidden[i]);
            }

            FloatTensor logits = Qwen35Forward.forwardMtp(c, w, state, tokens[position], position);
            float[] expected =
                    referenceMtp(hidden, tokens[position], position, keyCache, valueCache);
            for (int i = 0; i < VOCAB; i++) {
                assertEquals(
                        "logit " + i + " at position " + position,
                        expected[i],
                        logits.getFloat(i),
                        2e-4f);
            }
        }
    }

    /** Drafting is a question, not a step: it must not move the trunk's residual stream. */
    @Test
    public void draftingLeavesTheTrunkAlone() {
        Qwen35Configuration c = config();
        Qwen35StandardWeights w = weights();
        Qwen35State state = new Qwen35State(c, -1);

        for (int i = 0; i < DIM; i++) {
            state.x.setFloat(i, 0.25f * i);
            state.hNextn.setFloat(i, 0.5f - 0.03f * i);
        }
        float[] before = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            before[i] = state.x.getFloat(i);
        }

        Qwen35Forward.forwardMtp(c, w, state, 5, 0);

        for (int i = 0; i < DIM; i++) {
            assertEquals("x[" + i + "]", before[i], state.x.getFloat(i), 0f);
        }
        // The recurrent state is untouched too: the head has no recurrent layers to advance.
        for (int l = 0; l < TRUNK; l++) {
            if (state.convState[l] != null) {
                for (int i = 0; i < state.convState[l].size(); i++) {
                    assertEquals(0f, state.convState[l].getFloat(i), 0f);
                }
            }
        }
    }

    /** The trunk's own logits must survive a draft, or the loop would commit the head's guess. */
    @Test
    public void draftingUsesItsOwnLogitsBuffer() {
        Qwen35Configuration c = config();
        Qwen35StandardWeights w = weights();
        Qwen35State state = new Qwen35State(c, -1);

        for (int i = 0; i < VOCAB; i++) {
            state.logits.setFloat(i, i);
        }
        for (int i = 0; i < DIM; i++) {
            state.hNextn.setFloat(i, 0.1f * i);
        }
        Qwen35Forward.forwardMtp(c, w, state, 3, 0);

        for (int i = 0; i < VOCAB; i++) {
            assertEquals("trunk logit " + i, (float) i, state.logits.getFloat(i), 0f);
        }
        boolean drafted = false;
        for (int i = 0; i < VOCAB; i++) {
            drafted |= state.nextnLogits.getFloat(i) != 0f;
        }
        assertTrue("the draft head produced no logits at all", drafted);
    }
}
