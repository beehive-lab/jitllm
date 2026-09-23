package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jitllm.inference.weights.standard.StandardWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Devstral's distinguishing property is that its head dimension is <b>independent of {@code dim /
 * heads}</b>. The fixture makes that real: {@code dim = 64} with 4 heads gives 16, while {@code
 * headDim = 8}, so {@code qDim = 32 != dim}. An implementation that derived the head size, rotated
 * over {@code dim} instead of {@code qDim}, or sized the output projection from {@code dim} would
 * fail here rather than pass by coincidence.
 *
 * <p><b>{@code qDim} must stay {@code <= dim}</b>, which is a real constraint of the family rather
 * than a convenience: {@code DevstralState} allocates {@code xb} at {@code dim} and attention
 * writes {@code heads * headDim} into it. Real Devstral satisfies it comfortably — 4096 against
 * 5120 — and a fixture that did not would fail identically on both sides of this comparison.
 *
 * <p><b>Mistral has no test of its own on purpose.</b> It routes to {@code forwardJava}, so it is
 * covered by the Llama slice's test — the two families share a weight layout and a forward pass.
 */
public class DevstralCpuOperationEquivalenceTest {

    private static final int DIM = 64;
    private static final int HIDDEN_DIM = 96;
    private static final int LAYERS = 3;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 8; // deliberately != DIM / HEADS, which is 16
    private static final int Q_DIM = HEADS * HEAD_DIM; // 32, and must stay <= DIM
    private static final int KV_DIM = KV_HEADS * HEAD_DIM; // 16
    private static final int VOCAB = 40;
    private static final int CONTEXT = 24;
    private static final float EPS = 1e-5f;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        DevstralConfiguration config = config();
        StandardWeights weights = syntheticWeights(new Random(20260831L));

        State refactored = new DevstralState(config, -1);
        State reference = new DevstralState(config, -1);

        int[] tokens = {7, 3, 31, 0, 19};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaDevstral(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaDevstral(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        DevstralConfiguration config = config();
        StandardWeights weights = syntheticWeights(new Random(48L));

        State refactored = new DevstralState(config, -1);
        State reference = new DevstralState(config, -1);

        int[] tokens = {2, 21, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaDevstral(
                    config, weights, refactored, tokens[position], position);
            referenceForwardJavaDevstral(config, weights, reference, tokens[position], position);
        }

        for (int l = 0; l < LAYERS; l++) {
            assertBitIdentical(
                    "key cache, layer " + l,
                    reference.keyCache[l],
                    refactored.keyCache[l],
                    tokens.length * KV_DIM);
            assertBitIdentical(
                    "value cache, layer " + l,
                    reference.valueCache[l],
                    refactored.valueCache[l],
                    tokens.length * KV_DIM);
        }
    }

    /** The property the fixture depends on, stated so it cannot drift. */
    @Test
    public void theHeadDimensionIsIndependentOfDimOverHeads() {
        DevstralConfiguration config = config();
        assertTrue(
                "headSize must not equal dim / heads, or the fixture proves nothing",
                config.headSize() != DIM / HEADS);
        assertTrue("qDim must not equal dim", config.qDim() != DIM);

        FloatTensor logits =
                InferenceCore.forwardJavaDevstral(
                        config,
                        syntheticWeights(new Random(1L)),
                        new DevstralState(config, -1),
                        5,
                        0);
        float first = logits.getFloat(0);
        boolean varies = false;
        for (int i = 0; i < VOCAB; i++) {
            assertTrue("logit " + i + " is not finite", Float.isFinite(logits.getFloat(i)));
            varies |= logits.getFloat(i) != first;
        }
        assertTrue("every logit is identical; the comparison would prove nothing", varies);
    }

    private static DevstralConfiguration config() {
        return new DevstralConfiguration(
                "FP16",
                DIM,
                HIDDEN_DIM,
                LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                VOCAB,
                CONTEXT,
                EPS,
                10000f);
    }

    private static StandardWeights syntheticWeights(Random random) {
        return new LlamaStandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, Q_DIM * DIM),
                tensors(random, LAYERS, KV_DIM * DIM),
                tensors(random, LAYERS, KV_DIM * DIM),
                tensors(random, LAYERS, DIM * Q_DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensor(random, DIM),
                tensor(random, CONTEXT * (HEAD_DIM / 2)),
                tensor(random, CONTEXT * (HEAD_DIM / 2)),
                tensor(random, VOCAB * DIM),
                DataType.F32);
    }

    private static FloatTensor tensor(Random random, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (random.nextFloat() - 0.5f) * 0.5f;
        }
        return new ArrayFloatTensor(values);
    }

    private static FloatTensor[] tensors(Random random, int count, int size) {
        FloatTensor[] out = new FloatTensor[count];
        for (int i = 0; i < count; i++) {
            out[i] = tensor(random, size);
        }
        return out;
    }

    private static void assertBitIdentical(
            String what, FloatTensor expected, FloatTensor actual, int size) {
        for (int i = 0; i < size; i++) {
            float e = expected.getFloat(i);
            float a = actual.getFloat(i);
            if (Float.floatToRawIntBits(e) != Float.floatToRawIntBits(a)) {
                assertEquals(what + ", element " + i + " differs", e, a, 0f);
            }
        }
    }

    private static void referenceRmsnorm(
            FloatTensor out,
            FloatTensor x,
            FloatTensor weight,
            int offset,
            int size,
            float rmsNormEps) {
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(
                offset,
                size,
                (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    static FloatTensor referenceForwardJavaDevstral(
            DevstralConfiguration config,
            StandardWeights weights,
            State state,
            int token,
            int position) {
        int dim = config.dim();
        int headSize = config.headSize(); // 128 (independent head_dim)
        int qDim = config.qDim(); // 4096 = 32 * 128
        int kvDim = config.kvDim(); // 1024 = 8 * 128
        int kvMul = config.kvMul();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            referenceRmsnorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wq[l].matmul(state.xb, state.q, qDim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE over qDim (not dim)
            for (int i = 0; i < qDim; i += 2) {
                int head_dim = i % headSize;
                float fcr =
                        weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci =
                        weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            Parallel.parallelFor(
                    0,
                    config.numberOfHeads(),
                    h -> {
                        int qOffset = h * headSize;
                        int attOffset = h * config.contextLength();

                        for (int t = 0; t <= position; t++) {
                            int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                            float score =
                                    state.q.dot(
                                            qOffset,
                                            state.keyCache[curLayer],
                                            keyCacheOffset,
                                            headSize);
                            score /= sqrtHeadSize;
                            state.att.setFloat(attOffset + t, score);
                        }

                        state.att.softmaxInPlace(attOffset, position + 1);

                        int xbOffset = h * headSize;
                        state.xb.fillInPlace(xbOffset, headSize, 0f);

                        for (int t = 0; t <= position; t++) {
                            int vOffset = t * kvDim + (h / kvMul) * headSize;
                            float a = state.att.getFloat(attOffset + t);
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                        }
                    });

            // O projection: input qDim, output dim
            weights.wo[l].matmul(state.xb, state.xb2, dim, qDim);

            state.x.addInPlace(state.xb2);

            referenceRmsnorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());
            state.x.addInPlace(state.xb);
        }

        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }
}
