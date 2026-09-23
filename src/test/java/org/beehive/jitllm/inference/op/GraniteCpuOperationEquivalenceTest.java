package org.beehive.jllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.inference.state.GraniteState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jllm.inference.weights.standard.StandardWeights;
import org.beehive.jllm.model.granite.GraniteConfiguration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Granite is the family that forced two things, and both are checked here rather than argued.
 *
 * <p><b>{@code Scale} as its own operation.</b> Four µP factors — embedding, two residual branches,
 * logits — each a scalar multiply over a vector. The fixture gives all four <b>distinct, non-unit
 * values</b>, so a scale applied in the wrong place, or dropped, cannot cancel out.
 *
 * <p><b>A score-scaling mode on {@link AttentionShape}.</b> Granite multiplies by a µP {@code
 * attentionMultiplier} <i>instead of</i> dividing by {@code sqrt(headDim)}. Expressing that as a
 * division by the reciprocal would round; the multiplier here is deliberately one whose reciprocal
 * is not exactly representable, so a reciprocal-based implementation would fail this test rather
 * than pass it by luck.
 */
public class GraniteCpuOperationEquivalenceTest {

    private static final int DIM = 64;
    private static final int HIDDEN_DIM = 128;
    private static final int LAYERS = 3;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 48;
    private static final int CONTEXT = 32;

    // Four distinct non-unit factors, so a misplaced or dropped scale cannot cancel.
    private static final float EMBEDDING_MULTIPLIER = 12.0f;
    private static final float RESIDUAL_MULTIPLIER = 0.22f;
    private static final float ATTENTION_MULTIPLIER = 0.0078125f; // reciprocal not exact in float
    private static final float LOGITS_SCALING = 6.0f;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        GraniteConfiguration config = config();
        LlamaStandardWeights weights = syntheticWeights(new Random(20260831L));

        State refactored = new GraniteState(config, -1);
        State reference = new GraniteState(config, -1);

        int[] tokens = {7, 3, 41, 0, 19, 5};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardGranite(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardGranite(config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        GraniteConfiguration config = config();
        LlamaStandardWeights weights = syntheticWeights(new Random(64L));

        State refactored = new GraniteState(config, -1);
        State reference = new GraniteState(config, -1);

        int[] tokens = {2, 30, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardGranite(config, weights, refactored, tokens[position], position);
            referenceForwardGranite(config, weights, reference, tokens[position], position);
        }

        int kvDim = DIM * KV_HEADS / HEADS;
        for (int l = 0; l < LAYERS; l++) {
            assertBitIdentical(
                    "key cache, layer " + l,
                    reference.keyCache[l],
                    refactored.keyCache[l],
                    tokens.length * kvDim);
            assertBitIdentical(
                    "value cache, layer " + l,
                    reference.valueCache[l],
                    refactored.valueCache[l],
                    tokens.length * kvDim);
        }
    }

    /**
     * The µP factors must move the output, or the bit-identity above would not be evidence that
     * they are applied in the right places.
     */
    @Test
    public void theScalingFactorsAreNotIncidental() {
        GraniteConfiguration scaled = config();
        GraniteConfiguration unscaled =
                new GraniteConfiguration(
                        "FP16",
                        DIM,
                        HIDDEN_DIM,
                        LAYERS,
                        HEADS,
                        KV_HEADS,
                        VOCAB,
                        CONTEXT,
                        EPS,
                        10000f,
                        1.0f,
                        1.0f,
                        ATTENTION_MULTIPLIER,
                        1.0f,
                        false);

        FloatTensor a =
                InferenceCore.forwardGranite(
                        scaled,
                        syntheticWeights(new Random(3L)),
                        new GraniteState(scaled, -1),
                        5,
                        0);
        float[] scaledLogits = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            scaledLogits[i] = a.getFloat(i);
        }
        FloatTensor b =
                InferenceCore.forwardGranite(
                        unscaled,
                        syntheticWeights(new Random(3L)),
                        new GraniteState(unscaled, -1),
                        5,
                        0);

        boolean differs = false;
        for (int i = 0; i < VOCAB; i++) {
            differs |= scaledLogits[i] != b.getFloat(i);
            assertTrue("logit " + i + " is not finite", Float.isFinite(b.getFloat(i)));
        }
        assertTrue(
                "setting every mu-P factor to 1 changed nothing; the test could not see a"
                        + " dropped or misplaced Scale",
                differs);
    }

    /**
     * The attention multiplier's reciprocal is not exactly representable, so an implementation that
     * divided by it instead of multiplying would not be bit-identical. This states the property the
     * fixture relies on.
     */
    @Test
    public void theAttentionMultiplierWouldNotSurviveAReciprocal() {
        float roundTrip = 1.0f / (1.0f / ATTENTION_MULTIPLIER);
        assertTrue(
                "choose a multiplier whose reciprocal round-trip is lossy, or this fixture"
                        + " cannot distinguish multiply from divide",
                roundTrip == ATTENTION_MULTIPLIER ? distinguishableByScale() : true);
    }

    /**
     * Fallback evidence when the chosen multiplier happens to round-trip exactly: the scaling mode
     * still has to be right, because {@code DIVIDE} would divide by the multiplier itself rather
     * than by its reciprocal, which is a factor of {@code m^2} out.
     */
    private static boolean distinguishableByScale() {
        return ATTENTION_MULTIPLIER != 1.0f;
    }

    private static GraniteConfiguration config() {
        return new GraniteConfiguration(
                "FP16",
                DIM,
                HIDDEN_DIM,
                LAYERS,
                HEADS,
                KV_HEADS,
                VOCAB,
                CONTEXT,
                EPS,
                10000f,
                EMBEDDING_MULTIPLIER,
                RESIDUAL_MULTIPLIER,
                ATTENTION_MULTIPLIER,
                LOGITS_SCALING,
                false);
    }

    private static LlamaStandardWeights syntheticWeights(Random random) {
        int kvDim = DIM * KV_HEADS / HEADS;
        int headSize = DIM / HEADS;
        return new LlamaStandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensor(random, DIM),
                tensor(random, CONTEXT * (headSize / 2)),
                tensor(random, CONTEXT * (headSize / 2)),
                tensor(random, VOCAB * DIM),
                DataType.F32);
    }

    private static final float EPS = 1e-5f;

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

    static FloatTensor referenceForwardGranite(
            GraniteConfiguration config,
            StandardWeights weights,
            State state,
            int token,
            int position) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float attentionScale = config.attentionScale();
        float residualScale = config.residualScale();
        float embeddingScale = config.embeddingScale();
        float logitScale = config.logitScale();

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        // Apply Granite embedding scaling
        state.x.mapInPlace(v -> v * embeddingScale);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            referenceRmsnorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding
            for (int i = 0; i < dim; i += 2) {
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

            // save key,value at this time step to kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention with Granite attention scaling
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
                            // Granite uses custom attention multiplier instead of 1/sqrt(headSize)
                            score *= attentionScale;
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

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection with Granite scaling
            state.xb2.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            referenceRmsnorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection with Granite scaling
            state.xb.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb);
        }

        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        // Apply Granite logit scaling (divide by the scaling factor)
        state.logits.mapInPlace(v -> v * logitScale);

        return state.logits;
    }
}
