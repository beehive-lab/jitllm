package org.beehive.jllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.inference.state.Qwen2State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.jllm.model.qwen2.Qwen2Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Qwen2 is the family that motivated {@code BiasAdd} and {@code RopeLayout.NEOX_HALF}, so the
 * synthetic model gives every bias a non-zero value — a bias add that was dropped entirely would
 * pass against zeros — and uses four query heads against two key/value heads, so the {@code h <
 * keyValueHeads} guard in the NeoX rotation is exercised on both sides of the branch.
 */
public class Qwen2CpuOperationEquivalenceTest {

    private static final int DIM = 64;
    private static final int HIDDEN_DIM = 128;
    private static final int LAYERS = 3;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 48;
    private static final int CONTEXT = 32;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        Qwen2Configuration config = config();
        Qwen2StandardWeights weights = syntheticWeights(new Random(20260831L));

        State refactored = new Qwen2State(config, -1);
        State reference = new Qwen2State(config, -1);

        int[] tokens = {7, 3, 41, 0, 19, 5};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaQwen2(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaQwen2(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        Qwen2Configuration config = config();
        Qwen2StandardWeights weights = syntheticWeights(new Random(99L));

        State refactored = new Qwen2State(config, -1);
        State reference = new Qwen2State(config, -1);

        int[] tokens = {2, 30, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaQwen2(config, weights, refactored, tokens[position], position);
            referenceForwardJavaQwen2(config, weights, reference, tokens[position], position);
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

    /** The biases must actually move the result, or dropping them would go unnoticed. */
    @Test
    public void theBiasesAreNotIncidental() {
        Qwen2Configuration config = config();
        Random seed = new Random(5L);
        Qwen2StandardWeights withBias = syntheticWeights(new Random(5L));
        Qwen2StandardWeights withoutBias = syntheticWeights(new Random(5L), true);

        FloatTensor a =
                InferenceCore.forwardJavaQwen2(config, withBias, new Qwen2State(config, -1), 5, 0);
        float[] withBiasLogits = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            withBiasLogits[i] = a.getFloat(i);
        }
        FloatTensor b =
                InferenceCore.forwardJavaQwen2(
                        config, withoutBias, new Qwen2State(config, -1), 5, 0);

        boolean differs = false;
        for (int i = 0; i < VOCAB; i++) {
            differs |= withBiasLogits[i] != b.getFloat(i);
            assertTrue("logit " + i + " is not finite", Float.isFinite(b.getFloat(i)));
        }
        assertTrue(
                "zeroing the QKV biases changed nothing; the test could not see a dropped"
                        + " BiasAdd",
                differs);
    }

    private static Qwen2Configuration config() {
        return new Qwen2Configuration(
                "FP16",
                DIM,
                HIDDEN_DIM,
                LAYERS,
                HEADS,
                KV_HEADS,
                DIM / HEADS,
                DIM / HEADS,
                VOCAB,
                CONTEXT,
                CONTEXT,
                false,
                EPS,
                10000f);
    }

    private static Qwen2StandardWeights syntheticWeights(Random random) {
        return syntheticWeights(random, false);
    }

    private static Qwen2StandardWeights syntheticWeights(Random random, boolean zeroBias) {
        int kvDim = DIM * KV_HEADS / HEADS;
        int headSize = DIM / HEADS;
        FloatTensor[] qBias = zeroBias ? zeros(LAYERS, DIM) : tensors(random, LAYERS, DIM);
        FloatTensor[] kBias = zeroBias ? zeros(LAYERS, kvDim) : tensors(random, LAYERS, kvDim);
        FloatTensor[] vBias = zeroBias ? zeros(LAYERS, kvDim) : tensors(random, LAYERS, kvDim);
        if (zeroBias) {
            // consume the same draws so every other weight matches the non-zero-bias model
            tensors(random, LAYERS, DIM);
            tensors(random, LAYERS, kvDim);
            tensors(random, LAYERS, kvDim);
        }
        return new Qwen2StandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                qBias,
                kBias,
                vBias,
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensor(random, DIM),
                arrayTensor(random, CONTEXT * (headSize / 2)),
                arrayTensor(random, CONTEXT * (headSize / 2)),
                tensor(random, VOCAB * DIM),
                DataType.F32);
    }

    private static FloatTensor[] zeros(int count, int size) {
        FloatTensor[] out = new FloatTensor[count];
        for (int i = 0; i < count; i++) {
            out[i] = new ArrayFloatTensor(new float[size]);
        }
        return out;
    }

    private static final float EPS = 1e-5f;

    private static FloatTensor tensor(Random random, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (random.nextFloat() - 0.5f) * 0.5f;
        }
        return new ArrayFloatTensor(values);
    }

    private static ArrayFloatTensor arrayTensor(Random random, int size) {
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

    static FloatTensor referenceForwardJavaQwen2(
            Qwen2Configuration config,
            Qwen2StandardWeights weights,
            State state,
            int token,
            int position) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul =
                config.numberOfHeads()
                        / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in
        // multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            referenceRmsnorm(
                    state.xb,
                    state.x,
                    weights.rms_att_weight[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // qkv additions with qkv bias
            state.q.addInPlace(weights.q_bias[curLayer]);
            state.k.addInPlace(weights.k_bias[curLayer]);
            state.v.addInPlace(weights.v_bias[curLayer]);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset
            // per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn =
                        h < config.numberOfKeyValueHeads()
                                ? 2
                                : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = weights.freq_cis_real.getFloat((position) * (headSize / 2) + ic);
                    float fci = weights.freq_cis_imag.getFloat((position) * (headSize / 2) + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec =
                                (vi == 0)
                                        ? state.q
                                        : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + headSize / 2);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + headSize / 2, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            // int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[curLayer], position * kvDim, kvDim);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(
                    0,
                    config.numberOfHeads(),
                    h -> {
                        // get the query vector for this head
                        // float* q = s.q + h * headSize;
                        int qOffset = h * headSize;

                        // attention scores for this head
                        // float* att = s.att + h * config.seq_len;
                        int attOffset = h * config.contextLength();

                        // iterate over all timesteps, including the current one
                        for (int t = 0; t <= position; t++) {
                            // get the key vector for this head and at this timestep
                            // float* k = s.key_cache + loff + t * dim + h * headSize;
                            int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                            // calculate the attention score as the dot product of q and k
                            float score =
                                    state.q.dot(
                                            qOffset,
                                            state.keyCache[curLayer],
                                            keyCacheOffset,
                                            headSize);
                            score /= sqrtHeadSize;
                            // save the score to the attention buffer
                            state.att.setFloat(attOffset + t, score);
                        }

                        // softmax the scores to get attention weights, from 0.position inclusively
                        state.att.softmaxInPlace(attOffset, position + 1);

                        // weighted sum of the values, store back into xb
                        // float* xb = s.xb + h * headSize;
                        int xbOffset = h * headSize;
                        // memset(xb, 0, headSize * sizeof(float));
                        state.xb.fillInPlace(xbOffset, headSize, 0f);

                        for (int t = 0; t <= position; t++) {
                            // get the value vector for this head and at this timestep
                            // float* v = s.value_cache + loff + t * dim + h * headSize;C
                            int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                            // get the attention weight for this timestep
                            float a = state.att.getFloat(attOffset + t);
                            // accumulate the weighted value into xb
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                        }
                    });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            referenceRmsnorm(
                    state.xb,
                    state.x,
                    weights.rms_ffn_weight[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }
}
