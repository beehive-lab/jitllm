package org.beehive.jllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jllm.inference.weights.standard.StandardWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * The comparison is <b>exact</b>. Not a tolerance, not a cosine similarity: the same arithmetic in
 * the same order on the same inputs must produce the same bits, and anything less would let a
 * reassociated sum through. The one thing that could legitimately differ is the parallel reduction
 * order inside {@code parallelFor}, which is why the same call sits in both paths rather than being
 * restructured.
 *
 * <p>The model is synthetic and tiny — deterministic pseudo-random weights, four layers, two
 * key/value heads against four query heads so grouped query attention is actually exercised. No
 * GGUF file, no device, so this runs in the ordinary test suite rather than behind {@code
 * -Paccel-tests}. It pins the operation decomposition, not the model; the goldens do that.
 */
public class LlamaCpuOperationEquivalenceTest {

    private static final int DIM = 64;
    private static final int HIDDEN_DIM = 128;
    private static final int LAYERS = 4;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2; // grouped query attention: kvMul = 2
    private static final int VOCAB = 48;
    private static final int CONTEXT = 32;
    private static final float EPS = 1e-5f;

    /**
     * The refactored path and the pre-refactor path agree bit for bit, over several positions.
     *
     * <p>Several positions because attention is the step whose work grows with the position: a
     * single-token comparison would exercise one score and one weighted value, which is exactly the
     * case where an off-by-one in the {@code 0.position} bound cannot be seen.
     */
    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        LlamaConfiguration config = config();
        LlamaStandardWeights weights = syntheticWeights(config, new Random(20260831L));

        State refactored = new LlamaState(config, -1);
        State reference = new LlamaState(config, -1);

        int[] tokens = {7, 3, 41, 0, 19, 5, 33, 12};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJava(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJava(config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    /**
     * The key/value caches agree too, not just the logits.
     *
     * <p>Logits at the last position could match while an earlier cache write went to the wrong
     * offset, if the error happened to fall outside the attended range. Comparing the caches closes
     * that: the append moved into {@link CpuOperations#attention}, and this is what proves it
     * landed in the same place.
     */
    @Test
    public void theKeyValueCachesAgree() {
        LlamaConfiguration config = config();
        LlamaStandardWeights weights = syntheticWeights(config, new Random(4242L));

        State refactored = new LlamaState(config, -1);
        State reference = new LlamaState(config, -1);

        int[] tokens = {2, 30, 11, 8, 25};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJava(config, weights, refactored, tokens[position], position);
            referenceForwardJava(config, weights, reference, tokens[position], position);
        }

        int kvDim = config.kvDim();
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
     * A guard against the test fooling itself: the synthetic model must produce logits that are not
     * all zero and not all equal, or "bit-identical" would be a comparison of two empty buffers.
     */
    @Test
    public void theSyntheticModelProducesRealOutput() {
        LlamaConfiguration config = config();
        LlamaStandardWeights weights = syntheticWeights(config, new Random(1L));
        FloatTensor logits =
                InferenceCore.forwardJava(config, weights, new LlamaState(config, -1), 5, 0);

        float first = logits.getFloat(0);
        boolean varies = false;
        for (int i = 0; i < VOCAB; i++) {
            float v = logits.getFloat(i);
            assertTrue("logit " + i + " is not finite: " + v, Float.isFinite(v));
            varies |= v != first;
        }
        assertTrue("every logit is identical; the comparison would prove nothing", varies);
    }

    // the model

    private static LlamaConfiguration config() {
        return new LlamaConfiguration(
                "FP16", DIM, HIDDEN_DIM, LAYERS, HEADS, KV_HEADS, VOCAB, CONTEXT, EPS, 10000f);
    }

    private static LlamaStandardWeights syntheticWeights(LlamaConfiguration config, Random random) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int headSize = config.headSize();
        return new LlamaStandardWeights(
                tensor(random, VOCAB * dim),
                tensors(random, LAYERS, dim),
                tensors(random, LAYERS, dim * dim),
                tensors(random, LAYERS, kvDim * dim),
                tensors(random, LAYERS, kvDim * dim),
                tensors(random, LAYERS, dim * dim),
                tensors(random, LAYERS, dim),
                tensors(random, LAYERS, HIDDEN_DIM * dim),
                tensors(random, LAYERS, dim * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * dim),
                tensor(random, dim),
                tensor(random, CONTEXT * (headSize / 2)),
                tensor(random, CONTEXT * (headSize / 2)),
                tensor(random, VOCAB * dim),
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

    static FloatTensor referenceForwardJava(
            Configuration config, StandardWeights weights, State state, int token, int position) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul =
                config.numberOfHeads()
                        / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in
        // multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            referenceRmsnorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position

            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr =
                        weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci =
                        weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec =
                            v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            // int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

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
                            // float* v = s.value_cache + loff + t * dim + h * headSize;
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
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

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

        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }
}
