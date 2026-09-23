package org.beehive.jllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Qwen3 is the family that forced {@link AttentionShape}: {@code attention.key_length} and {@code
 * attention.value_length} are separate metadata. The synthetic model therefore sets <b>{@code
 * numberOfHeadsKey} != {@code numberOfHeadsValue}</b> — 16 against 8 — so a shape that collapsed
 * them to one head size would address the value cache with the key length and fail here. Neither
 * equals {@code dim / heads}, which is 12, so an implementation that quietly derived the head size
 * would fail too.
 *
 * <p>Per-head query and key normalization is exercised by having four query heads and two key/value
 * heads: the two loops run different numbers of times.
 */
public class Qwen3CpuOperationEquivalenceTest {

    private static final int DIM = 48;
    private static final int HIDDEN_DIM = 96;
    private static final int LAYERS = 3;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_K = 16; // attention.key_length
    private static final int HEAD_V = 8; // attention.value_length, deliberately different
    private static final int VOCAB = 40;
    private static final int CONTEXT = 24;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        Qwen3Configuration config = config();
        Qwen3StandardWeights weights = syntheticWeights(new Random(20260831L));

        State refactored = new Qwen3State(config, -1);
        State reference = new Qwen3State(config, -1);

        int[] tokens = {7, 3, 31, 0, 19};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaQwen3(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaQwen3(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        Qwen3Configuration config = config();
        Qwen3StandardWeights weights = syntheticWeights(new Random(31L));

        State refactored = new Qwen3State(config, -1);
        State reference = new Qwen3State(config, -1);

        int[] tokens = {2, 21, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaQwen3(config, weights, refactored, tokens[position], position);
            referenceForwardJavaQwen3(config, weights, reference, tokens[position], position);
        }

        int nEmbdGqa = HEAD_V * KV_HEADS;
        for (int l = 0; l < LAYERS; l++) {
            assertBitIdentical(
                    "key cache, layer " + l,
                    reference.keyCache[l],
                    refactored.keyCache[l],
                    tokens.length * nEmbdGqa);
            assertBitIdentical(
                    "value cache, layer " + l,
                    reference.valueCache[l],
                    refactored.valueCache[l],
                    tokens.length * nEmbdGqa);
        }
    }

    /**
     * The shape this family needs is not derivable from {@code dim / heads}, which is the mistake a
     * single head-size parameter would bake in.
     */
    @Test
    public void theKeyAndValueHeadWidthsAreGenuinelyDifferent() {
        Qwen3Configuration config = config();
        assertTrue(
                "the fixture must not let key and value lengths coincide",
                config.numberOfHeadsKey() != config.numberOfHeadsValue());
        assertTrue(
                "neither may equal dim / heads, or the shape would be derivable",
                config.numberOfHeadsKey() != DIM / HEADS
                        && config.numberOfHeadsValue() != DIM / HEADS);

        FloatTensor logits =
                InferenceCore.forwardJavaQwen3(
                        config, syntheticWeights(new Random(1L)), new Qwen3State(config, -1), 5, 0);
        float first = logits.getFloat(0);
        boolean varies = false;
        for (int i = 0; i < VOCAB; i++) {
            assertTrue("logit " + i + " is not finite", Float.isFinite(logits.getFloat(i)));
            varies |= logits.getFloat(i) != first;
        }
        assertTrue("every logit is identical; the comparison would prove nothing", varies);
    }

    private static Qwen3Configuration config() {
        return new Qwen3Configuration(
                "FP16",
                DIM,
                HIDDEN_DIM,
                LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_K,
                HEAD_V,
                VOCAB,
                CONTEXT,
                CONTEXT,
                false,
                EPS,
                10000f);
    }

    private static Qwen3StandardWeights syntheticWeights(Random random) {
        int nEmbdGqa = HEAD_V * KV_HEADS;
        return new Qwen3StandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HEAD_K * HEADS * DIM),
                tensors(random, LAYERS, nEmbdGqa * DIM),
                tensors(random, LAYERS, nEmbdGqa * DIM),
                tensors(random, LAYERS, DIM * HEAD_K * HEADS),
                tensors(random, LAYERS, HEAD_V),
                tensors(random, LAYERS, HEAD_V),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensor(random, DIM),
                tensor(random, CONTEXT * (HEAD_V / 2)),
                tensor(random, CONTEXT * (HEAD_V / 2)),
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

    static FloatTensor referenceForwardJavaQwen3(
            Qwen3Configuration config,
            Qwen3StandardWeights weights,
            State state,
            int token,
            int position) {
        int dim = config.dim();
        int nHeadKv = config.numberOfKeyValueHeads(); // n_head_kv = numberOfKeyValueHeads
        int nEmbdHeadK = config.numberOfHeadsKey(); // n_embd_head_k = n_embd / n_head;
        // %s.attention.key_length
        int nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v = n_embd / n_head;
        // %s.attention.value_length
        int nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        int nEmbdHead = nEmbdHeadV;
        int nEmbdGqa = nEmbdVGqa;
        int gqa =
                config.numberOfHeads()
                        / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in
        // multiquery
        float sqrtHeadSize = (float) Math.sqrt(nEmbdHead);

        // copy the token embedding into x
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
            weights.wq[curLayer].matmul(
                    state.xb, state.q, nEmbdHeadK * config.numberOfHeads(), dim);
            weights.wk[curLayer].matmul(state.xb, state.k, nEmbdGqa, dim);
            weights.wv[curLayer].matmul(state.xb, state.v, nEmbdGqa, dim);

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            for (int i = 0; i < config.numberOfHeads(); i++) {
                referenceRmsnorm(
                        state.q,
                        state.q,
                        weights.attnQNorm[curLayer],
                        i * nEmbdHead,
                        nEmbdHead,
                        config.rmsNormEps());
            }
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                referenceRmsnorm(
                        state.k,
                        state.k,
                        weights.attnKNorm[curLayer],
                        i * nEmbdHead,
                        nEmbdHead,
                        config.rmsNormEps());
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset
            // per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn =
                        h < config.numberOfKeyValueHeads()
                                ? 2
                                : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * nEmbdHead;
                int nComplEmbdHead = nEmbdHead / 2;
                for (int ic = 0; ic < nComplEmbdHead; ic++) {
                    float fcr = weights.freq_cis_real.getFloat(position * nComplEmbdHead + ic);
                    float fci = weights.freq_cis_imag.getFloat(position * nComplEmbdHead + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec =
                                (vi == 0)
                                        ? state.q
                                        : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + nComplEmbdHead);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + nComplEmbdHead, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            // int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * nEmbdGqa, nEmbdGqa);
            state.v.copyTo(0, state.valueCache[curLayer], position * nEmbdGqa, nEmbdGqa);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(
                    0,
                    config.numberOfHeads(),
                    h -> {
                        // get the query vector for this head
                        int qOffset = h * nEmbdHead;
                        // attention scores for this head
                        int attOffset = h * config.contextLength();

                        // iterate over all timesteps, including the current one
                        for (int t = 0; t <= position; t++) {
                            // get the key vector for this head and at this timestep
                            int keyCacheOffset = /* loff + */
                                    (t * nEmbdGqa + (h / gqa) * nEmbdHead);
                            // calculate the attention score as the dot product of q and k
                            float score =
                                    state.q.dot(
                                            qOffset,
                                            state.keyCache[curLayer],
                                            keyCacheOffset,
                                            nEmbdHeadK);
                            score /= sqrtHeadSize;
                            // save the score to the attention buffer
                            state.att.setFloat(attOffset + t, score);
                        }

                        // softmax the scores to get attention weights, from 0.position inclusively
                        state.att.softmaxInPlace(attOffset, position + 1); // position + 0 + 1

                        // weighted sum of the values, store back into xb
                        int xbOffset = h * nEmbdHeadV;
                        state.xb.fillInPlace(xbOffset, nEmbdHeadV, 0f);

                        for (int t = 0; t <= position; t++) {
                            // get the value vector for this head and at this timestep
                            int vOffset = /* loff + */ t * nEmbdGqa + (h / gqa) * nEmbdHeadV;
                            // get the attention weight for this timestep
                            float a = state.att.getFloat(attOffset + t);
                            // accumulate the weighted value into xb
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[curLayer], vOffset, nEmbdHeadV, a);
                        }
                    });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, nEmbdHeadK * config.numberOfHeads());

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
