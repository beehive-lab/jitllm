package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.weights.standard.Phi3StandardWeights;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * Phi3 is the family with two fused projections — QKV in one matrix and gate-up in another — and
 * the interesting claim this test checks is that its rotation is the <b>same {@code NEOX_HALF}
 * layout the Qwen families use</b>, written differently. The pre-refactor code steps {@code i} over
 * the whole dimension and derives a head base; {@code CpuOperations.ropeNeox} loops heads directly.
 * If that reading is wrong, this fails.
 *
 * <p>Grouped query attention with four query heads against two key/value heads, so the {@code i <
 * kvDim} / {@code h < keyValueHeads} equivalence is exercised on both sides of the branch.
 */
public class Phi3CpuOperationEquivalenceTest {

    private static final int DIM = 64;
    private static final int HIDDEN_DIM = 128;
    private static final int LAYERS = 3;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 48;
    private static final int CONTEXT = 32;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        Phi3Configuration config = config();
        Phi3StandardWeights weights = syntheticWeights(new Random(20260831L));

        Phi3State refactored = new Phi3State(config, -1);
        Phi3State reference = new Phi3State(config, -1);

        int[] tokens = {7, 3, 41, 0, 19, 5};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaPhi3(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaPhi3(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        Phi3Configuration config = config();
        Phi3StandardWeights weights = syntheticWeights(new Random(77L));

        Phi3State refactored = new Phi3State(config, -1);
        Phi3State reference = new Phi3State(config, -1);

        int[] tokens = {2, 30, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaPhi3(config, weights, refactored, tokens[position], position);
            referenceForwardJavaPhi3(config, weights, reference, tokens[position], position);
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

    /** The output must vary, or bit-identity would be a comparison of two empty buffers. */
    @Test
    public void theSyntheticModelProducesRealOutput() {
        Phi3Configuration config = config();
        FloatTensor logits =
                InferenceCore.forwardJavaPhi3(
                        config, syntheticWeights(new Random(1L)), new Phi3State(config, -1), 5, 0);
        float first = logits.getFloat(0);
        boolean varies = false;
        for (int i = 0; i < VOCAB; i++) {
            assertTrue("logit " + i + " is not finite", Float.isFinite(logits.getFloat(i)));
            varies |= logits.getFloat(i) != first;
        }
        assertTrue("every logit is identical; the comparison would prove nothing", varies);
    }

    private static Phi3Configuration config() {
        return new Phi3Configuration(
                "FP16", DIM, HIDDEN_DIM, LAYERS, HEADS, KV_HEADS, VOCAB, CONTEXT, EPS, 10000f);
    }

    private static Phi3StandardWeights syntheticWeights(Random random) {
        int headSize = DIM / HEADS;
        int opSize = DIM + 2 * (KV_HEADS * headSize);
        return new Phi3StandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, opSize * DIM),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, 2 * HIDDEN_DIM * DIM),
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

    private static void referenceCopyChunk(
            FloatTensor in, FloatTensor out, int dim1In, int dim1Out, int nChunks, int chunkNo) {
        assert (dim1In == dim1Out * nChunks);
        final int startOffsetInDim1 = chunkNo * dim1Out;
        Parallel.parallelFor(
                0,
                dim1Out,
                i -> {
                    out.setFloat(i, in.getFloat(startOffsetInDim1 + i));
                });
    }

    static FloatTensor referenceForwardJavaPhi3(
            Phi3Configuration config,
            Phi3StandardWeights weights,
            Phi3State state,
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

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        final int opSize = dim + 2 * (config.numberOfKeyValueHeads() * headSize);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            referenceRmsnorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wqkv[l].matmul(state.xb, state.qkv, opSize, dim);
            state.qkv.copyTo(0, state.q, 0, dim);
            // key_states = qkv[., query_pos: query_pos + self.num_key_value_heads *
            // self.head_dim]
            state.qkv.copyTo(dim, state.k, 0, config.numberOfKeyValueHeads() * headSize);
            // value_states = qkv[., query_pos + self.num_key_value_heads * self.head_dim:]
            state.qkv.copyTo(
                    dim + config.numberOfKeyValueHeads() * headSize,
                    state.v,
                    0,
                    config.numberOfKeyValueHeads() * headSize);

            int dimHalf = headSize / 2;
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                int base = i - head_dim;
                int ic = base + head_dim / 2;
                float fcr =
                        weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci =
                        weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec =
                            v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(ic);
                    float v1 = vec.getFloat(ic + dimHalf);
                    vec.setFloat(ic, v0 * fcr - v1 * fci);
                    vec.setFloat(ic + dimHalf, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
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
                            int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
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
                            int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                            float a = state.att.getFloat(attOffset + t);
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                        }
                    });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            referenceRmsnorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.wGateUp[l].matmul(state.xb, state.hb, 2 * config.hiddenDim(), dim);
            referenceCopyChunk(
                    state.hb, state.hbG, 2 * config.hiddenDim(), config.hiddenDim(), 2, 0);
            referenceCopyChunk(
                    state.hb, state.hbU, 2 * config.hiddenDim(), config.hiddenDim(), 2, 1);

            state.hbG.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            state.hbU.multiplyInPlace(state.hbG);

            weights.wDown[l].matmul(state.hbU, state.xb, dim, config.hiddenDim());

            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }
}
