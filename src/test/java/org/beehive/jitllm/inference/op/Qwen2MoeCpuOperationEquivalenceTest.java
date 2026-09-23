package org.beehive.jitllm.inference.op;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;
import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.standard.Qwen2MoEStandardWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

public class Qwen2MoeCpuOperationEquivalenceTest {

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

    private static final int DIM = 48;
    private static final int HIDDEN_DIM = 96;
    private static final int LAYERS = 2;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 40;
    private static final int CONTEXT = 24;
    private static final int EXPERTS = 6;
    private static final int TOP_K = 3;
    private static final int MOE_HIDDEN = 32;
    private static final int SHARED_HIDDEN = 24;
    private static final float EPS = 1e-5f;

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(20260831L));

        Qwen2MoEState refactored = new Qwen2MoEState(config, -1);
        Qwen2MoEState reference = new Qwen2MoEState(config, -1);

        int[] tokens = {7, 3, 31, 0, 19};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaQwen2MoE(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaQwen2MoE(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(606L));

        Qwen2MoEState refactored = new Qwen2MoEState(config, -1);
        Qwen2MoEState reference = new Qwen2MoEState(config, -1);

        int[] tokens = {2, 21, 11, 8};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaQwen2MoE(
                    config, weights, refactored, tokens[position], position);
            referenceForwardJavaQwen2MoE(config, weights, reference, tokens[position], position);
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
     * Deliberately tied router scores: every expert scores identically, so the selection is decided
     * entirely by the tie rule. Lowest index first, in order.
     *
     * <p>Ties are not hypothetical — a router whose projection weight is uniform, or whose input is
     * zero, produces them, and an implementation using {@code >=} instead of {@code >} would select
     * the <i>highest</i> indices and still look plausible.
     */
    @Test
    public void tiedRouterScoresSelectTheLowestExpertIndices() {
        FloatTensor scores = ArrayFloatTensor.allocate(EXPERTS);
        FloatTensor uniformRouter = uniform(EXPERTS * DIM, 0.25f);
        FloatTensor input = uniform(DIM, 1.0f);

        int[] ids = new int[TOP_K];
        float[] weights = new float[TOP_K];
        CpuOperations.moeRouter(input, uniformRouter, scores, ids, weights, EXPERTS, TOP_K, DIM);

        assertArrayEquals(
                "all scores equal, so the tie rule alone decides: lowest index first",
                new int[] {0, 1, 2},
                ids);
        for (int i = 0; i < TOP_K; i++) {
            assertEquals(
                    "a uniform softmax gives every expert 1/n", 1.0f / EXPERTS, weights[i], 1e-6f);
        }
    }

    /** Selection is by descending weight, and the ids are distinct. */
    @Test
    public void selectionIsOrderedByDescendingWeight() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(11L));
        Qwen2MoEState state = new Qwen2MoEState(config, -1);
        InferenceCore.forwardJavaQwen2MoE(config, weights, state, 5, 0);

        for (int i = 1; i < TOP_K; i++) {
            assertTrue(
                    "routing weights must be non-increasing: "
                            + state.selectedExpertWeights[i - 1]
                            + " then "
                            + state.selectedExpertWeights[i],
                    state.selectedExpertWeights[i - 1] >= state.selectedExpertWeights[i]);
        }
        for (int i = 0; i < TOP_K; i++) {
            for (int j = i + 1; j < TOP_K; j++) {
                assertNotEquals(
                        "selected experts must be distinct",
                        state.selectedExperts[i],
                        state.selectedExperts[j]);
            }
        }
    }

    /**
     * Reversing the expert accumulation order must change the result.
     *
     * <p>This is the evidence that rank order is load-bearing rather than incidental.
     * Floating-point addition is not associative, so a different order is a different sum — and
     * because the accumulation <i>is</i> the residual connection here, there is nothing downstream
     * to wash it out.
     */
    @Test
    public void reversingTheExpertOrderChangesTheResult() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(77L));

        FloatTensor inOrder =
                copyOf(
                        InferenceCore.forwardJavaQwen2MoE(
                                config, weights, new Qwen2MoEState(config, -1), 5, 0),
                        VOCAB);
        FloatTensor reversed =
                copyOf(
                        forwardWithReversedExperts(
                                config, weights, new Qwen2MoEState(config, -1), 5, 0),
                        VOCAB);

        assertDiffers(
                "reversing the expert accumulation order must change the logits; if it does"
                        + " not, the order is untested",
                inOrder,
                reversed,
                VOCAB);
    }

    /**
     * Combining the shared expert before the routed ones must change the result — the same argument
     * as above, applied to the one expert that is not selected by the router.
     */
    @Test
    public void movingTheSharedExpertEarlierChangesTheResult() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(88L));

        FloatTensor sharedLast =
                copyOf(
                        InferenceCore.forwardJavaQwen2MoE(
                                config, weights, new Qwen2MoEState(config, -1), 5, 0),
                        VOCAB);
        FloatTensor sharedFirst =
                copyOf(
                        forwardWithSharedExpertFirst(
                                config, weights, new Qwen2MoEState(config, -1), 5, 0),
                        VOCAB);

        assertDiffers(
                "combining the shared expert first must change the logits",
                sharedLast,
                sharedFirst,
                VOCAB);
    }

    /**
     * The selection arrays are fixed workspace on the state, not allocated per token.
     *
     * <p>Checked by identity: the same array objects must come back after many forward passes. The
     * pre-refactor code allocated two per layer per token.
     */
    @Test
    public void selectionWorkspaceIsNotAllocatedPerToken() {
        Qwen2MoEConfiguration config = config();
        Qwen2MoEStandardWeights weights = syntheticWeights(new Random(9L));
        Qwen2MoEState state = new Qwen2MoEState(config, -1);

        int[] idsBefore = state.selectedExperts;
        float[] weightsBefore = state.selectedExpertWeights;
        assertEquals("sized by the model's top-k", TOP_K, idsBefore.length);
        assertEquals("sized by the model's top-k", TOP_K, weightsBefore.length);

        for (int position = 0; position < 6; position++) {
            InferenceCore.forwardJavaQwen2MoE(config, weights, state, position + 1, position);
        }

        assertTrue(
                "selectedExperts must be reused, not reallocated",
                idsBefore == state.selectedExperts);
        assertTrue(
                "selectedExpertWeights must be reused, not reallocated",
                weightsBefore == state.selectedExpertWeights);
    }

    // deliberately-wrong variants, used only to prove the order tests can fail

    private static FloatTensor forwardWithReversedExperts(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights w,
            Qwen2MoEState state,
            int token,
            int position) {
        return forwardVariant(config, w, state, token, position, true, false);
    }

    private static FloatTensor forwardWithSharedExpertFirst(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights w,
            Qwen2MoEState state,
            int token,
            int position) {
        return forwardVariant(config, w, state, token, position, false, true);
    }

    private static FloatTensor forwardVariant(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights w,
            Qwen2MoEState state,
            int token,
            int position,
            boolean reverseExperts,
            boolean sharedFirst) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (dim * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        AttentionShape shape =
                AttentionShape.uniform(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        (float) Math.sqrt(headSize));

        CpuOperations.embeddingLookup(w.token_embedding_table, token, state.x, dim);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, w.rms_att_weight[l], 0, dim, config.rmsNormEps());
            CpuOperations.matVec(w.wq[l], state.xb, state.q, dim, dim);
            CpuOperations.matVec(w.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(w.wv[l], state.xb, state.v, kvDim, dim);
            CpuOperations.biasAdd(state.q, w.q_bias[l]);
            CpuOperations.biasAdd(state.k, w.k_bias[l]);
            CpuOperations.biasAdd(state.v, w.v_bias[l]);
            CpuOperations.ropeNeox(
                    state.q,
                    state.k,
                    w.freq_cis_real,
                    w.freq_cis_imag,
                    position,
                    config.numberOfHeads(),
                    config.numberOfKeyValueHeads(),
                    headSize);
            CpuOperations.appendKeyValue(
                    state.k,
                    state.v,
                    state.keyCache[l],
                    state.valueCache[l],
                    position,
                    shape.kvDim());
            CpuOperations.attention(
                    state.q,
                    state.keyCache[l],
                    state.valueCache[l],
                    state.att,
                    state.xb,
                    position,
                    shape);
            CpuOperations.matVec(w.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, w.rms_ffn_weight[l], 0, dim, config.rmsNormEps());
            CpuOperations.moeRouter(
                    state.xb,
                    w.routerGate[l],
                    state.routerLogits,
                    state.selectedExperts,
                    state.selectedExpertWeights,
                    EXPERTS,
                    TOP_K,
                    dim);

            if (sharedFirst) {
                sharedExpert(config, w, state, l, dim);
            }
            for (int n = 0; n < TOP_K; n++) {
                int j = reverseExperts ? TOP_K - 1 - n : n;
                CpuOperations.expertFeedForward(
                        state.xb,
                        state.selectedExperts[j],
                        w.gateExps[l],
                        w.upExps[l],
                        w.downExps[l],
                        state.hbE,
                        state.hbE2,
                        state.yTmp,
                        MOE_HIDDEN,
                        dim);
                CpuOperations.weightedAccumulate(
                        state.x, state.yTmp, state.selectedExpertWeights[j], dim);
            }
            if (!sharedFirst) {
                sharedExpert(config, w, state, l, dim);
            }
        }
        CpuOperations.rmsNorm(state.x, state.x, w.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(w.wcls, state.x, state.logits, VOCAB, dim);
        return state.logits;
    }

    private static void sharedExpert(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights w,
            Qwen2MoEState state,
            int l,
            int dim) {
        CpuOperations.matVec(w.sharedGate[l], state.xb, state.hbS, SHARED_HIDDEN, dim);
        CpuOperations.matVec(w.sharedUp[l], state.xb, state.hbS2, SHARED_HIDDEN, dim);
        CpuOperations.swiGLU(state.hbS, state.hbS2);
        CpuOperations.matVec(w.sharedDown[l], state.hbS, state.yTmp, dim, SHARED_HIDDEN);
        float gateScore = w.sharedGateInp[l].dot(0, state.xb, 0, dim);
        CpuOperations.weightedAccumulate(
                state.x, state.yTmp, CpuOperations.logistic(gateScore), dim);
    }

    // the model

    private static Qwen2MoEConfiguration config() {
        return new Qwen2MoEConfiguration(
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
                EXPERTS,
                TOP_K,
                MOE_HIDDEN,
                SHARED_HIDDEN,
                false,
                EPS,
                10000f);
    }

    private static Qwen2MoEStandardWeights syntheticWeights(Random random) {
        int kvDim = DIM * KV_HEADS / HEADS;
        int headSize = DIM / HEADS;
        return new Qwen2MoEStandardWeights(
                tensor(random, VOCAB * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                tensors(random, LAYERS, kvDim * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, kvDim),
                tensors(random, LAYERS, kvDim),
                tensors(random, LAYERS, DIM * DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, DIM * HIDDEN_DIM),
                tensors(random, LAYERS, HIDDEN_DIM * DIM),
                tensors(random, LAYERS, EXPERTS * DIM),
                tensors(random, LAYERS, EXPERTS * MOE_HIDDEN * DIM),
                tensors(random, LAYERS, EXPERTS * MOE_HIDDEN * DIM),
                tensors(random, LAYERS, EXPERTS * DIM * MOE_HIDDEN),
                tensors(random, LAYERS, SHARED_HIDDEN * DIM),
                tensors(random, LAYERS, SHARED_HIDDEN * DIM),
                tensors(random, LAYERS, DIM * SHARED_HIDDEN),
                tensors(random, LAYERS, DIM),
                tensor(random, DIM),
                arrayTensor(random, CONTEXT * (headSize / 2)),
                arrayTensor(random, CONTEXT * (headSize / 2)),
                tensor(random, VOCAB * DIM),
                DataType.F32);
    }

    private static FloatTensor uniform(int size, float value) {
        float[] values = new float[size];
        java.util.Arrays.fill(values, value);
        return new ArrayFloatTensor(values);
    }

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

    private static FloatTensor copyOf(FloatTensor source, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = source.getFloat(i);
        }
        return new ArrayFloatTensor(values);
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

    private static void assertDiffers(String why, FloatTensor a, FloatTensor b, int size) {
        for (int i = 0; i < size; i++) {
            if (Float.floatToRawIntBits(a.getFloat(i)) != Float.floatToRawIntBits(b.getFloat(i))) {
                return;
            }
        }
        fail(why);
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

    private static void referenceMatmulExpert(
            FloatTensor w, int base, FloatTensor in, FloatTensor out, int d0, int d1) {
        Parallel.parallelFor(0, d0, i -> out.setFloat(i, w.dot(base + i * d1, in, 0, d1)));
    }

    static FloatTensor referenceForwardJavaQwen2MoE(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights weights,
            Qwen2MoEState state,
            int token,
            int position) {
        final Qwen2MoEState moeState = (Qwen2MoEState) state;
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

            // MoE FFN pre-normalization
            referenceRmsnorm(
                    state.xb,
                    state.x,
                    weights.rms_ffn_weight[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());

            int numberOfExperts = config.numberOfExperts();
            int topK = config.numberOfExpertsUsed();
            int expertHiddenDim = config.moeHiddenDim();

            // Compute routing probabilities over all experts, then select top-k.
            // Qwen1.5-MoE uses norm_topk_prob=false: each selected expert's routing weight is
            // its probability over all experts without rescaling the top-k weights to sum to one.
            weights.routerGate[l].matmul(state.xb, moeState.routerLogits, numberOfExperts, dim);
            moeState.routerLogits.softmaxInPlace(0, numberOfExperts);

            int[] selectedExperts = new int[topK];
            float[] routingWeights = new float[topK];
            for (int i = 0; i < topK; i++) {
                float best = Float.NEGATIVE_INFINITY;
                int index = -1;
                for (int j = 0; j < numberOfExperts; j++) {
                    if (moeState.routerLogits.getFloat(j) > best) {
                        best = moeState.routerLogits.getFloat(j);
                        index = j;
                    }
                }
                selectedExperts[i] = index;
                routingWeights[i] = best;
                moeState.routerLogits.setFloat(index, Float.NEGATIVE_INFINITY);
            }
            // Compute each selected expert and accumulate its weighted output.
            for (int j = 0; j < topK; j++) {
                int expert = selectedExperts[j];
                int gateUpOffset = expert * expertHiddenDim * dim;
                int downOffset = expert * dim * expertHiddenDim;
                referenceMatmulExpert(
                        weights.gateExps[l],
                        gateUpOffset,
                        state.xb,
                        moeState.hbE,
                        expertHiddenDim,
                        dim);
                referenceMatmulExpert(
                        weights.upExps[l],
                        gateUpOffset,
                        state.xb,
                        moeState.hbE2,
                        expertHiddenDim,
                        dim);
                moeState.hbE.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
                moeState.hbE.multiplyInPlace(moeState.hbE2);
                referenceMatmulExpert(
                        weights.downExps[l],
                        downOffset,
                        moeState.hbE,
                        moeState.yTmp,
                        dim,
                        expertHiddenDim);
                state.x.saxpyInPlace(0, moeState.yTmp, 0, dim, routingWeights[j]);
            }

            // Compute the always-on shared expert.
            int sharedExpertHiddenDim = config.sharedExpertHiddenDim();
            weights.sharedGate[l].matmul(state.xb, moeState.hbS, sharedExpertHiddenDim, dim);
            weights.sharedUp[l].matmul(state.xb, moeState.hbS2, sharedExpertHiddenDim, dim);
            moeState.hbS.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
            moeState.hbS.multiplyInPlace(moeState.hbS2);
            weights.sharedDown[l].matmul(moeState.hbS, moeState.yTmp, dim, sharedExpertHiddenDim);

            // Gate the shared expert output.
            float gateScore = weights.sharedGateInp[l].dot(0, state.xb, 0, dim);
            float sharedExpertWeight = 1f / (1f + (float) Math.exp(-gateScore));
            state.x.saxpyInPlace(0, moeState.yTmp, 0, dim, sharedExpertWeight);
        }

        // final rmsnorm + classifier (same as dense Qwen2)
        referenceRmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Like {@link FloatTensor#matmul}, but reads the weight matrix starting at element offset
     * {@code base} instead of 0 — so it can target ONE expert inside a stacked {@code [nExpert × d0
     * × d1]} tensor. out[d0] = (d0×d1 sub-matrix at base) · in[d1].
     */
}
