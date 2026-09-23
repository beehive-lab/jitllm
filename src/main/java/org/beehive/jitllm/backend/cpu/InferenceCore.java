package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.AttentionShape;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.*;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Low-level operations for model inference.
 *
 * <p>This class provides core computational operations such as RMS normalization and forward passes
 * through model layers. It supports both CPU and GPU implementations.
 */
public final class InferenceCore {

    private InferenceCore() {
        // prevent instantiation
    }

    /** Root-mean-square normalization. */
    public static void rmsnorm(
            FloatTensor out,
            FloatTensor x,
            FloatTensor weight,
            int offset,
            int size,
            float rmsNormEps) {
        CpuOperations.rmsNorm(out, x, weight, offset, size, rmsNormEps);
    }

    public static FloatTensor forwardJava(Model model, State state, int token, int position) {
        return forwardJava(
                model.configuration(), (StandardWeights) model.weights(), state, token, position);
    }

    // @formatter:off
    /**
     * Llama's host forward pass, as a sequence of named operations.
     *
     * <p>What changed here is the <b>naming</b>, not the arithmetic. Every step below was already
     * in this method; each is now a call to the {@link CpuOperations} implementation of the {@code
     * program.op} operation it always was, in the same order, over the same buffers. The kernel
     * bodies moved verbatim — {@code LlamaCpuOperationEquivalenceTest} holds a copy of the
     * pre-refactor method and asserts bit-identical logits.
     *
     * <p>Read as a program, one layer is:
     *
     * <pre>
     *   RmsNorm  →  MatVec ×3 (Q, K, V)  →  RoPE  →  Attention  →  MatVec (O)  →  ResidualAdd
     *            →  RmsNorm  →  MatVec ×2 (gate, up)  →  SwiGLU  →  MatVec (down)  →  ResidualAdd
     * </pre>
     *
     * <p>Takes the configuration and weights rather than the {@link Model} so that the forward pass
     * can be exercised without building one — the equivalence test needs a synthetic model, not a
     * GGUF file. The {@code Model}-taking overload above is unchanged for callers.
     */
    // @formatter:on
    public static FloatTensor forwardJava(
            Configuration config, StandardWeights weights, State state, int token, int position) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul =
                config.numberOfHeads()
                        / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in
        // multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, dim, dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

            CpuOperations.rope(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
                    position,
                    dim,
                    kvDim,
                    headSize);

            CpuOperations.appendKeyValue(
                    state.k, state.v, state.keyCache[l], state.valueCache[l], position, kvDim);
            CpuOperations.attention(
                    state.q,
                    state.keyCache[l],
                    state.valueCache[l],
                    state.att,
                    state.xb,
                    position,
                    AttentionShape.uniform(
                            config.numberOfHeads(),
                            kvMul,
                            headSize,
                            kvDim,
                            config.contextLength(),
                            sqrtHeadSize));

            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            CpuOperations.matVec(weights.w1[l], state.xb, state.hb, config.hiddenDim(), dim);
            CpuOperations.matVec(weights.w3[l], state.xb, state.hb2, config.hiddenDim(), dim);
            CpuOperations.swiGLU(state.hb, state.hb2);
            CpuOperations.matVec(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim());

            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Forward pass for Devstral 2 models where head_dim != dim/num_heads. Q projection outputs qDim
     * (num_heads * head_dim) instead of dim.
     */
    public static FloatTensor forwardJavaDevstral(
            Model model, State state, int token, int position) {
        return forwardJavaDevstral(
                (DevstralConfiguration) model.configuration(),
                (StandardWeights) model.weights(),
                state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Devstral's host forward pass, as a sequence of named operations.
     *
     * <p>Devstral's one structural difference from Llama is that its head dimension is
     * <b>independent of {@code dim / heads}</b>: {@code qDim = heads * headDim} need not equal
     * {@code dim}. That shows up in exactly three places and none of them is a new operation — the
     * query projection is {@code qDim} wide, the rotation covers {@code qDim} rather than {@code
     * dim}, and the output projection reads {@code qDim} and writes {@code dim}. The head geometry
     * itself is still uniform, so {@link AttentionShape#uniform} describes it.
     *
     * <p>Bodies moved verbatim; {@code DevstralCpuOperationEquivalenceTest} holds the pre-refactor
     * method and asserts bit-identical output.
     */
    // @formatter:on
    public static FloatTensor forwardJavaDevstral(
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
        AttentionShape shape =
                AttentionShape.uniform(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        sqrtHeadSize);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, qDim, dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

            // The rotation covers qDim, not dim: Devstral's head dimension is independent.
            CpuOperations.rope(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
                    position,
                    qDim,
                    kvDim,
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

            // O projection: input qDim, output dim
            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, qDim);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.w1[l], state.xb, state.hb, config.hiddenDim(), dim);
            CpuOperations.matVec(weights.w3[l], state.xb, state.hb2, config.hiddenDim(), dim);
            CpuOperations.swiGLU(state.hb, state.hb2);
            CpuOperations.matVec(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim());

            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen2MoE(
            Model model, State state, int token, int position) {
        return forwardJavaQwen2MoE(
                (Qwen2MoEConfiguration) model.configuration(),
                (Qwen2MoEStandardWeights) model.weights(),
                (Qwen2MoEState) state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Qwen2-MoE's host forward pass, as a sequence of named operations.
     *
     * <pre>
     *   RmsNorm
     *   MoeRouter                                  → topK expert ids + weights
     *   for each selected expert, in rank order:
     *       ExpertFeedForward                      → that expert's contribution
     *       WeightedAccumulate(NONE)               → into the residual stream
     *   MatVec ×2 + SwiGLU + MatVec                → the always-on shared expert
     *   MatVec(rows = 1)                           → its scalar gate
     *   WeightedAccumulate(LOGISTIC)               → into the residual stream, last
     * </pre>
     *
     * <p><b>There is no residual add.</b> Each accumulate goes straight into {@code state.x}, so
     * the weighted sum <i>is</i> the residual connection — which is why the order above is contract
     * and not convention.
     *
     * <p>Ordered expert components rather than one component consuming all of them: that is what
     * the previous implementation did, and it keeps the accumulation order explicit.
     */
    // @formatter:on
    public static FloatTensor forwardJavaQwen2MoE(
            Qwen2MoEConfiguration config,
            Qwen2MoEStandardWeights weights,
            Qwen2MoEState state,
            int token,
            int position) {
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        AttentionShape shape =
                AttentionShape.uniform(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        sqrtHeadSize);

        int numberOfExperts = config.numberOfExperts();
        int topK = config.numberOfExpertsUsed();
        int expertHiddenDim = config.moeHiddenDim();
        int sharedExpertHiddenDim = config.sharedExpertHiddenDim();

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, dim, dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

            CpuOperations.biasAdd(state.q, weights.q_bias[l]);
            CpuOperations.biasAdd(state.k, weights.k_bias[l]);
            CpuOperations.biasAdd(state.v, weights.v_bias[l]);

            CpuOperations.ropeNeox(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
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

            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.residualAdd(state.x, state.xb2);

            // MoE feed-forward. The shared input for the router, every routed expert and the
            // shared expert alike.
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.moeRouter(
                    state.xb,
                    weights.routerGate[l],
                    state.routerLogits,
                    state.selectedExperts,
                    state.selectedExpertWeights,
                    numberOfExperts,
                    topK,
                    dim);

            for (int j = 0; j < topK; j++) {
                CpuOperations.expertFeedForward(
                        state.xb,
                        state.selectedExperts[j],
                        weights.gateExps[l],
                        weights.upExps[l],
                        weights.downExps[l],
                        state.hbE,
                        state.hbE2,
                        state.yTmp,
                        expertHiddenDim,
                        dim);
                CpuOperations.weightedAccumulate(
                        state.x, state.yTmp, state.selectedExpertWeights[j], dim);
            }

            // The always-on shared expert, combined last.
            CpuOperations.matVec(
                    weights.sharedGate[l], state.xb, state.hbS, sharedExpertHiddenDim, dim);
            CpuOperations.matVec(
                    weights.sharedUp[l], state.xb, state.hbS2, sharedExpertHiddenDim, dim);
            CpuOperations.swiGLU(state.hbS, state.hbS2);
            CpuOperations.matVec(
                    weights.sharedDown[l], state.hbS, state.yTmp, dim, sharedExpertHiddenDim);

            float gateScore = weights.sharedGateInp[l].dot(0, state.xb, 0, dim);
            CpuOperations.weightedAccumulate(
                    state.x, state.yTmp, CpuOperations.logistic(gateScore), dim);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen2(Model model, State state, int token, int position) {
        return forwardJavaQwen2(
                (Qwen2Configuration) model.configuration(),
                (Qwen2StandardWeights) model.weights(),
                state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Qwen2's host forward pass, as a sequence of named operations.
     *
     * <p>The same shape as Llama's with two differences, and both are operations rather than
     * special cases:
     *
     * <ul>
     *   <li>a {@code BiasAdd} after each of the three projections — Qwen2 ships {@code q_bias},
     *       {@code k_bias} and {@code v_bias} where Llama ships none;
     *   <li>{@code RoPE} in the {@code NEOX_HALF} layout, pairing a component with the one half a
     *       head away rather than with its neighbour.
     * </ul>
     *
     * <p>Bodies moved verbatim; {@code Qwen2CpuOperationEquivalenceTest} holds the pre-refactor
     * method and asserts bit-identical output.
     */
    // @formatter:on
    public static FloatTensor forwardJavaQwen2(
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
        AttentionShape shape =
                AttentionShape.uniform(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        sqrtHeadSize);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, dim, dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

            CpuOperations.biasAdd(state.q, weights.q_bias[l]);
            CpuOperations.biasAdd(state.k, weights.k_bias[l]);
            CpuOperations.biasAdd(state.v, weights.v_bias[l]);

            CpuOperations.ropeNeox(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
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

            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.w1[l], state.xb, state.hb, config.hiddenDim(), dim);
            CpuOperations.matVec(weights.w3[l], state.xb, state.hb2, config.hiddenDim(), dim);
            CpuOperations.swiGLU(state.hb, state.hb2);
            CpuOperations.matVec(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim());

            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen3(Model model, State state, int token, int position) {
        return forwardJavaQwen3(
                (Qwen3Configuration) model.configuration(),
                (Qwen3StandardWeights) model.weights(),
                state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Qwen3's host forward pass, as a sequence of named operations.
     *
     * <p>Two things distinguish it from Qwen2, and neither needs a new operation:
     *
     * <ul>
     *   <li><b>Per-head query and key normalization.</b> {@code RmsNorm} applied at a head's
     *       offset, once per head — the same operation with an offset and a length, which is why
     *       {@code CpuOperations.rmsNorm} takes both.
     *   <li><b>Three head dimensions, not one.</b> {@code attention.key_length} and {@code
     *       attention.value_length} are separate metadata and need not equal {@code dim / heads}.
     *       That is what {@link AttentionShape} exists to carry; collapsing them to a head size
     *       works on Llama and mis-addresses this.
     * </ul>
     *
     * <p>Bodies moved verbatim; {@code Qwen3CpuOperationEquivalenceTest} holds the pre-refactor
     * method and asserts bit-identical output.
     */
    // @formatter:on
    public static FloatTensor forwardJavaQwen3(
            Qwen3Configuration config,
            Qwen3StandardWeights weights,
            State state,
            int token,
            int position) {
        int dim = config.dim();
        int nHeadKv = config.numberOfKeyValueHeads(); // n_head_kv = numberOfKeyValueHeads
        int nEmbdHeadK = config.numberOfHeadsKey(); // n_embd_head_k; %s.attention.key_length
        int nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v; %s.attention.value_length
        int nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        int nEmbdHead = nEmbdHeadV;
        int nEmbdGqa = nEmbdVGqa;
        int gqa =
                config.numberOfHeads()
                        / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in
        // multiquery
        float sqrtHeadSize = (float) Math.sqrt(nEmbdHead);
        AttentionShape shape =
                new AttentionShape(
                        config.numberOfHeads(),
                        gqa,
                        nEmbdHead,
                        nEmbdHead,
                        nEmbdHeadK,
                        nEmbdHeadV,
                        nEmbdGqa,
                        config.contextLength(),
                        AttentionShape.ScoreScaling.DIVIDE,
                        sqrtHeadSize,
                        0);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(
                    weights.wq[l], state.xb, state.q, nEmbdHeadK * config.numberOfHeads(), dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, nEmbdGqa, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, nEmbdGqa, dim);

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            for (int i = 0; i < config.numberOfHeads(); i++) {
                CpuOperations.rmsNorm(
                        state.q,
                        state.q,
                        weights.attnQNorm[l],
                        i * nEmbdHead,
                        nEmbdHead,
                        config.rmsNormEps());
            }
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                CpuOperations.rmsNorm(
                        state.k,
                        state.k,
                        weights.attnKNorm[l],
                        i * nEmbdHead,
                        nEmbdHead,
                        config.rmsNormEps());
            }

            CpuOperations.ropeNeox(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
                    position,
                    config.numberOfHeads(),
                    config.numberOfKeyValueHeads(),
                    nEmbdHead);

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

            CpuOperations.matVec(
                    weights.wo[l], state.xb, state.xb2, dim, nEmbdHeadK * config.numberOfHeads());
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.w1[l], state.xb, state.hb, config.hiddenDim(), dim);
            CpuOperations.matVec(weights.w3[l], state.xb, state.hb2, config.hiddenDim(), dim);
            CpuOperations.swiGLU(state.hb, state.hb2);
            CpuOperations.matVec(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim());

            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * RMS-normalizes without applying a learned scale (Gemma4 normalizes V with a plain,
     * weight-less RMSNorm).
     */
    private static void rmsnormNoWeight(
            FloatTensor out, FloatTensor x, int offset, int size, float rmsNormEps) {
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(offset, size, (value, index) -> finalss * x.getFloat(index));
    }

    public static FloatTensor forwardJavaGemma4(Model model, State state, int token, int position) {
        return forwardJavaGemma4(
                (Gemma4Configuration) model.configuration(),
                (Gemma4StandardWeights) model.weights(),
                (Gemma4State) state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Gemma4's host forward pass, as a sequence of named operations.
     *
     * <ul>
     *   <li><b>Layers that reuse an earlier layer's key/value store.</b> A reuse layer runs a
     *       <b>shorter component sequence</b> — no key/value projection, no key/value norms, no key
     *       rotation, no {@code KvAppend} — and its attention reads the layer that did. No no-op
     *       components are inserted.
     *   <li><b>Sliding-window layers</b>, alternating with full-attention ones, expressed as the
     *   <li><b>Per-layer embeddings</b>, a second embedding stream blended in and then contributed
     *       into the residual stream by every layer through a gated projection.
     *   <li><b>A post-norm residual topology</b>: the branch output is normalized before it is
     *       added back, where Llama normalizes the branch input. Component order only — no
     *       vocabulary change.
     * </ul>
     *
     * <p>Its attention scale is {@code 1.0} and needs no new mode: multiplying by {@code 1.0f} is
     * bit-exact identity, which is what {@code ScoreScaling.MULTIPLY} already does.
     */
    // @formatter:on
    public static FloatTensor forwardJavaGemma4(
            Gemma4Configuration config,
            Gemma4StandardWeights weights,
            Gemma4State gs,
            int token,
            int position) {
        final State state = gs;
        final int dim = config.dim();
        final int nHead = config.numberOfHeads();
        final int nHeadKv = config.numberOfKeyValueHeads();
        final int kvMul = config.kvMul();
        final int nLayers = config.numberOfLayers();
        final int nEmbdPerLayer = config.embeddingLengthPerLayer();
        final int perLayerTotal = nLayers * nEmbdPerLayer;
        final float attentionScale =
                1.0f; // Gemma4 attention uses scaling = 1.0 (no 1/sqrt(headDim))

        // 1. token embedding, scaled by sqrt(dim)
        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);
        CpuOperations.scale(state.x, (float) Math.sqrt(dim));

        // 2. per-layer embeddings (PLE)
        CpuOperations.embeddingLookupLongIndexed(
                weights.perLayerTokenEmbd, token, perLayerTotal, gs.perLayerInputs, 0);
        CpuOperations.scale(gs.perLayerInputs, (float) Math.sqrt(nEmbdPerLayer));

        CpuOperations.matVec(
                weights.perLayerModelProj, state.x, gs.perLayerProjScratch, perLayerTotal, dim);
        CpuOperations.scale(gs.perLayerProjScratch, (float) (1.0 / Math.sqrt(dim)));
        for (int l = 0; l < nLayers; l++) {
            CpuOperations.rmsNorm(
                    gs.perLayerProjScratch,
                    gs.perLayerProjScratch,
                    weights.perLayerProjNorm,
                    l * nEmbdPerLayer,
                    nEmbdPerLayer,
                    config.rmsNormEps());
        }
        final float perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));
        for (int i = 0; i < perLayerTotal; i++) {
            float v =
                    (gs.perLayerProjScratch.getFloat(i) + gs.perLayerInputs.getFloat(i))
                            * perLayerInputScale;
            gs.perLayerInputs.setFloat(i, v);
        }

        // 3. transformer layers
        for (int l = 0; l < nLayers; l++) {
            final int headDim = config.headDim(l);
            final boolean isSwa = config.isSwa(l);
            final int qDim = nHead * headDim;
            final int kvDim = nHeadKv * headDim;

            FloatTensor freqCisReal = isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull;
            FloatTensor freqCisImag = isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull;

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.attnNorm[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, qDim, dim);
            for (int h = 0; h < nHead; h++) {
                CpuOperations.rmsNorm(
                        state.q,
                        state.q,
                        weights.attnQNorm[l],
                        h * headDim,
                        headDim,
                        config.rmsNormEps());
            }
            CpuOperations.ropeNeoxSingle(
                    state.q, nHead, headDim, position, freqCisReal, freqCisImag);

            // Either this layer produces key/value data, or it borrows an earlier layer's.
            // A borrowing layer runs none of the components in this branch, and appends nothing.
            final int kvSrcLayer;
            if (config.hasOwnKv(l)) {
                CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
                CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);
                for (int h = 0; h < nHeadKv; h++) {
                    CpuOperations.rmsNorm(
                            state.k,
                            state.k,
                            weights.attnKNorm[l],
                            h * headDim,
                            headDim,
                            config.rmsNormEps());
                    CpuOperations.rmsNormUnweighted(
                            state.v, state.v, h * headDim, headDim, config.rmsNormEps());
                }
                CpuOperations.ropeNeoxSingle(
                        state.k, nHeadKv, headDim, position, freqCisReal, freqCisImag);

                CpuOperations.appendKeyValue(
                        state.k, state.v, state.keyCache[l], state.valueCache[l], position, kvDim);
                kvSrcLayer = l;
            } else {
                kvSrcLayer = config.kvReuseLayer(l);
            }

            AttentionShape shape =
                    new AttentionShape(
                            nHead,
                            kvMul,
                            headDim,
                            headDim,
                            headDim,
                            headDim,
                            kvDim,
                            config.contextLength(),
                            AttentionShape.ScoreScaling.MULTIPLY,
                            attentionScale,
                            isSwa ? config.slidingWindowSize() : 0);
            CpuOperations.attention(
                    state.q,
                    state.keyCache[kvSrcLayer],
                    state.valueCache[kvSrcLayer],
                    state.att,
                    state.xb,
                    position,
                    shape);

            // wo projection, post-attention norm, residual
            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, qDim);
            CpuOperations.rmsNorm(
                    state.xb2, state.xb2, weights.attnPostNorm[l], 0, dim, config.rmsNormEps());
            CpuOperations.residualAdd(state.x, state.xb2);

            // FFN (GeGLU), post-FFN norm, residual
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.ffnNorm[l], 0, dim, config.rmsNormEps());
            CpuOperations.matVec(
                    weights.ffnGate[l], state.xb, state.hb, config.feedForwardLength(l), dim);
            CpuOperations.matVec(
                    weights.ffnUp[l], state.xb, state.hb2, config.feedForwardLength(l), dim);
            CpuOperations.geGLU(state.hb, state.hb2);
            CpuOperations.matVec(
                    weights.ffnDown[l], state.hb, state.xb2, dim, config.feedForwardLength(l));
            CpuOperations.rmsNorm(
                    state.xb2, state.xb2, weights.ffnPostNorm[l], 0, dim, config.rmsNormEps());
            CpuOperations.residualAdd(state.x, state.xb2);

            // per-layer embedding contribution
            CpuOperations.matVec(
                    weights.perLayerInpGate[l], state.x, gs.perLayerGate, nEmbdPerLayer, dim);
            gs.perLayerGate.mapInPlace(CpuOperations::gelu);
            int peOffset = l * nEmbdPerLayer;
            for (int j = 0; j < nEmbdPerLayer; j++) {
                gs.perLayerGate.setFloat(
                        j, gs.perLayerGate.getFloat(j) * gs.perLayerInputs.getFloat(peOffset + j));
            }
            CpuOperations.matVec(
                    weights.perLayerProj[l], gs.perLayerGate, gs.perLayerOut, dim, nEmbdPerLayer);
            CpuOperations.rmsNorm(
                    gs.perLayerOut,
                    gs.perLayerOut,
                    weights.perLayerPostNorm[l],
                    0,
                    dim,
                    config.rmsNormEps());
            CpuOperations.residualAdd(state.x, gs.perLayerOut);

            // optional learned per-layer output scale — absent when the model has no such weight
            FloatTensor outScale = weights.layerOutputScale[l];
            if (outScale != null) {
                CpuOperations.scale(state.x, outScale.getFloat(0));
            }
        }

        CpuOperations.rmsNorm(state.x, state.x, weights.outputNorm, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.outputWeight, state.x, state.logits, config.vocabularySize(), dim);

        // A model without soft-capping omits the operation; it is never run with cap == 0.
        final float softcap = config.finalLogitSoftcapping();
        if (softcap != 0.0f) {
            CpuOperations.logitSoftCap(state.logits, softcap);
        }

        return state.logits;
    }

    public static FloatTensor forwardJavaPhi3(
            Model model, Phi3State state, int token, int position) {
        return forwardJavaPhi3(
                (Phi3Configuration) model.configuration(),
                (Phi3StandardWeights) model.weights(),
                state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Phi3's host forward pass, as a sequence of named operations.
     *
     * <p>Phi3 fuses two projections that other families keep separate, and both fusions are
     * <b>addressing</b> rather than new arithmetic:
     *
     * <ul>
     *   <li><b>QKV in one matrix.</b> One {@code MatVec} into a wide buffer, then three slices out
     *       of it. The slices are copies with no arithmetic in them, so they are not an operation —
     *       the same classification the key/value append gets.
     *   <li><b>Gate and up in one matrix.</b> One {@code MatVec} of width {@code 2 * hiddenDim},
     *       then two chunk copies, then {@code SwiGLU}. The result lands in the <i>up</i> half
     *       rather than the gate half, which is why {@link CpuOperations#swiGLUIntoUp} exists:
     *       float multiplication is commutative and exactly rounded, so the values match either
     *       way, but which buffer holds them is what keeps the family bit-identical without an
     *       extra copy.
     * </ul>
     *
     * <p>Its rotation turns out to be the <b>same {@code NEOX_HALF} layout as the Qwen
     * families</b>, written differently: stepping {@code i} over the whole dimension with {@code
     * base = i - i % headSize} visits exactly the head offsets, and {@code i < kvDim} selects
     * exactly the first {@code numberOfKeyValueHeads} heads. So it reuses {@link
     * CpuOperations#ropeNeox} rather than gaining a third layout — which is the kind of thing
     * naming the layouts is for.
     */
    // @formatter:on
    public static FloatTensor forwardJavaPhi3(
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
        AttentionShape shape =
                AttentionShape.uniform(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        sqrtHeadSize);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);

        // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        final int opSize = dim + 2 * (config.numberOfKeyValueHeads() * headSize);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wqkv[l], state.xb, state.qkv, opSize, dim);
            CpuOperations.splitFusedQkv(
                    state.qkv,
                    state.q,
                    state.k,
                    state.v,
                    dim,
                    config.numberOfKeyValueHeads() * headSize);

            CpuOperations.ropeNeox(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
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

            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(
                    weights.wGateUp[l], state.xb, state.hb, 2 * config.hiddenDim(), dim);
            copyChunk(state.hb, state.hbG, 2 * config.hiddenDim(), config.hiddenDim(), 2, 0);
            copyChunk(state.hb, state.hbU, 2 * config.hiddenDim(), config.hiddenDim(), 2, 1);
            CpuOperations.swiGLUIntoUp(state.hbG, state.hbU);

            CpuOperations.matVec(weights.wDown[l], state.hbU, state.xb, dim, config.hiddenDim());
            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Forward pass for Granite models with µP scaling factors applied.
     *
     * <p>Granite uses the same transformer architecture as Llama but with maximal update
     * parameterization (µP) scaling factors applied at specific points:
     *
     * <ul>
     *   <li>Embedding scaling: multiply embeddings after lookup
     *   <li>Attention scaling: use custom multiplier instead of 1/sqrt(headDim)
     *   <li>Residual scaling: multiply residual connections
     *   <li>Logit scaling: divide logits by the scaling factor
     * </ul>
     */
    public static FloatTensor forwardGranite(Model model, State state, int token, int position) {
        return forwardGranite(
                (GraniteConfiguration) model.configuration(),
                (StandardWeights) model.weights(),
                state,
                token,
                position);
    }

    // @formatter:off
    /**
     * Granite's host forward pass, as a sequence of named operations.
     *
     * <p>Llama's sequence with maximal-update-parameterization (µP) factors at four points, each
     * one a {@code Scale} operation rather than a parameter smuggled into a neighbour: after the
     * embedding lookup, on each of the two residual branches before they are added back, and on the
     * logits.
     *
     * <p>A fifth µP factor is <b>not</b> a {@code Scale}: Granite's attention multiplier
     * <i>replaces</i> the conventional {@code 1/sqrt(headDim)} division rather than following it.
     * That is why {@link AttentionShape} carries a scaling mode — expressing the multiplier as a
     * division by its reciprocal would round, and the outputs would stop matching what this family
     * computes today.
     */
    // @formatter:on
    public static FloatTensor forwardGranite(
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
        AttentionShape shape =
                AttentionShape.uniformScaled(
                        config.numberOfHeads(),
                        kvMul,
                        headSize,
                        kvDim,
                        config.contextLength(),
                        attentionScale);

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, state.x, dim);
        CpuOperations.scale(state.x, embeddingScale);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.wq[l], state.xb, state.q, dim, dim);
            CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
            CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

            CpuOperations.rope(
                    state.q,
                    state.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
                    position,
                    dim,
                    kvDim,
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

            CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
            CpuOperations.scale(state.xb2, residualScale);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            CpuOperations.matVec(weights.w1[l], state.xb, state.hb, config.hiddenDim(), dim);
            CpuOperations.matVec(weights.w3[l], state.xb, state.hb2, config.hiddenDim(), dim);
            CpuOperations.swiGLU(state.hb, state.hb2);
            CpuOperations.matVec(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim());

            CpuOperations.scale(state.xb, residualScale);
            CpuOperations.residualAdd(state.x, state.xb);
        }

        CpuOperations.rmsNorm(
                state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.wcls, state.x, state.logits, config.vocabularySize(), dim);
        CpuOperations.scale(state.logits, logitScale);

        return state.logits;
    }

    static void copyChunk(
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
}
