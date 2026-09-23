package org.beehive.jllm.backend.cpu;

import org.beehive.jllm.inference.op.AttentionShape;
import org.beehive.jllm.inference.op.CpuOperations;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * The host forward pass for the {@code qwen35} architecture.
 *
 * <p>In its own class rather than another method on {@code InferenceCore} because it is the first
 * family whose layers are not all the same shape: three quarters of them mix with a recurrent
 * delta-net rather than with attention, and the two branches are long enough that reading either
 * one inside a thousand-line switch of families would be worse than reading them here.
 *
 * <p>Everything it does is a call into {@code CpuOperations}. The operations it needs that no other
 * family did — {@code l2Norm}, {@code causalConv1d}, {@code deltaRuleUpdate}, {@code gatedNorm},
 * and RoPE over part of a head — are in the shared vocabulary rather than private here, because the
 * arithmetic is not this family's: every recurrent architecture uses it.
 *
 * <p>Verified against llama.cpp's {@code llama_model_qwen35::graph} and {@code
 * llm_build_delta_net_base::build_delta_net_autoregressive}.
 */
public final class Qwen35Forward {

    private Qwen35Forward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (Qwen35Configuration) model.configuration(),
                (Qwen35StandardWeights) model.weights(),
                (Qwen35State) state,
                token,
                position);
    }

    // @formatter:off
    /**
     * One decode step through the trunk.
     *
     * <p>Every block, of either kind, has the same outer shape — normalize, mix, add back,
     * normalize, feed forward, add back — and only the mixer differs. The MTP blocks past the trunk
     * are not executed here; they are a draft head, driven separately.
     *
     * <p>{@link Qwen35State#hNextn} is written before the vocabulary projection because the MTP
     * block consumes exactly that vector. Written unconditionally: it costs one copy of {@code dim}
     * floats, and a forward pass whose behaviour depends on whether speculation happens to be
     * running is a forward pass that cannot be compared against itself.
     */
    // @formatter:on
    public static FloatTensor forward(
            Qwen35Configuration config,
            Qwen35StandardWeights weights,
            Qwen35State state,
            int token,
            int position) {

        final int dim = config.dim();
        final float eps = config.rmsNormEps();

        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(state.xb, state.x, weights.attnNorm[l], 0, dim, eps);

            if (config.isRecurrentLayer(l)) {
                deltaNetBranch(config, weights, state, l);
            } else {
                attentionBranch(config, weights, state, l, position);
            }
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(state.xb, state.x, weights.ffnNorm[l], 0, dim, eps);
            feedForward(config, weights, state, l);
            CpuOperations.residualAdd(state.x, state.xb2);
        }

        CpuOperations.rmsNorm(state.x, state.x, weights.outputNorm, 0, dim, eps);
        state.x.copyTo(0, state.hNextn, 0, dim);
        CpuOperations.vocabProjection(
                weights.output, state.x, state.logits, config.vocabularySize(), dim);
        return state.logits;
    }

    // @formatter:off
    /**
     * One step of the MTP (NextN) draft head.
     *
     * <p>The block is a complete attention decoder layer with three tensors in front of it. Its
     * input is not the previous layer's output — the trunk has already finished — but the pair
     * <i>(what the trunk was thinking at position p, what token was actually chosen for position
     * p+1)</i>, each normalized by its own norm, concatenated, and projected back down to {@code
     * dim}. From there it predicts the token at {@code p + 2}.
     *
     * <p>It shares the trunk's attention scratch and its own key/value cache at block index {@code
     * numberOfLayers()}, and it shares the trunk's vocabulary projection: this file carries neither
     * {@code nextn.embed_tokens} nor {@code nextn.shared_head_head}, so both fall back to the
     * trunk's, which is what llama.cpp does when they are absent.
     *
     * <p>It does <b>not</b> touch {@link Qwen35State#x} or any recurrent state. A draft is a
     * question about the future, and asking it must not move the trunk.
     *
     * @param drafted the token chosen for {@code position}, whose successor is being predicted
     * @param position where {@code drafted} sits — the MTP block attends at that position
     * @return logits for the token at {@code position + 1}
     */
    // @formatter:on
    public static FloatTensor forwardMtp(
            Qwen35Configuration config,
            Qwen35StandardWeights weights,
            Qwen35State state,
            int drafted,
            int position) {

        final int block = config.numberOfLayers();
        if (config.numberOfNextnLayers() < 1) {
            throw new UnsupportedOperationException(
                    "this qwen35 file carries no MTP block, so it cannot draft");
        }
        final int dim = config.dim();
        final float eps = config.rmsNormEps();

        // [ enorm(embedding of the drafted token) | hnorm(the trunk's hidden state) ]
        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, drafted, state.nextnConcat, dim);
        CpuOperations.rmsNorm(
                state.nextnConcat, state.nextnConcat, weights.nextnENorm[block], 0, dim, eps);
        state.hNextn.copyTo(0, state.nextnConcat, dim, dim);
        CpuOperations.rmsNorm(
                state.nextnConcat, state.nextnConcat, weights.nextnHNorm[block], dim, dim, eps);

        CpuOperations.matVec(
                weights.nextnEhProj[block], state.nextnConcat, state.nextnX, dim, 2 * dim);

        CpuOperations.rmsNorm(state.xb, state.nextnX, weights.attnNorm[block], 0, dim, eps);
        attentionBranch(config, weights, state, block, position);
        CpuOperations.residualAdd(state.nextnX, state.xb2);

        CpuOperations.rmsNorm(state.xb, state.nextnX, weights.ffnNorm[block], 0, dim, eps);
        feedForward(config, weights, state, block);
        CpuOperations.residualAdd(state.nextnX, state.xb2);

        FloatTensor headNorm =
                weights.nextnSharedHeadNorm[block] != null
                        ? weights.nextnSharedHeadNorm[block]
                        : weights.outputNorm;
        CpuOperations.rmsNorm(state.nextnX, state.nextnX, headNorm, 0, dim, eps);
        CpuOperations.vocabProjection(
                weights.output, state.nextnX, state.nextnLogits, config.vocabularySize(), dim);
        return state.nextnLogits;
    }

    /** The dense SwiGLU feed-forward both layer kinds share, from {@code xb} into {@code xb2}. */
    private static void feedForward(
            Qwen35Configuration config, Qwen35StandardWeights weights, Qwen35State state, int l) {
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        CpuOperations.matVec(weights.ffnGate[l], state.xb, state.hb, hidden, dim);
        CpuOperations.matVec(weights.ffnUp[l], state.xb, state.hb2, hidden, dim);
        CpuOperations.swiGLU(state.hb, state.hb2);
        CpuOperations.matVec(weights.ffnDown[l], state.hb, state.xb2, dim, hidden);
    }

    // @formatter:off
    /**
     * A full-attention layer, from the normalized {@code xb} into {@code xb2}.
     *
     * <p>Three things separate it from Qwen3's attention, and none of them is a new operation.
     *
     * <ul>
     *   <li><b>The query projection carries an output gate.</b> {@code attn_q} is twice as wide as
     *       a query projection: per head, a query slice then a gate slice. They are de-interleaved
     *       here rather than addressed in place, because everything downstream — the per-head norm,
     *       the rotation, the attention itself — wants contiguous heads, and one copy of 12288
     *       floats is cheaper than a stride parameter on four operations.
     *   <li><b>The head is 256 wide but only 64 of it rotates.</b> {@code rope.dimension_count} is
     *       stated independently of the head width; the remaining 192 elements pass through.
     *   <li><b>The result is gated before the output projection</b>, by the logistic of the gate
     *       slice. This is the one place the branch is not a call into a named operation: it is an
     *       elementwise product of two activations, which {@code SWIGLU} would misname (that is a
     *       SiLU, and this is a plain logistic).
     * </ul>
     */
    // @formatter:on
    private static void attentionBranch(
            Qwen35Configuration config,
            Qwen35StandardWeights weights,
            Qwen35State state,
            int l,
            int position) {

        final int dim = config.dim();
        final int headDim = config.numberOfHeadsKey();
        final int heads = config.numberOfHeads();
        final int kvHeads = config.numberOfKeyValueHeads();
        final int kvDim = config.kvDim();
        final int attnDim = config.attentionOutputInputDim();
        final float eps = config.rmsNormEps();

        CpuOperations.matVec(weights.wq[l], state.xb, state.q, config.queryGateDim(), dim);
        CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
        CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

        // Per head: [query | gate], adjacent. Element stride between heads is 2 * headDim.
        for (int h = 0; h < heads; h++) {
            int fused = h * 2 * headDim;
            state.q.copyTo(fused, state.attnQ, h * headDim, headDim);
            state.q.copyTo(fused + headDim, state.attnGate, h * headDim, headDim);
        }

        for (int h = 0; h < heads; h++) {
            CpuOperations.rmsNorm(
                    state.attnQ, state.attnQ, weights.attnQNorm[l], h * headDim, headDim, eps);
        }
        for (int h = 0; h < kvHeads; h++) {
            CpuOperations.rmsNorm(
                    state.k, state.k, weights.attnKNorm[l], h * headDim, headDim, eps);
        }

        CpuOperations.ropeNeoxPartial(
                state.attnQ,
                heads,
                headDim,
                config.ropeDimensionCount(),
                position,
                weights.freqCisReal,
                weights.freqCisImag);
        CpuOperations.ropeNeoxPartial(
                state.k,
                kvHeads,
                headDim,
                config.ropeDimensionCount(),
                position,
                weights.freqCisReal,
                weights.freqCisImag);

        CpuOperations.appendKeyValue(
                state.k, state.v, state.keyCache[l], state.valueCache[l], position, kvDim);

        AttentionShape shape =
                new AttentionShape(
                        heads,
                        config.kvMul(),
                        headDim,
                        headDim,
                        headDim,
                        config.numberOfHeadsValue(),
                        kvDim,
                        config.contextLength(),
                        AttentionShape.ScoreScaling.DIVIDE,
                        (float) Math.sqrt(headDim),
                        0);
        CpuOperations.attention(
                state.attnQ,
                state.keyCache[l],
                state.valueCache[l],
                state.att,
                state.xb,
                position,
                shape);

        for (int i = 0; i < attnDim; i++) {
            state.xb.setFloat(
                    i, state.xb.getFloat(i) * CpuOperations.logistic(state.attnGate.getFloat(i)));
        }

        CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, attnDim);
    }

    // @formatter:off
    /**
     * A Gated Delta Net layer, from the normalized {@code xb} into {@code xb2}.
     *
     * <p>Linear attention: instead of scoring a query against every retained key, the layer keeps
     * one {@code headValueDim × headValueDim} matrix per value head into which the whole sequence
     * has been written, and each step decays it, corrects it towards the new value, and reads it
     * back. Cost per token is constant in the sequence length, which is why three quarters of the
     * stack can be built this way.
     *
     * <p>Order matters and follows llama.cpp: the fused projection is convolved <b>before</b> it is
     * split, so the convolution mixes across time within each channel while the split is what
     * separates queries from keys from values. Normalizing the queries and keys after the
     * convolution rather than before is likewise not interchangeable.
     */
    // @formatter:on
    private static void deltaNetBranch(
            Qwen35Configuration config, Qwen35StandardWeights weights, Qwen35State state, int l) {

        final int dim = config.dim();
        final int convDim = config.deltaNetConvDim();
        final int keyDim = config.deltaNetKeyDim();
        final int valueDim = config.deltaNetValueDim();
        final int keyHeads = config.numberOfKeyHeads();
        final int valueHeads = config.numberOfValueHeads();
        final int headK = config.headKeyDim();
        final int headV = config.headValueDim();
        final float eps = config.rmsNormEps();

        CpuOperations.matVec(weights.ssmQkv[l], state.xb, state.ssmQkv, convDim, dim);
        CpuOperations.matVec(weights.ssmGate[l], state.xb, state.ssmZ, valueDim, dim);
        CpuOperations.matVec(weights.ssmBeta[l], state.xb, state.ssmBeta, valueHeads, dim);
        CpuOperations.matVec(weights.ssmAlpha[l], state.xb, state.ssmAlpha, valueHeads, dim);

        // beta: how much of the correction to write. alpha: how much of the state survives.
        // ssm_a is already -exp(A_log), so the product is a log decay and its exponential is in
        // (0, 1) — the state is forgotten, never amplified.
        for (int h = 0; h < valueHeads; h++) {
            state.ssmBeta.setFloat(h, CpuOperations.logistic(state.ssmBeta.getFloat(h)));
            float a =
                    CpuOperations.softplus(
                            state.ssmAlpha.getFloat(h) + weights.ssmDtBias[l].getFloat(h));
            state.ssmAlpha.setFloat(h, (float) Math.exp(weights.ssmA[l].getFloat(h) * a));
        }

        CpuOperations.causalConv1d(
                state.ssmQkv,
                weights.ssmConv1d[l],
                state.convState[l],
                state.ssmConvOut,
                convDim,
                config.ssmConvKernel());
        state.ssmConvOut.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));

        state.ssmConvOut.copyTo(0, state.ssmQ, 0, keyDim);
        state.ssmConvOut.copyTo(keyDim, state.ssmK, 0, keyDim);
        state.ssmConvOut.copyTo(2 * keyDim, state.ssmV, 0, valueDim);

        for (int h = 0; h < keyHeads; h++) {
            CpuOperations.l2Norm(state.ssmQ, h * headK, headK, eps);
            CpuOperations.l2Norm(state.ssmK, h * headK, headK, eps);
        }
        CpuOperations.scale(state.ssmQ, (float) (1.0 / Math.sqrt(headK)));

        CpuOperations.deltaRuleUpdate(
                state.ssmQ,
                state.ssmK,
                state.ssmV,
                state.ssmAlpha,
                state.ssmBeta,
                state.deltaState[l],
                state.ssmOut,
                valueHeads,
                keyHeads,
                headV);

        CpuOperations.gatedNorm(
                state.ssmOut, state.ssmZ, weights.ssmNorm[l], valueHeads, headV, eps);
        CpuOperations.matVec(weights.ssmOut[l], state.ssmOut, state.xb2, dim, valueDim);
    }
}
