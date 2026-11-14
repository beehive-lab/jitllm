package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.op.AttentionShape;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.Lfm2StandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass for the {@code lfm2moe} architecture.
 *
 * <p>Every block has the same outer shape — normalize, mix, add back, normalize, feed forward, add
 * back — and differs in two independent choices: the mixer is attention or a gated short
 * convolution, and the feed-forward is dense or a routed mixture of experts.
 *
 * <p>Everything here is a call into {@code CpuOperations}. The short convolution needs no operation
 * of its own: it is two elementwise products around the depthwise {@code causalConv1d} Qwen 3.5's
 * recurrent layers already use.
 *
 * <p>Verified against llama.cpp's {@code llama_model_lfm2::graph} and {@code build_moe_ffn}.
 */
public final class Lfm2Forward {

    private Lfm2Forward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (Lfm2Configuration) model.configuration(),
                (Lfm2StandardWeights) model.weights(),
                (Lfm2State) state,
                token,
                position);
    }

    public static FloatTensor forward(
            Lfm2Configuration config,
            Lfm2StandardWeights weights,
            Lfm2State state,
            int token,
            int position) {

        final int dim = config.dim();
        final float eps = config.rmsNormEps();

        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(state.xb, state.x, weights.attnNorm[l], 0, dim, eps);
            if (config.isAttentionLayer(l)) {
                attentionBranch(config, weights, state, l, position);
            } else {
                shortConvBranch(config, weights, state, l);
            }
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(state.xb, state.x, weights.ffnNorm[l], 0, dim, eps);
            if (config.isMoeLayer(l)) {
                mixtureOfExperts(config, weights, state, l);
            } else {
                denseFeedForward(config, weights, state, l);
                CpuOperations.residualAdd(state.x, state.xb2);
            }
        }

        CpuOperations.rmsNorm(state.x, state.x, weights.outputNorm, 0, dim, eps);
        CpuOperations.vocabProjection(
                weights.output, state.x, state.logits, config.vocabularySize(), dim);
        return state.logits;
    }

    /** The dense SwiGLU of the leading layers, from {@code xb} into {@code xb2}. */
    private static void denseFeedForward(
            Lfm2Configuration config, Lfm2StandardWeights weights, Lfm2State state, int l) {
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        CpuOperations.matVec(weights.ffnGate[l], state.xb, state.hb, hidden, dim);
        CpuOperations.matVec(weights.ffnUp[l], state.xb, state.hb2, hidden, dim);
        CpuOperations.swiGLU(state.hb, state.hb2);
        CpuOperations.matVec(weights.ffnDown[l], state.hb, state.xb2, dim, hidden);
    }

    // @formatter:off
    /**
     * The routed feed-forward, from {@code xb} straight into the residual stream.
     *
     * <p>Each selected expert's output is added to {@code x} scaled by its normalized weight, in
     * selection order; there is no separate residual add afterwards and no shared expert.
     */
    // @formatter:on
    private static void mixtureOfExperts(
            Lfm2Configuration config, Lfm2StandardWeights weights, Lfm2State state, int l) {
        final int dim = config.dim();
        CpuOperations.moeRouterSigmoid(
                state.xb,
                weights.ffnGateInp[l],
                weights.expProbsBias[l],
                state.routerScores,
                state.selectedExperts,
                state.selectedExpertWeights,
                config.expertCount(),
                config.expertUsedCount(),
                dim);
        for (int j = 0; j < config.expertUsedCount(); j++) {
            CpuOperations.expertFeedForward(
                    state.xb,
                    state.selectedExperts[j],
                    weights.ffnGateExps[l],
                    weights.ffnUpExps[l],
                    weights.ffnDownExps[l],
                    state.hbE,
                    state.hbE2,
                    state.yTmp,
                    config.expertHiddenDim(),
                    dim);
            CpuOperations.weightedAccumulate(
                    state.x, state.yTmp, state.selectedExpertWeights[j], dim);
        }
    }

    /**
     * Grouped-query attention with per-head query and key RMS norms and NEOX RoPE over the whole
     * head, from {@code xb} into {@code xb2}.
     */
    private static void attentionBranch(
            Lfm2Configuration config,
            Lfm2StandardWeights weights,
            Lfm2State state,
            int l,
            int position) {

        final int dim = config.dim();
        final int headSize = config.headSize();
        final int heads = config.numberOfHeads();
        final int kvHeads = config.numberOfKeyValueHeads();
        final int kvDim = config.kvDim();
        final float eps = config.rmsNormEps();

        CpuOperations.matVec(weights.wq[l], state.xb, state.q, dim, dim);
        CpuOperations.matVec(weights.wk[l], state.xb, state.k, kvDim, dim);
        CpuOperations.matVec(weights.wv[l], state.xb, state.v, kvDim, dim);

        for (int h = 0; h < heads; h++) {
            CpuOperations.rmsNorm(
                    state.q, state.q, weights.attnQNorm[l], h * headSize, headSize, eps);
        }
        for (int h = 0; h < kvHeads; h++) {
            CpuOperations.rmsNorm(
                    state.k, state.k, weights.attnKNorm[l], h * headSize, headSize, eps);
        }

        CpuOperations.ropeNeox(
                state.q,
                state.k,
                weights.freqCisReal,
                weights.freqCisImag,
                position,
                heads,
                kvHeads,
                headSize);

        CpuOperations.appendKeyValue(
                state.k, state.v, state.keyCache[l], state.valueCache[l], position, kvDim);

        AttentionShape shape =
                AttentionShape.uniform(
                        heads,
                        config.kvMul(),
                        headSize,
                        kvDim,
                        config.contextLength(),
                        (float) Math.sqrt(headSize));
        CpuOperations.attention(
                state.q,
                state.keyCache[l],
                state.valueCache[l],
                state.att,
                state.xb,
                position,
                shape);

        CpuOperations.matVec(weights.wo[l], state.xb, state.xb2, dim, dim);
    }

    // @formatter:off
    /**
     * The gated short convolution, from {@code xb} into {@code xb2}.
     *
     * <pre>
     *   B ‖ C ‖ x = in_proj(xb)
     *   y         = C ⊙ conv(B ⊙ x)        // depthwise, causal, kernel taps per channel
     *   xb2       = out_proj(y)
     * </pre>
     *
     * <p>The window holds the gated input {@code B ⊙ x}, not {@code x}: the product is formed
     * before the convolution, and it is what llama.cpp retains between steps. There is no
     * activation after the convolution.
     */
    // @formatter:on
    private static void shortConvBranch(
            Lfm2Configuration config, Lfm2StandardWeights weights, Lfm2State state, int l) {
        final int dim = config.dim();
        CpuOperations.matVec(
                weights.convInProj[l],
                state.xb,
                state.convBcx,
                config.shortConvProjectionDim(),
                dim);
        state.convBcx.copyTo(0, state.convBx, 0, dim);
        for (int i = 0; i < dim; i++) {
            state.convBx.setFloat(
                    i, state.convBx.getFloat(i) * state.convBcx.getFloat(2 * dim + i));
        }
        CpuOperations.causalConv1d(
                state.convBx,
                weights.convKernel[l],
                state.convState[l],
                state.convOut,
                dim,
                config.shortConvCache());
        for (int i = 0; i < dim; i++) {
            state.convOut.setFloat(i, state.convOut.getFloat(i) * state.convBcx.getFloat(dim + i));
        }
        CpuOperations.matVec(weights.convOutProj[l], state.convOut, state.xb2, dim, dim);
    }
}
