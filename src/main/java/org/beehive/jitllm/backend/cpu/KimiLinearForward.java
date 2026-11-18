package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.inference.weights.standard.KimiLinearStandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass of a {@code kimi-linear} model, one token at a time.
 *
 * <p>It follows llama.cpp's graph for this architecture operation for operation: a pre-norm
 * residual block whose token mixer is KDA ({@link KdaForward}, softplus decay) or absorbed latent
 * attention without RoPE, and whose feed-forward is the dense SwiGLU of a leading block or the
 * sigmoid-routed experts plus the shared experts.
 */
public final class KimiLinearForward {

    private KimiLinearForward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (KimiLinearConfiguration) model.configuration(),
                (KimiLinearStandardWeights) model.weights(),
                (KimiLinearState) state,
                token,
                position);
    }

    public static FloatTensor forward(
            KimiLinearConfiguration config,
            KimiLinearStandardWeights weights,
            KimiLinearState state,
            int token,
            int position) {
        final int dim = config.dim();
        final float eps = config.rmsNormEps();
        KimiLinearLayerWeights<FloatTensor> w = weights.layers;
        KdaForward.Shape kda = kdaShape(config);

        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(state.xb, state.x, w.attnNorm()[l], 0, dim, eps);
            if (config.isKda(l)) {
                KdaForward.forward(kda, w.kda(), state.kda, state.xb, state.xb2, l);
            } else {
                latentAttention(config, w.mla(), state, l, position);
            }
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(state.xb, state.x, w.ffnNorm()[l], 0, dim, eps);
            if (config.isDenseLayer(l)) {
                denseFeedForward(config, w.ffn(), state, l);
            } else {
                mixtureOfExperts(config, w.ffn(), state, l);
            }
            CpuOperations.residualAdd(state.x, state.xb2);
        }

        CpuOperations.rmsNorm(state.x, state.x, weights.outputNorm, 0, dim, eps);
        CpuOperations.vocabProjection(
                weights.output, state.x, state.logits, config.vocabularySize(), dim);
        return state.logits;
    }

    /** The KDA geometry: llama.cpp's softplus decay, its L2 norm with the model's RMS epsilon. */
    public static KdaForward.Shape kdaShape(KimiLinearConfiguration config) {
        return new KdaForward.Shape(
                config.dim(),
                config.kdaHeads(),
                config.kdaHeadDim(),
                config.kdaConvKernel(),
                KdaForward.Decay.SOFTPLUS,
                0f,
                config.rmsNormEps(),
                config.rmsNormEps());
    }

    // @formatter:off
    /**
     * Absorbed multi-head latent attention without RoPE, leaving {@code attn_output}'s result in
     * {@code xb2}.
     *
     * <ol>
     *   <li>queries straight from {@code attn_q}: per head a no-rope part and a tail;
     *   <li>the position's key: {@code kv_a_mqa} gives a latent, normalized, and a key tail shared
     *       by every head; both are stored, unrotated, and the latent is also the value;
     *   <li>each head's no-rope query is absorbed into the latent space through {@code k_b};
     *   <li>scores over the stored keys, a softmax, and the weighted sum of stored latents;
     *   <li>each head's latent goes back out through {@code v_b}, and {@code attn_output} mixes
     *       them.
     * </ol>
     */
    // @formatter:on
    static void latentAttention(
            KimiLinearConfiguration config,
            KimiLinearLayerWeights.Mla<FloatTensor> w,
            KimiLinearState state,
            int l,
            int position) {
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int nope = config.nopeDim();
        final int rope = config.ropeDim();
        final int qHead = config.queryHeadDim();
        final int rank = config.kvLoraRank();
        final int keyWidth = config.keyWidth();
        final int valueHead = config.valueHeadDim();

        CpuOperations.matVec(w.q()[l], state.xb, state.q, config.queryDim(), dim);
        CpuOperations.matVec(
                w.kvAMqa()[l], state.xb, state.compressedKeyValue, config.keyWidth(), dim);
        CpuOperations.rmsNorm(
                state.compressedKeyValue,
                state.compressedKeyValue,
                w.kvANorm()[l],
                0,
                rank,
                config.rmsNormEps());
        FloatTensor keys = state.keyCache[l];
        state.compressedKeyValue.copyTo(0, keys, position * keyWidth, keyWidth);

        // Absorb: per head, latent[r] = k_b[h][r] . q_nope[h]; then the tail as it is.
        FloatTensor kB = w.kB()[l];
        FloatTensor q = state.q;
        FloatTensor absorbed = state.absorbedQuery;
        Parallel.parallelFor(
                0,
                heads * rank,
                hr -> {
                    int h = hr / rank;
                    int r = hr % rank;
                    absorbed.setFloat(h * keyWidth + r, kB.dot(hr * nope, q, h * qHead, nope));
                });
        for (int h = 0; h < heads; h++) {
            q.copyTo(h * qHead + nope, absorbed, h * keyWidth + rank, rope);
        }

        final float scale = config.attentionScale();
        final int context = config.contextLength();
        FloatTensor att = state.att;
        FloatTensor latentOut = state.latentOut;
        Parallel.parallelFor(
                0,
                heads,
                h -> {
                    int scores = h * context;
                    for (int t = 0; t <= position; t++) {
                        float s = absorbed.dot(h * keyWidth, keys, t * keyWidth, keyWidth);
                        att.setFloat(scores + t, s * scale);
                    }
                    att.softmaxInPlace(scores, position + 1);
                    latentOut.fillInPlace(h * rank, rank, 0f);
                    for (int t = 0; t <= position; t++) {
                        latentOut.saxpyInPlace(
                                h * rank, keys, t * keyWidth, rank, att.getFloat(scores + t));
                    }
                });

        // Decompress: per head, out[j] = v_b[h][j] . latent[h].
        FloatTensor vB = w.vB()[l];
        FloatTensor out = state.xb;
        Parallel.parallelFor(
                0,
                heads * valueHead,
                hj -> {
                    int h = hj / valueHead;
                    out.setFloat(hj, vB.dot(hj * rank, latentOut, h * rank, rank));
                });
        CpuOperations.matVec(
                w.output()[l], state.xb, state.xb2, dim, config.attentionOutputInputDim());
    }

    private static void denseFeedForward(
            KimiLinearConfiguration config,
            KimiLinearLayerWeights.FeedForward<FloatTensor> w,
            KimiLinearState state,
            int l) {
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        CpuOperations.matVec(w.gate()[l], state.xb, state.hb, hidden, dim);
        CpuOperations.matVec(w.up()[l], state.xb, state.hb2, hidden, dim);
        CpuOperations.swiGLU(state.hb, state.hb2);
        CpuOperations.matVec(w.down()[l], state.hb, state.xb2, dim, hidden);
    }

    // @formatter:off
    /**
     * The routed experts plus the shared experts, leaving their sum in {@code xb2}.
     *
     * <p>Routing is llama.cpp's {@code build_moe_ffn} with sigmoid scores: the selection bias is
     * added for choosing only, so the chosen weights are the unbiased sigmoids; they are normalized
     * to sum to one, then scaled by {@code expert_weights_scale}.
     */
    // @formatter:on
    private static void mixtureOfExperts(
            KimiLinearConfiguration config,
            KimiLinearLayerWeights.FeedForward<FloatTensor> w,
            KimiLinearState state,
            int l) {
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        FloatTensor probs = state.routerScores;
        FloatTensor select = state.selectionScores;

        w.router()[l].matmul(state.xb, probs, experts, dim);
        FloatTensor bias = w.routerBias()[l];
        for (int e = 0; e < experts; e++) {
            float p = CpuOperations.logistic(probs.getFloat(e));
            probs.setFloat(e, p);
            select.setFloat(e, p + bias.getFloat(e));
        }
        float total = 0f;
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            int index = -1;
            for (int e = 0; e < experts; e++) {
                if (select.getFloat(e) > best) {
                    best = select.getFloat(e);
                    index = e;
                }
            }
            select.setFloat(index, Float.NEGATIVE_INFINITY);
            state.expertIds[k] = index;
            state.expertWeights[k] = probs.getFloat(index);
            total += state.expertWeights[k];
        }
        float norm = config.expertWeightsNorm() ? Math.max(total, 6.103515625e-5f) : 1f;
        for (int k = 0; k < used; k++) {
            state.expertWeights[k] = state.expertWeights[k] / norm * config.expertWeightsScale();
        }

        state.xb2.fillInPlace(0, dim, 0f);
        for (int k = 0; k < used; k++) {
            CpuOperations.expertFeedForward(
                    state.xb,
                    state.expertIds[k],
                    w.gateExperts()[l],
                    w.upExperts()[l],
                    w.downExperts()[l],
                    state.expertHidden,
                    state.expertHiddenUp,
                    state.expertOut,
                    config.expertHiddenDim(),
                    dim);
            CpuOperations.weightedAccumulate(
                    state.xb2, state.expertOut, state.expertWeights[k], dim);
        }

        int shared = config.sharedHiddenDim();
        if (shared > 0) {
            CpuOperations.matVec(w.sharedGate()[l], state.xb, state.expertHidden, shared, dim);
            CpuOperations.matVec(w.sharedUp()[l], state.xb, state.expertHiddenUp, shared, dim);
            CpuOperations.swiGLU(state.expertHidden, state.expertHiddenUp);
            CpuOperations.matVec(
                    w.sharedDown()[l], state.expertHidden, state.expertOut, dim, shared);
            CpuOperations.weightedAccumulate(state.xb2, state.expertOut, 1f, dim);
        }
    }
}
