package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.inference.weights.standard.DeepSeek2StandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass of a {@code deepseek2} model, one token at a time.
 *
 * <p>It is the reference the device path is checked against, and follows llama.cpp's graph for this
 * architecture operation for operation: absorbed latent attention, then either the dense SwiGLU of
 * a leading block or the routed experts plus the shared expert.
 */
public final class DeepSeek2Forward {

    private DeepSeek2Forward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (DeepSeek2Configuration) model.configuration(),
                (DeepSeek2StandardWeights) model.weights(),
                (DeepSeek2State) state,
                token,
                position);
    }

    public static FloatTensor forward(
            DeepSeek2Configuration config,
            DeepSeek2StandardWeights weights,
            DeepSeek2State state,
            int token,
            int position) {
        final int dim = config.dim();
        final float eps = config.rmsNormEps();
        DeepSeek2LayerWeights<FloatTensor> w = weights.layers;

        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(state.xb, state.x, w.attnNorm()[l], 0, dim, eps);
            attention(config, weights, state, l, position);
            CpuOperations.residualAdd(state.x, state.xb2);

            CpuOperations.rmsNorm(state.xb, state.x, w.ffnNorm()[l], 0, dim, eps);
            if (config.isDenseLayer(l)) {
                denseFeedForward(config, w, state, l);
            } else {
                mixtureOfExperts(config, w, state, l);
            }
            CpuOperations.residualAdd(state.x, state.xb2);
        }

        CpuOperations.rmsNorm(state.x, state.x, weights.outputNorm, 0, dim, eps);
        CpuOperations.vocabProjection(
                weights.output, state.x, state.logits, config.vocabularySize(), dim);
        return state.logits;
    }

    // @formatter:off
    /**
     * Absorbed multi-head latent attention, leaving {@code attn_output}'s result in {@code xb2}.
     *
     * <ol>
     *   <li>queries: {@code q_a}, its norm, {@code q_b}; per head a no-rope part and a rotated
     *       part;
     *   <li>the position's key: {@code kv_a_mqa} gives a latent, normalized, and a rotated key;
     *       both are stored, and the latent is also the value;
     *   <li>each head's no-rope query is absorbed into the latent space through {@code k_b};
     *   <li>scores over the stored keys, a softmax, and the weighted sum of stored latents;
     *   <li>each head's latent goes back out through {@code v_b}, and {@code attn_output} mixes
     *       them.
     * </ol>
     */
    // @formatter:on
    private static void attention(
            DeepSeek2Configuration config,
            DeepSeek2StandardWeights weights,
            DeepSeek2State state,
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
        final float eps = config.rmsNormEps();
        DeepSeek2LayerWeights<FloatTensor> w = weights.layers;

        CpuOperations.matVec(w.qA()[l], state.xb, state.queryLatent, config.qLoraRank(), dim);
        CpuOperations.rmsNorm(
                state.queryLatent, state.queryLatent, w.qANorm()[l], 0, config.qLoraRank(), eps);
        CpuOperations.matVec(
                w.qB()[l], state.queryLatent, state.q, config.queryDim(), config.qLoraRank());

        CpuOperations.matVec(
                w.kvAMqa()[l],
                state.xb,
                state.compressedKeyValue,
                config.compressedKeyValueDim(),
                dim);
        CpuOperations.rmsNorm(
                state.compressedKeyValue, state.compressedKeyValue, w.kvANorm()[l], 0, rank, eps);
        ropePairs(state.compressedKeyValue, rank, rope, position, weights);
        for (int h = 0; h < heads; h++) {
            ropePairs(state.q, h * qHead + nope, rope, position, weights);
        }
        FloatTensor keys = state.keyCache[l];
        state.compressedKeyValue.copyTo(0, keys, position * keyWidth, keyWidth);

        // Absorb: per head, latent[r] = k_b[h][r] . q_nope[h].
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
        CpuOperations.matVec(w.wo()[l], state.xb, state.xb2, dim, config.attentionOutputInputDim());
    }

    /** Interleaved-pair rotation of {@code width} values starting at {@code offset}. */
    private static void ropePairs(
            FloatTensor vec,
            int offset,
            int width,
            int position,
            DeepSeek2StandardWeights weights) {
        int base = position * (width / 2);
        for (int i = 0; i < width; i += 2) {
            float fcr = weights.freqCisReal.getFloat(base + i / 2);
            float fci = weights.freqCisImag.getFloat(base + i / 2);
            float v0 = vec.getFloat(offset + i);
            float v1 = vec.getFloat(offset + i + 1);
            vec.setFloat(offset + i, v0 * fcr - v1 * fci);
            vec.setFloat(offset + i + 1, v0 * fci + v1 * fcr);
        }
    }

    private static void denseFeedForward(
            DeepSeek2Configuration config,
            DeepSeek2LayerWeights<FloatTensor> w,
            DeepSeek2State state,
            int l) {
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        CpuOperations.matVec(w.ffnGate()[l], state.xb, state.hb, hidden, dim);
        CpuOperations.matVec(w.ffnUp()[l], state.xb, state.hb2, hidden, dim);
        CpuOperations.swiGLU(state.hb, state.hb2);
        CpuOperations.matVec(w.ffnDown()[l], state.hb, state.xb2, dim, hidden);
    }

    // @formatter:off
    /**
     * The routed experts plus the shared expert, leaving their sum in {@code xb2}.
     *
     * <p>Routing is llama.cpp's {@code build_moe_ffn}: router logits, then sigmoid (or softmax)
     * probabilities; the selection bias is added for choosing only, so the chosen weights are the
     * unbiased probabilities; normalized to sum to one when the file says so, then scaled.
     */
    // @formatter:on
    private static void mixtureOfExperts(
            DeepSeek2Configuration config,
            DeepSeek2LayerWeights<FloatTensor> w,
            DeepSeek2State state,
            int l) {
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        FloatTensor probs = state.routerScores;
        FloatTensor select = state.selectionScores;

        w.router()[l].matmul(state.xb, probs, experts, dim);
        if (config.sigmoidGating()) {
            for (int e = 0; e < experts; e++) {
                probs.setFloat(e, CpuOperations.logistic(probs.getFloat(e)));
            }
        } else {
            probs.softmaxInPlace(0, experts);
        }
        FloatTensor bias = w.routerBias()[l];
        for (int e = 0; e < experts; e++) {
            select.setFloat(e, probs.getFloat(e) + (bias != null ? bias.getFloat(e) : 0f));
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
