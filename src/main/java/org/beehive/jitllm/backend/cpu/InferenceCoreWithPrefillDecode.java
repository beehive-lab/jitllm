package org.beehive.jllm.backend.cpu;

import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.StandardWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Low-level forward passes for the prefill/decode separated inference path.
 *
 * <p>Parallel to {@link InferenceCore} — does NOT modify it.
 *
 * <p>The key addition is {@link #forwardJavaPrefill}, which runs a full transformer forward pass
 * but skips the final RMSNorm and vocabulary projection (wcls matmul). This is correct for all
 * prefill positions because their logits are discarded anyway; only the KV-cache update matters.
 * Skipping the projection saves one large matmul (vocabularySize × dim) per prefill token.
 */
public final class InferenceCoreWithPrefillDecode {

    private InferenceCoreWithPrefillDecode() {}

    /**
     * Prefill-only forward pass for LLaMA (CPU, FP32 weights).
     *
     * <p>Identical to {@link InferenceCore#forwardJava} except the final RMSNorm and vocabulary
     * projection are omitted. The KV cache is populated correctly at {@code position}.
     *
     * @param model the LLaMA model (must carry {@link StandardWeights})
     * @param state mutable inference state (KV cache, activations …)
     * @param token input token id
     * @param position sequence position being processed
     */
    public static void forwardJavaPrefill(Model model, State state, int token, int position) {
        final Configuration config = model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // Token embedding
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Transformer layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // Attention RMSNorm
            InferenceCore.rmsnorm(
                    state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // QKV projections
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr =
                        weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci =
                        weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // KV cache update
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            // Multi-head attention
            int curLayer = l;
            Parallel.parallelFor(
                    0,
                    config.numberOfHeads(),
                    h -> {
                        int qOffset = h * headSize;
                        int attOffset = h * config.contextLength();

                        for (int t = 0; t <= position; t++) {
                            int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
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
                            int vOffset = t * kvDim + (h / kvMul) * headSize;
                            float a = state.att.getFloat(attOffset + t);
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                        }
                    });

            // Attention output projection + residual
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);
            state.x.addInPlace(state.xb2);

            // FFN RMSNorm
            InferenceCore.rmsnorm(
                    state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // FFN (SwiGLU)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // FFN residual
            state.x.addInPlace(state.xb);
        }

        // Final RMSNorm and vocab projection intentionally omitted:
        // logits are not needed for prefill positions — only the KV cache matters.
    }
}
