package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.inference.weights.standard.GptOssStandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

// @formatter:off
/**
 * gpt-oss's host forward pass: the reference every device path is judged against.
 *
 * <pre>
 *   EmbeddingLookup
 *   per layer:
 *     RmsNorm(attn_norm)
 *     MatVec q, k, v + BiasAdd
 *     RoPE (NeoX halves, YaRN tables)
 *     AppendKeyValue
 *     Attention over the window (sliding on even layers), sinks in the softmax denominator
 *     MatVec(attn_output) + BiasAdd, residual add
 *     RmsNorm(post_attention_norm)
 *     Router: MatVec + BiasAdd, top-k of the logits, softmax over the k chosen
 *     per chosen expert: gate, up (+ biases), clamped SwiGLU, down (+ bias), weighted accumulate
 *     residual add
 *   RmsNorm(output_norm), vocabulary projection
 * </pre>
 *
 * <p>The SwiGLU is gpt-oss's: {@code min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) +
 * 1)}. The sink of head {@code h} is one more logit in that head's softmax, with no value.
 */
// @formatter:on
public final class GptOssForward {

    private GptOssForward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        GptOssConfiguration config = (GptOssConfiguration) model.configuration();
        GptOssStandardWeights weights = (GptOssStandardWeights) model.weights();
        GptOssState s = (GptOssState) state;
        GptOssLayerWeights<FloatTensor> layers = weights.layers();

        int dim = config.dim();
        int qDim = config.queryDim();
        int kvDim = config.kvDim();
        int headSize = config.headSize();
        float eps = config.rmsNormEps();

        CpuOperations.embeddingLookup(weights.token_embedding_table, token, s.x, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            CpuOperations.rmsNorm(s.xb, s.x, weights.rms_att_weight[l], 0, dim, eps);
            weights.wq[l].matmul(s.xb, s.q, qDim, dim);
            weights.wk[l].matmul(s.xb, s.k, kvDim, dim);
            weights.wv[l].matmul(s.xb, s.v, kvDim, dim);
            CpuOperations.biasAdd(s.q, layers.queryBias()[l]);
            CpuOperations.biasAdd(s.k, layers.keyBias()[l]);
            CpuOperations.biasAdd(s.v, layers.valueBias()[l]);

            CpuOperations.ropeNeox(
                    s.q,
                    s.k,
                    weights.freq_cis_real,
                    weights.freq_cis_imag,
                    position,
                    config.numberOfHeads(),
                    config.numberOfKeyValueHeads(),
                    headSize);
            CpuOperations.appendKeyValue(s.k, s.v, s.keyCache[l], s.valueCache[l], position, kvDim);

            attention(
                    config,
                    s.q,
                    s.keyCache[l],
                    s.valueCache[l],
                    layers.sinks()[l],
                    s.att,
                    s.attentionOut,
                    config.windowStart(l, position),
                    position);

            weights.wo[l].matmul(s.attentionOut, s.xb2, dim, qDim);
            CpuOperations.biasAdd(s.xb2, layers.outputBias()[l]);
            CpuOperations.residualAdd(s.x, s.xb2);

            CpuOperations.rmsNorm(s.xb, s.x, weights.rms_ffn_weight[l], 0, dim, eps);
            route(config, layers, l, s);
            s.xb2.fillInPlace(0, dim, 0f);
            for (int k = 0; k < config.numberOfExpertsUsed(); k++) {
                expert(config, layers, l, s.selectedExperts[k], s);
                CpuOperations.weightedAccumulate(
                        s.xb2, s.expertOut, s.selectedExpertWeights[k], dim);
            }
            CpuOperations.residualAdd(s.x, s.xb2);
        }

        CpuOperations.rmsNorm(s.x, s.x, weights.rms_final_weight, 0, dim, eps);
        CpuOperations.vocabProjection(weights.wcls, s.x, s.logits, config.vocabularySize(), dim);
        return s.logits;
    }

    /**
     * Grouped-query attention over {@code [windowStart, position]} with one sink logit per head.
     * Scores are scaled by {@code 1 / sqrt(headSize)}; the sink joins the maximum and the
     * denominator and contributes no value.
     */
    static void attention(
            GptOssConfiguration config,
            FloatTensor query,
            FloatTensor keyCache,
            FloatTensor valueCache,
            FloatTensor sinks,
            FloatTensor scores,
            FloatTensor out,
            int windowStart,
            int position) {
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int kvMul = config.kvMul();
        int contextLength = config.contextLength();
        float scale = (float) (1.0 / Math.sqrt(headSize));
        Parallel.parallelFor(
                0,
                config.numberOfHeads(),
                h -> {
                    int qOffset = h * headSize;
                    int attOffset = h * contextLength;
                    int kvHead = (h / kvMul) * headSize;
                    float sink = sinks.getFloat(h);
                    float max = sink;
                    for (int t = windowStart; t <= position; t++) {
                        float score =
                                query.dot(qOffset, keyCache, t * kvDim + kvHead, headSize) * scale;
                        scores.setFloat(attOffset + t, score);
                        max = Math.max(max, score);
                    }
                    float sum = (float) Math.exp(sink - max);
                    for (int t = windowStart; t <= position; t++) {
                        float e = (float) Math.exp(scores.getFloat(attOffset + t) - max);
                        scores.setFloat(attOffset + t, e);
                        sum += e;
                    }
                    float inv = 1f / sum;
                    out.fillInPlace(qOffset, headSize, 0f);
                    for (int t = windowStart; t <= position; t++) {
                        out.saxpyInPlace(
                                qOffset,
                                valueCache,
                                t * kvDim + kvHead,
                                headSize,
                                scores.getFloat(attOffset + t) * inv);
                    }
                });
    }

    /** Router logits plus bias, the top {@code k} of them, and a softmax over those {@code k}. */
    static void route(
            GptOssConfiguration config,
            GptOssLayerWeights<FloatTensor> layers,
            int layer,
            GptOssState s) {
        int experts = config.numberOfExperts();
        int used = config.numberOfExpertsUsed();
        layers.router()[layer].matmul(s.xb, s.routerLogits, experts, config.dim());
        CpuOperations.biasAdd(s.routerLogits, layers.routerBias()[layer]);
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            int index = -1;
            for (int e = 0; e < experts; e++) {
                float v = s.routerLogits.getFloat(e);
                if (v > best) {
                    best = v;
                    index = e;
                }
            }
            s.selectedExperts[k] = index;
            s.selectedExpertWeights[k] = best;
            s.routerLogits.setFloat(index, Float.NEGATIVE_INFINITY);
        }
        float max = s.selectedExpertWeights[0];
        float sum = 0f;
        for (int k = 0; k < used; k++) {
            s.selectedExpertWeights[k] = (float) Math.exp(s.selectedExpertWeights[k] - max);
            sum += s.selectedExpertWeights[k];
        }
        for (int k = 0; k < used; k++) {
            s.selectedExpertWeights[k] /= sum;
        }
    }

    /** One expert: gate and up with biases, the clamped SwiGLU, and down with its bias. */
    static void expert(
            GptOssConfiguration config,
            GptOssLayerWeights<FloatTensor> layers,
            int layer,
            int expert,
            GptOssState s) {
        int dim = config.dim();
        int hidden = config.expertHiddenDim();
        FloatTensor gate = layers.gateExperts()[layer];
        FloatTensor up = layers.upExperts()[layer];
        FloatTensor down = layers.downExperts()[layer];
        FloatTensor gateBias = layers.gateExpertsBias()[layer];
        FloatTensor upBias = layers.upExpertsBias()[layer];
        FloatTensor downBias = layers.downExpertsBias()[layer];
        int gateUpBase = expert * hidden * dim;
        int hiddenBias = expert * hidden;
        Parallel.parallelFor(
                0,
                hidden,
                i -> {
                    float g =
                            gate.dot(gateUpBase + i * dim, s.xb, 0, dim)
                                    + gateBias.getFloat(hiddenBias + i);
                    float u =
                            up.dot(gateUpBase + i * dim, s.xb, 0, dim)
                                    + upBias.getFloat(hiddenBias + i);
                    s.expertHidden.setFloat(i, swiglu(g, u));
                });
        int downBase = expert * dim * hidden;
        int dimBias = expert * dim;
        Parallel.parallelFor(
                0,
                dim,
                i ->
                        s.expertOut.setFloat(
                                i,
                                down.dot(downBase + i * hidden, s.expertHidden, 0, hidden)
                                        + downBias.getFloat(dimBias + i)));
    }

    /** gpt-oss's SwiGLU: the gate capped at the limit, the linear part clamped and offset by 1. */
    public static float swiglu(float gate, float up) {
        float x = Math.min(gate, GptOssConfiguration.SWIGLU_LIMIT);
        float y =
                Math.max(
                        -GptOssConfiguration.SWIGLU_LIMIT,
                        Math.min(up, GptOssConfiguration.SWIGLU_LIMIT));
        float glu = x / (1f + (float) Math.exp(-GptOssConfiguration.SWIGLU_ALPHA * x));
        return glu * (y + 1f);
    }
}
