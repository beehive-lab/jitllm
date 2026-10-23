package org.beehive.jitllm.backend.cpu;

import java.util.List;
import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Glm5NextLayerWeights;
import org.beehive.jitllm.inference.weights.standard.Glm5NextStandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass of a {@code glm5next} model (GLM-5.3-Flash), one token at a time.
 *
 * <p>It follows the reference implementation operation for operation:
 *
 * <ol>
 *   <li>the token's embedding is copied into every residual stream;
 *   <li>each layer runs two sublayers, token mixing then feed-forward, each wrapped in a
 *       hyper-connection: the streams are collapsed into the sublayer's input, the input is
 *       normalized, and the sublayer's output is expanded back and added to the streams mixed by a
 *       doubly-stochastic matrix;
 *   <li>the streams are averaged, normalized, and projected to the vocabulary.
 * </ol>
 *
 * <p>Token mixing is either KDA, a delta-rule recurrence whose decay is per key channel, or sparse
 * latent attention without RoPE over the positions its indexer selects.
 */
public final class Glm5NextForward {

    private Glm5NextForward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (Glm5NextConfiguration) model.configuration(),
                (Glm5NextStandardWeights) model.weights(),
                (Glm5NextState) state,
                token,
                position);
    }

    public static FloatTensor forward(
            Glm5NextConfiguration config,
            Glm5NextStandardWeights weights,
            Glm5NextState state,
            int token,
            int position) {
        embed(config, weights, state, token);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            layer(config, weights, state, l, position);
        }
        return head(config, weights, state);
    }

    /** The token's embedding, copied into every residual stream. */
    static void embed(
            Glm5NextConfiguration config,
            Glm5NextStandardWeights weights,
            Glm5NextState state,
            int token) {
        final int dim = config.dim();
        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);
        for (int s = 0; s < config.hcCount(); s++) {
            state.x.copyTo(0, state.streams, s * dim, dim);
        }
    }

    /** One trunk layer: token mixing, then the feed-forward, each inside a hyper-connection. */
    static void layer(
            Glm5NextConfiguration config,
            Glm5NextStandardWeights weights,
            Glm5NextState state,
            int l,
            int position) {
        final int dim = config.dim();
        final float eps = config.rmsNormEps();
        Glm5NextLayerWeights<FloatTensor> w = weights.layers;
        Glm5NextLayerWeights.Mhc<FloatTensor> mhc = w.mhc();

        hyperConnectionPre(config, state, mhc.attnFn()[l], mhc.attnBase()[l], mhc.attnScale()[l]);
        CpuOperations.rmsNorm(state.xb, state.x, w.attnNorm()[l], 0, dim, eps);
        if (config.isKda(l)) {
            kda(config, w.kda(), state, l);
        } else {
            sparseAttention(config, w.sparse(), state, l, position);
        }
        hyperConnectionPost(config, state);

        hyperConnectionPre(config, state, mhc.ffnFn()[l], mhc.ffnBase()[l], mhc.ffnScale()[l]);
        CpuOperations.rmsNorm(state.xb, state.x, w.ffnNorm()[l], 0, dim, eps);
        feedForward(config, w.ffn(), state, l);
        hyperConnectionPost(config, state);
    }

    /** The streams' mean, normalized into {@code hidden}, then the vocabulary projection. */
    static FloatTensor head(
            Glm5NextConfiguration config, Glm5NextStandardWeights weights, Glm5NextState state) {
        final int dim = config.dim();
        final int n = config.hcCount();
        state.x.fillInPlace(0, dim, 0f);
        for (int s = 0; s < n; s++) {
            state.x.saxpyInPlace(0, state.streams, s * dim, dim, 1f / n);
        }
        CpuOperations.rmsNorm(
                state.hidden, state.x, weights.outputNorm, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.output, state.hidden, state.logits, config.vocabularySize(), dim);
        return state.logits;
    }

    // @formatter:off
    /**
     * The MTP layer: predicts the token after {@code nextToken} from it and from the trunk's last
     * hidden state, which {@link #forward} left in {@link Glm5NextState#hidden}.
     *
     * <p>{@code position} is the trunk position whose hidden state is used; the layer keeps its own
     * latent cache there. As in vLLM, the embedding is zeroed at position 0. The block is a plain
     * pre-norm residual block, without hyper-connections.
     */
    // @formatter:on
    public static FloatTensor forwardNextn(
            Glm5NextConfiguration config,
            Glm5NextStandardWeights weights,
            Glm5NextState state,
            int nextToken,
            int position) {
        final int dim = config.dim();
        final float eps = config.rmsNormEps();
        final int l = config.numberOfLayers();
        Glm5NextLayerWeights<FloatTensor> w = weights.layers;
        Glm5NextLayerWeights.Nextn<FloatTensor> nextn = w.nextn();

        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, nextToken, state.x, dim);
        if (position == 0) {
            state.x.fillInPlace(0, dim, 0f);
        }
        CpuOperations.rmsNorm(state.nextnConcat, state.x, nextn.embedNorm()[l], 0, dim, eps);
        state.hidden.copyTo(0, state.xb2, 0, dim);
        CpuOperations.rmsNorm(state.xb2, state.xb2, nextn.hiddenNorm()[l], 0, dim, eps);
        state.xb2.copyTo(0, state.nextnConcat, dim, dim);
        CpuOperations.matVec(
                nextn.inputProjection()[l], state.nextnConcat, state.nextnX, dim, 2 * dim);

        CpuOperations.rmsNorm(state.xb, state.nextnX, w.attnNorm()[l], 0, dim, eps);
        sparseAttention(config, w.sparse(), state, l, position);
        CpuOperations.residualAdd(state.nextnX, state.xb2);
        CpuOperations.rmsNorm(state.xb, state.nextnX, w.ffnNorm()[l], 0, dim, eps);
        feedForward(config, w.ffn(), state, l);
        CpuOperations.residualAdd(state.nextnX, state.xb2);

        CpuOperations.rmsNorm(state.nextnX, state.nextnX, nextn.headNorm()[l], 0, dim, eps);
        CpuOperations.vocabProjection(
                weights.output, state.nextnX, state.nextnLogits, config.vocabularySize(), dim);
        return state.nextnLogits;
    }

    // ── Hyper-connections ─────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Computes the sublayer's mHC coefficients from the streams and collapses them into {@code x}.
     *
     * <ul>
     *   <li>the streams, flattened, are RMS-normalized without a weight and projected by {@code fn}
     *       to {@code (2 + n) * n} values;
     *   <li>{@code pre = sigmoid(a * s0 + b) + eps} collapses the streams, {@code post = 2 *
     *       sigmoid(a * s1 + b)} expands the output;
     *   <li>the mix is a row softmax plus eps, then Sinkhorn: a column normalization, followed by
     *       {@code iterations - 1} row-then-column normalizations.
     * </ul>
     */
    // @formatter:on
    static void hyperConnectionPre(
            Glm5NextConfiguration config,
            Glm5NextState state,
            FloatTensor fn,
            FloatTensor base,
            FloatTensor scale) {
        final int dim = config.dim();
        final int n = config.hcCount();
        final int width = n * dim;
        final float eps = config.hcEps();
        FloatTensor streams = state.streams;
        streams.copyTo(0, state.residualStreams, 0, width);

        float ss = streams.reduce(0, width, 0f, (acc, v) -> acc + v * v);
        final float inv = (float) (1.0 / Math.sqrt(ss / width + config.rmsNormEps()));
        float[] mixes = state.hcMixes;
        Parallel.parallelFor(
                0, mixes.length, k -> mixes[k] = inv * fn.dot(k * width, streams, 0, width));

        float s0 = scale.getFloat(0);
        float s1 = scale.getFloat(1);
        float s2 = scale.getFloat(2);
        for (int i = 0; i < n; i++) {
            state.hcPre[i] = sigmoid(mixes[i] * s0 + base.getFloat(i)) + eps;
            state.hcPost[i] =
                    config.hcPostScale() * sigmoid(mixes[n + i] * s1 + base.getFloat(n + i));
        }
        float[] comb = state.hcComb;
        for (int r = 0; r < n; r++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < n; c++) {
                int k = 2 * n + r * n + c;
                comb[r * n + c] = mixes[k] * s2 + base.getFloat(k);
                max = Math.max(max, comb[r * n + c]);
            }
            float sum = 0f;
            for (int c = 0; c < n; c++) {
                comb[r * n + c] = (float) Math.exp(comb[r * n + c] - max);
                sum += comb[r * n + c];
            }
            for (int c = 0; c < n; c++) {
                comb[r * n + c] = comb[r * n + c] / sum + eps;
            }
        }
        normalizeColumns(comb, n, eps);
        for (int it = 1; it < config.hcSinkhornIterations(); it++) {
            normalizeRows(comb, n, eps);
            normalizeColumns(comb, n, eps);
        }

        state.x.fillInPlace(0, dim, 0f);
        for (int s = 0; s < n; s++) {
            state.x.saxpyInPlace(0, streams, s * dim, dim, state.hcPre[s]);
        }
    }

    /** {@code streams[k] = post[k] * xb2 + sum_m comb[m][k] * residual[m]}. */
    static void hyperConnectionPost(Glm5NextConfiguration config, Glm5NextState state) {
        final int dim = config.dim();
        final int n = config.hcCount();
        FloatTensor out = state.streams;
        FloatTensor residual = state.residualStreams;
        FloatTensor branch = state.xb2;
        Parallel.parallelFor(
                0,
                n,
                k -> {
                    for (int i = 0; i < dim; i++) {
                        float v = state.hcPost[k] * branch.getFloat(i);
                        for (int m = 0; m < n; m++) {
                            v += state.hcComb[m * n + k] * residual.getFloat(m * dim + i);
                        }
                        out.setFloat(k * dim + i, v);
                    }
                });
    }

    private static void normalizeRows(float[] comb, int n, float eps) {
        for (int r = 0; r < n; r++) {
            float sum = 0f;
            for (int c = 0; c < n; c++) {
                sum += comb[r * n + c];
            }
            for (int c = 0; c < n; c++) {
                comb[r * n + c] /= sum + eps;
            }
        }
    }

    private static void normalizeColumns(float[] comb, int n, float eps) {
        for (int c = 0; c < n; c++) {
            float sum = 0f;
            for (int r = 0; r < n; r++) {
                sum += comb[r * n + c];
            }
            for (int r = 0; r < n; r++) {
                comb[r * n + c] /= sum + eps;
            }
        }
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Kimi delta attention for one token, leaving {@code attn_output}'s result in {@code xb2}.
     *
     * <ol>
     *   <li>query, key and value projections, each through its own causal depthwise convolution and
     *       a SiLU; query and key L2-normalized per head, the query scaled by {@code
     *       1/sqrt(headDim)};
     *   <li>a per-channel log decay {@code g = lowerBound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) +
     *       dt_bias))}, so the decay {@code exp(g)} stays within {@code (exp(lowerBound), 1)};
     *   <li>per head {@code S = diag(exp(g)) S}, {@code d = (v - Sᵀk) * beta}, {@code S += k ⊗ d},
     *       {@code o = Sᵀq};
     *   <li>{@code o} RMS-normalized per head and gated by {@code sigmoid(g_b(g_a(x)))}.
     * </ol>
     */
    // @formatter:on
    static void kda(
            Glm5NextConfiguration config,
            Glm5NextLayerWeights.Kda<FloatTensor> w,
            Glm5NextState state,
            int l) {
        final int dim = config.dim();
        final int heads = config.kdaHeads();
        final int hd = config.kdaHeadDim();
        final int width = config.kdaDim();
        final int kernel = config.kdaConvKernel();
        FloatTensor x = state.xb;

        CpuOperations.matVec(w.q()[l], x, state.kdaQ, width, dim);
        CpuOperations.matVec(w.k()[l], x, state.kdaK, width, dim);
        CpuOperations.matVec(w.v()[l], x, state.kdaV, width, dim);
        CpuOperations.causalConv1d(
                state.kdaQ, w.convQ()[l], state.convWindowQ[l], state.kdaConvQ, width, kernel);
        CpuOperations.causalConv1d(
                state.kdaK, w.convK()[l], state.convWindowK[l], state.kdaConvK, width, kernel);
        CpuOperations.causalConv1d(
                state.kdaV, w.convV()[l], state.convWindowV[l], state.kdaConvV, width, kernel);
        FloatTensor q = state.kdaConvQ;
        FloatTensor k = state.kdaConvK;
        FloatTensor v = state.kdaConvV;
        q.mapInPlace(Glm5NextForward::silu);
        k.mapInPlace(Glm5NextForward::silu);
        v.mapInPlace(Glm5NextForward::silu);
        final float qScale = (float) (1.0 / Math.sqrt(hd));
        for (int h = 0; h < heads; h++) {
            l2Norm(q, h * hd, hd, qScale);
            l2Norm(k, h * hd, hd, 1f);
        }

        CpuOperations.matVec(w.decayA()[l], x, state.kdaLowRank, hd, dim);
        CpuOperations.matVec(w.decayB()[l], state.kdaLowRank, state.kdaDecay, width, hd);
        FloatTensor dtBias = w.dtBias()[l];
        FloatTensor aLog = w.aLog()[l];
        final float lower = config.kdaGateLowerBound();
        for (int c = 0; c < width; c++) {
            float rate = (float) Math.exp(aLog.getFloat(c / hd));
            float g = lower * sigmoid(rate * (state.kdaDecay.getFloat(c) + dtBias.getFloat(c)));
            state.kdaDecay.setFloat(c, (float) Math.exp(g));
        }
        CpuOperations.matVec(w.beta()[l], x, state.kdaBeta, heads, dim);
        for (int h = 0; h < heads; h++) {
            state.kdaBeta.setFloat(h, sigmoid(state.kdaBeta.getFloat(h)));
        }

        FloatTensor s = state.deltaState[l];
        FloatTensor decay = state.kdaDecay;
        FloatTensor out = state.kdaOut;
        Parallel.parallelFor(
                0,
                heads,
                h -> deltaRuleHead(q, k, v, decay, state.kdaBeta.getFloat(h), s, out, h, hd));

        CpuOperations.matVec(w.gateA()[l], x, state.kdaLowRank, hd, dim);
        CpuOperations.matVec(w.gateB()[l], state.kdaLowRank, state.kdaGate, width, hd);
        FloatTensor outNorm = w.outNorm()[l];
        final float eps = config.rmsNormEps();
        for (int h = 0; h < heads; h++) {
            int base = h * hd;
            float ss = out.reduce(base, hd, 0f, (acc, e) -> acc + e * e);
            float inv = (float) (1.0 / Math.sqrt(ss / hd + eps));
            for (int i = 0; i < hd; i++) {
                float g = sigmoid(state.kdaGate.getFloat(base + i));
                out.setFloat(base + i, outNorm.getFloat(i) * inv * out.getFloat(base + i) * g);
            }
        }
        CpuOperations.matVec(w.output()[l], out, state.xb2, dim, width);
    }

    /** One head's delta-rule step with a decay per key channel; state row = key channel. */
    private static void deltaRuleHead(
            FloatTensor q,
            FloatTensor k,
            FloatTensor v,
            FloatTensor decay,
            float beta,
            FloatTensor state,
            FloatTensor out,
            int h,
            int hd) {
        final int stateBase = h * hd * hd;
        final int base = h * hd;
        final float[] correction = new float[hd];
        for (int i = 0; i < hd; i++) {
            final float g = decay.getFloat(base + i);
            final float ki = k.getFloat(base + i);
            final int row = stateBase + i * hd;
            for (int j = 0; j < hd; j++) {
                float decayed = state.getFloat(row + j) * g;
                state.setFloat(row + j, decayed);
                correction[j] += decayed * ki;
            }
        }
        for (int j = 0; j < hd; j++) {
            correction[j] = (v.getFloat(base + j) - correction[j]) * beta;
        }
        final float[] o = new float[hd];
        for (int i = 0; i < hd; i++) {
            final float ki = k.getFloat(base + i);
            final float qi = q.getFloat(base + i);
            final int row = stateBase + i * hd;
            for (int j = 0; j < hd; j++) {
                float updated = state.getFloat(row + j) + ki * correction[j];
                state.setFloat(row + j, updated);
                o[j] += updated * qi;
            }
        }
        for (int j = 0; j < hd; j++) {
            out.setFloat(base + j, o[j]);
        }
    }

    /** {@code x / sqrt(sum(x²) + eps) * scale}: the norm FLA uses, with eps under the root. */
    private static void l2Norm(FloatTensor values, int offset, int size, float scale) {
        float ss = values.reduce(offset, size, 0f, (acc, e) -> acc + e * e);
        final float inv = (float) (scale / Math.sqrt(ss + Glm5NextConfiguration.KDA_L2_NORM_EPS));
        for (int i = 0; i < size; i++) {
            values.setFloat(offset + i, values.getFloat(offset + i) * inv);
        }
    }

    // ── Sparse attention ──────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Sparse latent attention without RoPE for one token, leaving the result in {@code xb2}.
     *
     * <ol>
     *   <li>the query latent {@code q_a}, its norm, {@code q_b}: a query per head;
     *   <li>the position's latent: {@code kv_a_mqa} and its norm, stored; it is every head's key
     *       and value through {@code k_b} and {@code v_b};
     *   <li>the indexer chooses the positions to attend to ({@link #indexerSelect});
     *   <li>absorbed attention over them: each head's query through {@code k_b} into latent space,
     *       scores against the stored latents, a softmax, the weighted latent back out through
     *       {@code v_b}, then {@code attn_output}.
     * </ol>
     */
    // @formatter:on
    static void sparseAttention(
            Glm5NextConfiguration config,
            Glm5NextLayerWeights.SparseAttention<FloatTensor> w,
            Glm5NextState state,
            int l,
            int position) {
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int keyHead = config.keyHeadDim();
        final int rank = config.kvLoraRank();
        final int valueHead = config.valueHeadDim();
        final float eps = config.rmsNormEps();
        FloatTensor x = state.xb;

        CpuOperations.matVec(w.qA()[l], x, state.queryLatent, config.qLoraRank(), dim);
        CpuOperations.rmsNorm(
                state.queryLatent, state.queryLatent, w.qANorm()[l], 0, config.qLoraRank(), eps);
        CpuOperations.matVec(
                w.qB()[l], state.queryLatent, state.q, config.queryDim(), config.qLoraRank());
        CpuOperations.matVec(w.kvA()[l], x, state.kvLatent, rank, dim);
        CpuOperations.rmsNorm(state.kvLatent, state.kvLatent, w.kvANorm()[l], 0, rank, eps);
        FloatTensor latents = state.keyCache[l];
        state.kvLatent.copyTo(0, latents, position * rank, rank);

        indexerSelect(config, w, state, l, position);

        FloatTensor kB = w.kB()[l];
        FloatTensor q = state.q;
        FloatTensor absorbed = state.absorbedQuery;
        Parallel.parallelFor(
                0,
                heads * rank,
                hr ->
                        absorbed.setFloat(
                                hr, kB.dot(hr * keyHead, q, (hr / rank) * keyHead, keyHead)));

        final float scale = config.attentionScale();
        final int context = config.contextLength();
        final int count = state.selectedCount;
        final int[] selected = state.selected;
        FloatTensor att = state.att;
        FloatTensor latentOut = state.latentOut;
        Parallel.parallelFor(
                0,
                heads,
                h -> {
                    int scores = h * context;
                    for (int j = 0; j < count; j++) {
                        float s = absorbed.dot(h * rank, latents, selected[j] * rank, rank);
                        att.setFloat(scores + j, s * scale);
                    }
                    att.softmaxInPlace(scores, count);
                    latentOut.fillInPlace(h * rank, rank, 0f);
                    for (int j = 0; j < count; j++) {
                        latentOut.saxpyInPlace(
                                h * rank,
                                latents,
                                selected[j] * rank,
                                rank,
                                att.getFloat(scores + j));
                    }
                });

        FloatTensor vB = w.vB()[l];
        FloatTensor out = state.xb;
        Parallel.parallelFor(
                0,
                heads * valueHead,
                hj ->
                        out.setFloat(
                                hj, vB.dot(hj * rank, latentOut, (hj / valueHead) * rank, rank)));
        CpuOperations.matVec(w.output()[l], out, state.xb2, dim, config.attentionOutputInputDim());
    }

    // @formatter:off
    /**
     * The k-pool indexer: fills {@link Glm5NextState#selected} with the positions this query
     * attends to, ascending.
     *
     * <ul>
     *   <li>each position stores an indexer key, {@code LayerNorm(wk x)}, and a per-channel pool
     *       gate, {@code gate x};
     *   <li>once a pool of {@code poolSize} positions is complete its compressed key is fixed: per
     *       channel, a softmax over the pool of {@code gate + ape[slot]} weights the keys;
     *   <li>a pool's score is {@code sum_h w_h * relu(q_h . key * scale)}, with {@code q} from the
     *       query latent and {@code w} from the input;
     *   <li>the best {@code topK / poolSize} complete pools are attended, plus the incomplete pool
     *       holding this position.
     * </ul>
     */
    // @formatter:on
    static void indexerSelect(
            Glm5NextConfiguration config,
            Glm5NextLayerWeights.SparseAttention<FloatTensor> w,
            Glm5NextState state,
            int l,
            int position) {
        final int dim = config.dim();
        final int ih = config.indexerHeads();
        final int id = config.indexerHeadDim();
        final int pool = config.indexerPoolSize();
        FloatTensor x = state.xb;
        FloatTensor keys = state.indexKeyCache[l];
        FloatTensor gates = state.poolGateCache[l];
        FloatTensor poolKeys = state.poolKeyCache[l];

        CpuOperations.matVec(w.indexK()[l], x, state.indexKey, id, dim);
        layerNorm(state.indexKey, id, w.indexKNorm()[l], w.indexKNormBias()[l]);
        state.indexKey.copyTo(0, keys, position * id, id);
        CpuOperations.matVec(w.poolGate()[l], x, state.poolGate, id, dim);
        state.poolGate.copyTo(0, gates, position * id, id);

        if ((position + 1) % pool == 0) {
            int p = position / pool;
            int first = p * pool;
            FloatTensor ape = w.poolApe()[l];
            for (int c = 0; c < id; c++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < pool; j++) {
                    max =
                            Math.max(
                                    max,
                                    gates.getFloat((first + j) * id + c)
                                            + ape.getFloat(j * id + c));
                }
                float sum = 0f;
                float acc = 0f;
                for (int j = 0; j < pool; j++) {
                    float e =
                            (float)
                                    Math.exp(
                                            gates.getFloat((first + j) * id + c)
                                                    + ape.getFloat(j * id + c)
                                                    - max);
                    sum += e;
                    acc += e * keys.getFloat((first + j) * id + c);
                }
                poolKeys.setFloat(p * id + c, acc / sum);
            }
        }

        int complete = (position + 1) / pool;
        int budget = config.indexerPoolBudget();
        int count = 0;
        int[] selected = state.selected;
        if (complete <= budget) {
            for (int t = 0; t < complete * pool; t++) {
                selected[count++] = t;
            }
        } else {
            CpuOperations.matVec(
                    w.indexQ()[l],
                    state.queryLatent,
                    state.indexQuery,
                    ih * id,
                    config.qLoraRank());
            CpuOperations.matVec(w.indexWeights()[l], x, state.indexHeadWeights, ih, dim);
            final float headScale = (float) (1.0 / Math.sqrt(ih));
            final float keyScale = (float) (1.0 / Math.sqrt(id));
            float[] scores = state.poolScores;
            FloatTensor iq = state.indexQuery;
            FloatTensor hw = state.indexHeadWeights;
            Parallel.parallelFor(
                    0,
                    complete,
                    p -> {
                        float score = 0f;
                        for (int h = 0; h < ih; h++) {
                            float s = iq.dot(h * id, poolKeys, p * id, id) * keyScale;
                            score += hw.getFloat(h) * headScale * Math.max(s, 0f);
                        }
                        scores[p] = score;
                    });
            boolean[] chosen = topPools(scores, complete, budget);
            for (int p = 0; p < complete; p++) {
                if (chosen[p]) {
                    for (int j = 0; j < pool; j++) {
                        selected[count++] = p * pool + j;
                    }
                }
            }
        }
        if (config.indexerSelectTail()) {
            for (int t = complete * pool; t <= position; t++) {
                selected[count++] = t;
            }
        }
        state.selectedCount = count;
    }

    /** Marks the {@code budget} highest scores; ties go to the lower pool. */
    private static boolean[] topPools(float[] scores, int count, int budget) {
        Integer[] order = new Integer[count];
        for (int p = 0; p < count; p++) {
            order[p] = p;
        }
        java.util.Arrays.sort(
                order,
                (a, b) -> {
                    int c = Float.compare(scores[b], scores[a]);
                    return c != 0 ? c : Integer.compare(a, b);
                });
        boolean[] chosen = new boolean[count];
        for (int i = 0; i < budget; i++) {
            chosen[order[i]] = true;
        }
        return chosen;
    }

    private static void layerNorm(FloatTensor x, int size, FloatTensor weight, FloatTensor bias) {
        float mean = x.reduce(0, size, 0f, (acc, e) -> acc + e) / size;
        float var = 0f;
        for (int i = 0; i < size; i++) {
            float d = x.getFloat(i) - mean;
            var += d * d;
        }
        float inv = (float) (1.0 / Math.sqrt(var / size + Glm5NextConfiguration.INDEXER_NORM_EPS));
        for (int i = 0; i < size; i++) {
            x.setFloat(i, (x.getFloat(i) - mean) * inv * weight.getFloat(i) + bias.getFloat(i));
        }
    }

    // ── Feed-forward ──────────────────────────────────────────────────────────

    /** The dense SwiGLU or the mixture of experts, leaving its output in {@code xb2}. */
    static void feedForward(
            Glm5NextConfiguration config,
            Glm5NextLayerWeights.FeedForward<FloatTensor> w,
            Glm5NextState state,
            int l) {
        final int dim = config.dim();
        final float limit = config.swigluLimit();
        if (!config.isMoe(l)) {
            int hidden = config.hiddenDim();
            CpuOperations.matVec(w.gate()[l], state.xb, state.hb, hidden, dim);
            CpuOperations.matVec(w.up()[l], state.xb, state.hb2, hidden, dim);
            clampedSwiGLU(state.hb, state.hb2, hidden, limit);
            CpuOperations.matVec(w.down()[l], state.hb, state.xb2, dim, hidden);
            return;
        }
        mixtureOfExperts(config, w, state, l);
    }

    // @formatter:off
    /**
     * Sigmoid routing with a selection-only bias, as DeepSeek V3: the top {@code expertsUsed} of
     * {@code sigmoid(logit) + bias}, weighted by their unbiased sigmoids normalized to sum to one
     * and scaled; plus the ungated shared expert.
     */
    // @formatter:on
    private static void mixtureOfExperts(
            Glm5NextConfiguration config,
            Glm5NextLayerWeights.FeedForward<FloatTensor> w,
            Glm5NextState state,
            int l) {
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final float limit = config.swigluLimit();
        FloatTensor probs = state.routerScores;
        FloatTensor select = state.selectionScores;

        w.router()[l].matmul(state.xb, probs, experts, dim);
        FloatTensor bias = w.routerBias()[l];
        for (int e = 0; e < experts; e++) {
            float p = sigmoid(probs.getFloat(e));
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
        float norm = config.expertWeightsNorm() ? total + 1e-20f : 1f;
        for (int k = 0; k < used; k++) {
            state.expertWeights[k] = state.expertWeights[k] / norm * config.expertWeightsScale();
        }

        state.xb2.fillInPlace(0, dim, 0f);
        int chunk = config.expertChunkSize();
        for (int k = 0; k < used; k++) {
            int e = state.expertIds[k];
            int local = e % chunk;
            FloatTensor gate = chunkOf(w.gateExperts(), l, e / chunk);
            FloatTensor up = chunkOf(w.upExperts(), l, e / chunk);
            FloatTensor down = chunkOf(w.downExperts(), l, e / chunk);
            rows(gate, local * hidden * dim, state.xb, state.expertHidden, hidden, dim);
            rows(up, local * hidden * dim, state.xb, state.expertHiddenUp, hidden, dim);
            clampedSwiGLU(state.expertHidden, state.expertHiddenUp, hidden, limit);
            rows(down, local * dim * hidden, state.expertHidden, state.expertOut, dim, hidden);
            CpuOperations.weightedAccumulate(
                    state.xb2, state.expertOut, state.expertWeights[k], dim);
        }

        int shared = config.sharedHiddenDim();
        if (shared > 0) {
            CpuOperations.matVec(w.sharedGate()[l], state.xb, state.expertHidden, shared, dim);
            CpuOperations.matVec(w.sharedUp()[l], state.xb, state.expertHiddenUp, shared, dim);
            clampedSwiGLU(state.expertHidden, state.expertHiddenUp, shared, limit);
            CpuOperations.matVec(
                    w.sharedDown()[l], state.expertHidden, state.expertOut, dim, shared);
            CpuOperations.weightedAccumulate(state.xb2, state.expertOut, 1f, dim);
        }
    }

    private static FloatTensor chunkOf(List<List<FloatTensor>> stacks, int l, int chunk) {
        return stacks.get(l).get(chunk);
    }

    /** {@code out[r] = W[base + r * cols ...] . in} for {@code rows} rows of one expert. */
    private static void rows(
            FloatTensor weight, int base, FloatTensor in, FloatTensor out, int rows, int cols) {
        Parallel.parallelFor(
                0, rows, r -> out.setFloat(r, weight.dot(base + r * cols, in, 0, cols)));
    }

    /** {@code gate = silu(min(gate, limit)) * clamp(up, -limit, limit)}. */
    private static void clampedSwiGLU(FloatTensor gate, FloatTensor up, int size, float limit) {
        for (int i = 0; i < size; i++) {
            float g = Math.min(gate.getFloat(i), limit);
            float u = Math.max(-limit, Math.min(up.getFloat(i), limit));
            gate.setFloat(i, silu(g) * u);
        }
    }

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }
}
