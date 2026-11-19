package org.beehive.jitllm.backend.cpu;

import java.util.List;
import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.inference.weights.standard.DeepSeek4StandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass of a {@code deepseek4} model (DeepSeek-V4), one token at a time.
 *
 * <p>It follows the reference implementation's decode path ({@code inference/model.py}) operation
 * for operation:
 *
 * <ol>
 *   <li>the token's embedding is copied into every residual stream;
 *   <li>each layer runs attention then the mixture of experts, each wrapped in a hyper-connection,
 *       as GLM-5.3 does;
 *   <li>the streams are weighed into one by the learned mHC head, normalized, and projected to the
 *       vocabulary.
 * </ol>
 *
 * <p>The reference rounds some activations to FP8 or FP4 to match its quantization-aware training;
 * this pass does not, as llama.cpp and transformers do not. It also leaves out the Hadamard
 * rotation of the indexer's queries and keys: applied to both sides, it does not change their dot
 * products.
 */
public final class DeepSeek4Forward {

    private DeepSeek4Forward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        return forward(
                (DeepSeek4Configuration) model.configuration(),
                (DeepSeek4StandardWeights) model.weights(),
                (DeepSeek4State) state,
                token,
                position);
    }

    public static FloatTensor forward(
            DeepSeek4Configuration config,
            DeepSeek4StandardWeights weights,
            DeepSeek4State state,
            int token,
            int position) {
        embed(config, weights, state, token);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            layer(config, weights, state, l, token, position);
        }
        return head(config, weights, state);
    }

    /** The token's embedding, copied into every residual stream. */
    static void embed(
            DeepSeek4Configuration config,
            DeepSeek4StandardWeights weights,
            DeepSeek4State state,
            int token) {
        final int dim = config.dim();
        CpuOperations.embeddingLookup(weights.tokenEmbeddingTable, token, state.x, dim);
        for (int s = 0; s < config.hcCount(); s++) {
            state.x.copyTo(0, state.streams, s * dim, dim);
        }
    }

    /** One layer: attention, then the mixture of experts, each inside a hyper-connection. */
    static void layer(
            DeepSeek4Configuration config,
            DeepSeek4StandardWeights weights,
            DeepSeek4State state,
            int l,
            int token,
            int position) {
        final int dim = config.dim();
        final float eps = config.rmsNormEps();
        DeepSeek4LayerWeights<FloatTensor> w = weights.layers;
        DeepSeek4LayerWeights.Mhc<FloatTensor> mhc = w.mhc();

        hyperConnectionPre(config, state, mhc.attnFn()[l], mhc.attnBase()[l], mhc.attnScale()[l]);
        CpuOperations.rmsNorm(state.xb, state.x, w.attnNorm()[l], 0, dim, eps);
        attention(config, w, state, l, position);
        hyperConnectionPost(config, state);

        hyperConnectionPre(config, state, mhc.ffnFn()[l], mhc.ffnBase()[l], mhc.ffnScale()[l]);
        CpuOperations.rmsNorm(state.xb, state.x, w.ffnNorm()[l], 0, dim, eps);
        mixtureOfExperts(config, w.ffn(), state, l, token);
        hyperConnectionPost(config, state);
    }

    // @formatter:off
    /**
     * The mHC head: the streams weighed by {@code sigmoid(scale * fn . rmsnorm(streams) + base) +
     * eps} into one, normalized, then the vocabulary projection.
     */
    // @formatter:on
    static FloatTensor head(
            DeepSeek4Configuration config, DeepSeek4StandardWeights weights, DeepSeek4State state) {
        final int dim = config.dim();
        final int n = config.hcCount();
        final int width = n * dim;
        FloatTensor streams = state.streams;
        float ss = streams.reduce(0, width, 0f, (acc, v) -> acc + v * v);
        final float inv = (float) (1.0 / Math.sqrt(ss / width + config.rmsNormEps()));
        float scale = weights.hcHeadScale.getFloat(0);
        state.x.fillInPlace(0, dim, 0f);
        for (int s = 0; s < n; s++) {
            float mix = inv * weights.hcHeadFn.dot(s * width, streams, 0, width);
            float pre = sigmoid(mix * scale + weights.hcHeadBase.getFloat(s)) + config.hcEps();
            state.x.saxpyInPlace(0, streams, s * dim, dim, pre);
        }
        CpuOperations.rmsNorm(state.xb, state.x, weights.outputNorm, 0, dim, config.rmsNormEps());
        CpuOperations.vocabProjection(
                weights.output, state.xb, state.logits, config.vocabularySize(), dim);
        return state.logits;
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
            DeepSeek4Configuration config,
            DeepSeek4State state,
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
                    DeepSeek4Configuration.HC_POST_SCALE
                            * sigmoid(mixes[n + i] * s1 + base.getFloat(n + i));
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
    static void hyperConnectionPost(DeepSeek4Configuration config, DeepSeek4State state) {
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

    // ── Attention ─────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Attention for one token, leaving the result in {@code xb2}.
     *
     * <ol>
     *   <li>{@code q = q_b(norm(q_a x))}, each head RMS-normalized without a weight; {@code kv =
     *       norm(kv x)}, stored in the window ring; both rotated on their tails;
     *   <li>with a compression ratio, the compressor takes the position, and completes a block
     *       every ratio positions ({@link #compress}); a ratio-4 layer's indexer picks blocks;
     *   <li>each head attends to the window and the visible blocks, with its sink in the softmax;
     *       {@code kv} is both key and value; the output's tail is rotated back;
     *   <li>{@code output_a} per group of heads, then {@code output_b}.
     * </ol>
     */
    // @formatter:on
    static void attention(
            DeepSeek4Configuration config,
            DeepSeek4LayerWeights<FloatTensor> weights,
            DeepSeek4State state,
            int l,
            int position) {
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int hd = config.headDim();
        final int nope = config.nopeDim();
        final int window = config.slidingWindow();
        final int ratio = config.ratio(l);
        final float eps = config.rmsNormEps();
        DeepSeek4LayerWeights.Attention<FloatTensor> w = weights.attention();
        float[] freqs = ratio == 0 ? state.ropeFrequencies : state.compressRopeFrequencies;
        FloatTensor x = state.xb;

        CpuOperations.matVec(w.qA()[l], x, state.queryLatent, config.qLoraRank(), dim);
        CpuOperations.rmsNorm(
                state.queryLatent, state.queryLatent, w.qANorm()[l], 0, config.qLoraRank(), eps);
        CpuOperations.matVec(
                w.qB()[l], state.queryLatent, state.q, config.queryDim(), config.qLoraRank());
        FloatTensor q = state.q;
        for (int h = 0; h < heads; h++) {
            int base = h * hd;
            float ss = q.reduce(base, hd, 0f, (acc, e) -> acc + e * e);
            float inv = (float) (1.0 / Math.sqrt(ss / hd + eps));
            for (int i = 0; i < hd; i++) {
                q.setFloat(base + i, q.getFloat(base + i) * inv);
            }
            rotate(q, base + nope, freqs, position, false);
        }
        CpuOperations.matVec(w.kv()[l], x, state.keyValue, hd, dim);
        CpuOperations.rmsNorm(state.keyValue, state.keyValue, w.kvNorm()[l], 0, hd, eps);
        rotate(state.keyValue, nope, freqs, position, false);
        FloatTensor ring = state.keyCache[l];
        state.keyValue.copyTo(0, ring, (position % window) * hd, hd);

        int blocks = 0;
        if (ratio != 0) {
            DeepSeek4LayerWeights.Compressor<FloatTensor> c = weights.compressor();
            compress(
                    config,
                    state,
                    c.kv()[l],
                    c.gate()[l],
                    c.ape()[l],
                    c.norm()[l],
                    state.compressorKv[l],
                    state.compressorScore[l],
                    state.compressedKeys[l],
                    ratio,
                    hd,
                    position);
            blocks =
                    config.hasIndexer(l)
                            ? indexerSelect(config, weights, state, l, position)
                            : allBlocks(state, (position + 1) / ratio);
        }

        int first = Math.max(0, position - window + 1);
        int positions = position - first + 1;
        for (int j = 0; j < positions; j++) {
            state.attendedPositions[j] = (first + j) % window;
        }
        final int windowCount = positions;
        final int blockCount = blocks;
        final float scale = config.attentionScale();
        final int stride = config.maxAttended();
        FloatTensor keys = state.compressedKeys[l];
        FloatTensor sinks = w.sinks()[l];
        FloatTensor att = state.att;
        FloatTensor out = state.attentionOut;
        Parallel.parallelFor(
                0,
                heads,
                h -> {
                    int scores = h * stride;
                    int qBase = h * hd;
                    float max = sinks.getFloat(h);
                    for (int j = 0; j < windowCount; j++) {
                        float s = q.dot(qBase, ring, state.attendedPositions[j] * hd, hd) * scale;
                        att.setFloat(scores + j, s);
                        max = Math.max(max, s);
                    }
                    for (int j = 0; j < blockCount; j++) {
                        float s = q.dot(qBase, keys, state.attendedBlocks[j] * hd, hd) * scale;
                        att.setFloat(scores + windowCount + j, s);
                        max = Math.max(max, s);
                    }
                    float sum = (float) Math.exp(sinks.getFloat(h) - max);
                    for (int j = 0; j < windowCount + blockCount; j++) {
                        float e = (float) Math.exp(att.getFloat(scores + j) - max);
                        att.setFloat(scores + j, e);
                        sum += e;
                    }
                    out.fillInPlace(qBase, hd, 0f);
                    for (int j = 0; j < windowCount; j++) {
                        out.saxpyInPlace(
                                qBase,
                                ring,
                                state.attendedPositions[j] * hd,
                                hd,
                                att.getFloat(scores + j) / sum);
                    }
                    for (int j = 0; j < blockCount; j++) {
                        out.saxpyInPlace(
                                qBase,
                                keys,
                                state.attendedBlocks[j] * hd,
                                hd,
                                att.getFloat(scores + windowCount + j) / sum);
                    }
                });
        for (int h = 0; h < heads; h++) {
            rotate(out, h * hd + nope, freqs, position, true);
        }

        final int groupIn = config.groupInputDim();
        final int rank = config.outputLoraRank();
        FloatTensor outputA = w.outputA()[l];
        FloatTensor grouped = state.groupedOut;
        Parallel.parallelFor(
                0,
                config.groupedOutputDim(),
                r ->
                        grouped.setFloat(
                                r, outputA.dot(r * groupIn, out, (r / rank) * groupIn, groupIn)));
        CpuOperations.matVec(w.outputB()[l], grouped, state.xb2, dim, config.groupedOutputDim());
    }

    /** Lists blocks {@code 0..count-1} as attended; returns {@code count}. */
    private static int allBlocks(DeepSeek4State state, int count) {
        for (int b = 0; b < count; b++) {
            state.attendedBlocks[b] = b;
        }
        return count;
    }

    // @formatter:off
    /**
     * Feeds one position to a compressor of ratio {@code r} and, when it completes a block, pools
     * the block into {@code cache[position / r]}.
     *
     * <ul>
     *   <li>the position's projection {@code kv x} and gate score {@code gate x + ape[position %
     *       r]} are stored as a pending entry;
     *   <li>a block's key is, per channel, the softmax of the scores over its entries weighing the
     *       projections, then RMS-normalized and rotated at the block's first position;
     *   <li>for the overlapping ratio 4, the projections are twice as wide: the first half of each
     *       entry pools into the next block. A block's softmax runs over the previous block's first
     *       halves and its own second halves, and its entries become the previous block's after.
     * </ul>
     */
    // @formatter:on
    static void compress(
            DeepSeek4Configuration config,
            DeepSeek4State state,
            FloatTensor kvWeight,
            FloatTensor gateWeight,
            FloatTensor ape,
            FloatTensor norm,
            FloatTensor pendingKv,
            FloatTensor pendingScore,
            FloatTensor cache,
            int r,
            int width,
            int position) {
        final int dim = config.dim();
        final int o = DeepSeek4Configuration.overlap(r);
        final int row = o * width;
        final int slot = position % r;
        FloatTensor kv = state.compressorProjection;
        FloatTensor gate = state.compressorGate;
        CpuOperations.matVec(kvWeight, state.xb, kv, row, dim);
        CpuOperations.matVec(gateWeight, state.xb, gate, row, dim);
        int entry = (o == 2 ? r + slot : slot) * row;
        for (int i = 0; i < row; i++) {
            pendingKv.setFloat(entry + i, kv.getFloat(i));
            pendingScore.setFloat(entry + i, gate.getFloat(i) + ape.getFloat(slot * row + i));
        }
        if ((position + 1) % r != 0) {
            return;
        }
        FloatTensor pooled = state.pooled;
        for (int c = 0; c < width; c++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < o * r; j++) {
                max = Math.max(max, pendingScore.getFloat(pendingIndex(j, c, r, o, width)));
            }
            float sum = 0f;
            float acc = 0f;
            for (int j = 0; j < o * r; j++) {
                int at = pendingIndex(j, c, r, o, width);
                float e = (float) Math.exp(pendingScore.getFloat(at) - max);
                sum += e;
                acc += e * pendingKv.getFloat(at);
            }
            pooled.setFloat(c, acc / sum);
        }
        CpuOperations.rmsNorm(pooled, pooled, norm, 0, width, config.rmsNormEps());
        rotate(
                pooled,
                width - config.ropeDim(),
                state.compressRopeFrequencies,
                position + 1 - r,
                false);
        pooled.copyTo(0, cache, (position / r) * width, width);
        if (o == 2) {
            pendingKv.copyTo(r * row, pendingKv, 0, r * row);
            pendingScore.copyTo(r * row, pendingScore, 0, r * row);
        }
    }

    /**
     * Where entry {@code j} of a block's softmax holds channel {@code c}: for an overlapping
     * compressor, the previous block's first halves ({@code j < r}), then the block's own second
     * halves.
     */
    private static int pendingIndex(int j, int c, int r, int o, int width) {
        if (o == 1) {
            return j * width + c;
        }
        return j < r ? j * 2 * width + c : j * 2 * width + width + c;
    }

    // @formatter:off
    /**
     * The indexer of a ratio-4 layer: compresses the position into its own keys, then fills {@link
     * DeepSeek4State#attendedBlocks} with the complete blocks scoring highest, at most {@code
     * indexerTopK}. A block's score is {@code sum_h w_h * relu(q_h . key)}, with {@code q} from the
     * query latent, rotated, and {@code w = proj x / sqrt(headDim * heads)}. Returns the count.
     */
    // @formatter:on
    static int indexerSelect(
            DeepSeek4Configuration config,
            DeepSeek4LayerWeights<FloatTensor> weights,
            DeepSeek4State state,
            int l,
            int position) {
        final int dim = config.dim();
        final int ih = config.indexerHeads();
        final int id = config.indexerHeadDim();
        final int r = config.ratio(l);
        DeepSeek4LayerWeights.Indexer<FloatTensor> w = weights.indexer();
        compress(
                config,
                state,
                w.compressorKv()[l],
                w.compressorGate()[l],
                w.compressorApe()[l],
                w.compressorNorm()[l],
                state.indexerKv[l],
                state.indexerScore[l],
                state.indexerKeys[l],
                r,
                id,
                position);
        int complete = (position + 1) / r;
        int budget = Math.min(config.indexerTopK(), complete);
        if (budget == complete) {
            return allBlocks(state, complete);
        }
        CpuOperations.matVec(
                w.qB()[l], state.queryLatent, state.indexQuery, ih * id, config.qLoraRank());
        for (int h = 0; h < ih; h++) {
            rotate(
                    state.indexQuery,
                    h * id + id - config.ropeDim(),
                    state.compressRopeFrequencies,
                    position,
                    false);
        }
        CpuOperations.matVec(w.headWeights()[l], state.xb, state.indexHeadWeights, ih, dim);
        final float weightScale = (float) (1.0 / Math.sqrt((double) id * ih));
        float[] scores = state.blockScores;
        FloatTensor iq = state.indexQuery;
        FloatTensor hw = state.indexHeadWeights;
        FloatTensor keys = state.indexerKeys[l];
        Parallel.parallelFor(
                0,
                complete,
                b -> {
                    float score = 0f;
                    for (int h = 0; h < ih; h++) {
                        float s = iq.dot(h * id, keys, b * id, id);
                        score += hw.getFloat(h) * weightScale * Math.max(s, 0f);
                    }
                    scores[b] = score;
                });
        Integer[] order = new Integer[complete];
        for (int b = 0; b < complete; b++) {
            order[b] = b;
        }
        java.util.Arrays.sort(
                order,
                (a, b) -> {
                    int c = Float.compare(scores[b], scores[a]);
                    return c != 0 ? c : Integer.compare(a, b);
                });
        for (int i = 0; i < budget; i++) {
            state.attendedBlocks[i] = order[i];
        }
        return budget;
    }

    /**
     * Rotates the interleaved pairs {@code (2i, 2i + 1)} of the {@code ropeDim} values from {@code
     * offset} by {@code position * freqs[i]}, or back by the same angle when {@code inverse}.
     */
    static void rotate(FloatTensor x, int offset, float[] freqs, int position, boolean inverse) {
        for (int i = 0; i < freqs.length; i++) {
            float angle = position * freqs[i];
            float cos = (float) Math.cos(angle);
            float sin = inverse ? -(float) Math.sin(angle) : (float) Math.sin(angle);
            int a = offset + 2 * i;
            float x0 = x.getFloat(a);
            float x1 = x.getFloat(a + 1);
            x.setFloat(a, x0 * cos - x1 * sin);
            x.setFloat(a + 1, x0 * sin + x1 * cos);
        }
    }

    // ── Mixture of experts ────────────────────────────────────────────────────

    // @formatter:off
    /**
     * The routed experts plus the shared expert, leaving the sum in {@code xb2}.
     *
     * <ul>
     *   <li>each expert's score is {@code sqrt(softplus(router . x))};
     *   <li>a hash-routed layer takes its experts from the token's row of the table, any other the
     *       top {@code expertsUsed} of score plus bias;
     *   <li>the chosen experts are weighted by their scores, normalized to sum to one and scaled;
     *   <li>every SwiGLU, the shared expert's included, clamps its gate and up projections.
     * </ul>
     */
    // @formatter:on
    static void mixtureOfExperts(
            DeepSeek4Configuration config,
            DeepSeek4LayerWeights.FeedForward<FloatTensor> w,
            DeepSeek4State state,
            int l,
            int token) {
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final float limit = config.swigluLimit();
        FloatTensor scores = state.routerScores;
        FloatTensor select = state.selectionScores;

        w.router()[l].matmul(state.xb, scores, experts, dim);
        for (int e = 0; e < experts; e++) {
            scores.setFloat(e, (float) Math.sqrt(softplus(scores.getFloat(e))));
        }
        if (config.isHashRouted(l)) {
            int[] row = w.tokenExperts()[l];
            for (int k = 0; k < used; k++) {
                state.expertIds[k] = row[token * used + k];
            }
        } else {
            FloatTensor bias = w.routerBias()[l];
            for (int e = 0; e < experts; e++) {
                select.setFloat(e, scores.getFloat(e) + bias.getFloat(e));
            }
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
            }
        }
        float total = 0f;
        for (int k = 0; k < used; k++) {
            state.expertWeights[k] = scores.getFloat(state.expertIds[k]);
            total += state.expertWeights[k];
        }
        float norm = config.expertWeightsNorm() ? total : 1f;
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
        CpuOperations.matVec(w.sharedGate()[l], state.xb, state.expertHidden, shared, dim);
        CpuOperations.matVec(w.sharedUp()[l], state.xb, state.expertHiddenUp, shared, dim);
        clampedSwiGLU(state.expertHidden, state.expertHiddenUp, shared, limit);
        CpuOperations.matVec(w.sharedDown()[l], state.expertHidden, state.expertOut, dim, shared);
        CpuOperations.weightedAccumulate(state.xb2, state.expertOut, 1f, dim);
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

    private static double softplus(float x) {
        return x > 20f ? x : Math.log1p(Math.exp(x));
    }

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }
}
