package org.beehive.jitllm.inference.op;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host's implementations of the operation vocabulary.
 *
 * <p>One method per {@code program.op} operation, over plain tensors. This is the CPU side of
 * "unifying the vocabulary does not mean unifying the implementations": the names come from {@code
 * program.op}, the arithmetic is the arithmetic that was already here.
 *
 * <h2>These bodies were moved, not rewritten</h2>
 *
 * <p>Inside {@link FloatTensor#matmul}, which dispatches to the subclass for the representation the
 * weights are stored in — {@code Q4_KFloatTensor} and friends decode super-blocks in their dot
 * product. There is no decode step before the multiply and no decode operation to place there That
 * is why {@code OperationSupport} lists the K-quants for the host and not for the device.
 *
 * <p>Static methods over explicit tensors, deliberately: an operation implementation that took a
 * {@code State} would be reaching for a session's storage layout, and the whole point of naming
 * these is that they do not know whose buffers they are given.
 */
public final class CpuOperations {

    private CpuOperations() {}

    /** {@code EmbeddingLookup} — copy one token's row out of the embedding table. */
    public static void embeddingLookup(FloatTensor table, int token, FloatTensor out, int dim) {
        table.copyTo(token * dim, out, 0, dim);
    }

    /**
     * {@code EmbeddingLookup} over a table too large for the int-indexed tensor API.
     *
     * <p>Gemma4's per-layer embedding table is roughly 2.35 billion elements, so it is addressed
     * one row at a time through {@code LongIndexedTensor}, whose element index is a {@code long}.
     * The table is <b>never copied or materialized</b>: one row is read per token.
     */
    public static void embeddingLookupLongIndexed(
            LongIndexedTensor table, long rowIndex, int rowSize, FloatTensor out, int outOffset) {
        long rowStart = rowIndex * rowSize;
        for (int i = 0; i < rowSize; i++) {
            out.setFloat(outOffset + i, table.valueAt(rowStart + i));
        }
    }

    /**
     * {@code RmsNorm} — normalize by the root mean square and scale.
     *
     * <p>The epsilon is added to the mean square before the reciprocal square root, which is where
     * it has to be: applying it afterwards changes the result for small activations.
     */
    public static void rmsNorm(
            FloatTensor out,
            FloatTensor x,
            FloatTensor weight,
            int offset,
            int size,
            float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(
                offset,
                size,
                (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    /**
     * {@code MatVec} — a weight matrix times one activation vector.
     *
     * <p>The dispatch on the weight's representation happens inside {@link FloatTensor#matmul},
     * which is where it belongs: a quantized subclass decodes its blocks in the dot product.
     */
    public static void matVec(
            FloatTensor weight, FloatTensor in, FloatTensor out, int rows, int columns) {
        weight.matmul(in, out, rows, columns);
    }

    /**
     * {@code RoPE} — rotate the query and key projections in place, interleaved pairs.
     *
     * <p>Reads the precomputed sine and cosine tables rather than computing angles inline, which is
     * what {@code RopeFrequencies} builds at load. Keys are rotated only where they exist: {@code
     * rotn} is 2 below {@code kvDim} and 1 above it, because grouped query attention has fewer key
     * dimensions than query dimensions.
     */
    public static void rope(
            FloatTensor query,
            FloatTensor key,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            int position,
            int dim,
            int kvDim,
            int headSize) {
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % headSize;
            float fcr = freqCisReal.getFloat(position * (headSize / 2) + (head_dim / 2));
            float fci = freqCisImag.getFloat(position * (headSize / 2) + (head_dim / 2));
            int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                FloatTensor vec = v == 0 ? query : key; // the vector to rotate (query or key)
                float v0 = vec.getFloat(i);
                float v1 = vec.getFloat(i + 1);
                vec.setFloat(i, v0 * fcr - v1 * fci);
                vec.setFloat(i + 1, v0 * fci + v1 * fcr);
            }
        }
    }

    /** {@code KvAppend} — write this step's key and value into the retained store. */

    /**
     * {@code Attention} — scaled dot-product attention over the retained keys and values.
     *
     * <p>Attends over positions {@code 0.position} inclusive of whichever store it is given. That
     * store is <b>not necessarily this layer's</b>: a Gemma4 reuse layer passes an earlier layer's,
     * read-only.
     *
     * <p>The geometry comes in as an {@link AttentionShape} rather than as a head size, because the
     * families disagree about what a head size is — Qwen3 has separate key and value lengths that
     * need not equal {@code dim / heads}. Llama is {@link AttentionShape#uniform} of that geometry,
     * and produces bit-identical results to the head-size form it replaced.
     *
     * <p>Softmax is inside the head loop for the same reason: on the host it normalizes one head's
     * scores between the score pass and the weighted sum, and pulling it out would mean either an
     * extra pass over the scores or an operation that runs per head from outside. The vocabulary
     * still names {@code Softmax} because the CPU path is not the only path.
     */
    public static void appendKeyValue(
            FloatTensor key,
            FloatTensor value,
            FloatTensor keyCache,
            FloatTensor valueCache,
            int position,
            int kvDim) {
        // save key,value at this time step (position) to our kv cache
        key.copyTo(0, keyCache, position * kvDim, kvDim);
        value.copyTo(0, valueCache, position * kvDim, kvDim);
    }

    public static void attention(
            FloatTensor query,
            FloatTensor keyCache,
            FloatTensor valueCache,
            FloatTensor scores,
            FloatTensor out,
            int position,
            AttentionShape shape) {
        int kvDim = shape.kvDim();
        int kvMul = shape.kvMul();
        int contextLength = shape.contextLength();
        int queryHeadStride = shape.queryHeadStride();
        int keyHeadStride = shape.keyHeadStride();
        int keyDotLength = shape.keyDotLength();
        int valueHeadDim = shape.valueHeadDim();

        // multihead attention. iterate over all heads
        Parallel.parallelFor(
                0,
                shape.heads(),
                h -> {
                    // get the query vector for this head
                    int qOffset = h * queryHeadStride;

                    // attention scores for this head
                    int attOffset = h * contextLength;

                    // iterate over all timesteps in scope, including the current one
                    int windowStart = shape.windowStart(position);
                    for (int t = windowStart; t <= position; t++) {
                        // get the key vector for this head and at this timestep
                        int keyCacheOffset = t * kvDim + (h / kvMul) * keyHeadStride;
                        // calculate the attention score as the dot product of q and k
                        float score = query.dot(qOffset, keyCache, keyCacheOffset, keyDotLength);
                        score = shape.scaleScore(score);
                        // save the score to the attention buffer
                        scores.setFloat(attOffset + t, score);
                    }

                    // softmax the scores over the attended span, inclusive of the current position
                    scores.softmaxInPlace(attOffset + windowStart, position - windowStart + 1);

                    // weighted sum of the values, store back into xb
                    int xbOffset = h * valueHeadDim;
                    out.fillInPlace(xbOffset, valueHeadDim, 0f);

                    for (int t = windowStart; t <= position; t++) {
                        // get the value vector for this head and at this timestep
                        int vOffset = t * kvDim + (h / kvMul) * valueHeadDim;
                        // get the attention weight for this timestep
                        float a = scores.getFloat(attOffset + t);
                        // accumulate the weighted value into xb
                        out.saxpyInPlace(xbOffset, valueCache, vOffset, valueHeadDim, a);
                    }
                });
    }

    /**
     * {@code RoPE}, GPT-NeoX layout — a component is paired with the one half a head away.
     *
     * <p>Qwen2, Qwen2-MoE and Qwen3. Structurally the same rotation as {@link #rope}: the frequency
     * tables are read identically and the complex multiply is the same, only the two components
     * that form a pair differ ({@code RopeLayout}). Keys are rotated for the first {@code
     * keyValueHeads} heads only, which is how grouped query attention shows up here.
     *
     * @param headDimension the head width these projections are laid out with
     */
    public static void ropeNeox(
            FloatTensor query,
            FloatTensor key,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            int position,
            int heads,
            int keyValueHeads,
            int headDimension) {
        for (int h = 0; h < heads; ++h) {
            int rotn = h < keyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            int poffset = h * headDimension;
            int halfHead = headDimension / 2;
            for (int ic = 0; ic < halfHead; ic++) {
                float fcr = freqCisReal.getFloat(position * halfHead + ic);
                float fci = freqCisImag.getFloat(position * halfHead + ic);
                for (int vi = 0; vi < rotn; vi++) {
                    FloatTensor vec =
                            (vi == 0) ? query : key; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(poffset + ic);
                    float v1 = vec.getFloat(poffset + ic + halfHead);
                    vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                    vec.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
                }
            }
        }
    }

    /**
     * {@code BiasAdd} — add a learned bias vector to a projection, in place.
     *
     * <p>The same {@code addInPlace} as {@link #residualAdd}, and named separately for the reason
     * {@code BiasAdd} exists: the right-hand side is a model weight, not a branch result.
     */
    public static void biasAdd(FloatTensor projection, FloatTensor bias) {
        projection.addInPlace(bias);
    }

    /** {@code SwiGLU} — {@code silu(gate) * up}, in place on {@code gate}. */
    public static void swiGLU(FloatTensor gate, FloatTensor up) {
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        gate.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
        // elementwise multiply with w3(x)
        gate.multiplyInPlace(up);
    }

    /**
     * {@code RmsNorm}'s unweighted form — normalize with no learned scale.
     *
     * <p>Gemma4 normalizes its values this way. Not the weighted form with an all-ones weight: that
     * would perform one multiplication per element the model does not perform.
     */
    public static void rmsNormUnweighted(
            FloatTensor out, FloatTensor x, int offset, int size, float rmsNormEps) {
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(offset, size, (value, index) -> finalss * x.getFloat(index));
    }

    /** Tanh-approximation GELU, matching ggml's {@code ggml_gelu_f32}. */
    public static float gelu(float x) {
        return 0.5f
                * x
                * (1.0f + (float) Math.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
    }

    /**
     * {@code RoPE}, NeoX layout, rotating only the first {@code rotaryDim} of each head.
     *
     * <p>Partial rotation is a parameter of RoPE, not a second scheme: the head's first {@code
     * rotaryDim} elements rotate in pairs {@code (ic, ic + rotaryDim/2)} and everything above
     * {@code rotaryDim} passes through untouched. Qwen3.5 states {@code rope.dimension_count = 64}
     * against a 256-wide head, so three quarters of every head is unrotated.
     *
     * <p>The frequency tables must have been precomputed over {@code rotaryDim}, not over the head
     * width — they are indexed {@code position * rotaryDim / 2 + ic}.
     *
     * @param vec the buffer to rotate in place, {@code heads * headDim} long
     * @param heads how many heads it holds
     * @param headDim the head width
     * @param rotaryDim how much of each head rotates; must be even and at most {@code headDim}
     */
    public static void ropeNeoxPartial(
            FloatTensor vec,
            int heads,
            int headDim,
            int rotaryDim,
            int position,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag) {
        final int half = rotaryDim / 2;
        for (int h = 0; h < heads; h++) {
            final int base = h * headDim;
            for (int ic = 0; ic < half; ic++) {
                float fcr = freqCisReal.getFloat(position * half + ic);
                float fci = freqCisImag.getFloat(position * half + ic);
                float v0 = vec.getFloat(base + ic);
                float v1 = vec.getFloat(base + ic + half);
                vec.setFloat(base + ic, v0 * fcr - v1 * fci);
                vec.setFloat(base + ic + half, v0 * fci + v1 * fcr);
            }
        }
    }

    /** {@code RoPE}, NeoX layout, over one vector with its own head count. */
    public static void ropeNeoxSingle(
            FloatTensor vec,
            int nHeads,
            int headDim,
            int position,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag) {
        int nComplHead = headDim / 2;
        for (int h = 0; h < nHeads; h++) {
            int base = h * headDim;
            for (int ic = 0; ic < nComplHead; ic++) {
                float fcr = freqCisReal.getFloat(position * nComplHead + ic);
                float fci = freqCisImag.getFloat(position * nComplHead + ic);
                float v0 = vec.getFloat(base + ic);
                float v1 = vec.getFloat(base + ic + nComplHead);
                vec.setFloat(base + ic, v0 * fcr - v1 * fci);
                vec.setFloat(base + ic + nComplHead, v0 * fci + v1 * fcr);
            }
        }
    }

    /**
     * {@code GeGLU} — {@code gelu(gate) * up}, in place on {@code gate}.
     *
     * <p>{@link #swiGLU} with a different activation on the gate half. Two passes rather than one
     * fused loop, because that is what was here.
     */
    public static void geGLU(FloatTensor gate, FloatTensor up) {
        gate.mapInPlace(CpuOperations::gelu);
        gate.multiplyInPlace(up);
    }

    /**
     * {@code LogitSoftCap} — {@code cap * tanh(x / cap)} over the logits.
     *
     * <p>Part of the forward pass, so it runs in the same place whether sampling is on the host or
     * the device. A model without soft-capping does not call this with a neutral value; it does not
     * call it.
     */
    public static void logitSoftCap(FloatTensor logits, float cap) {
        final float invCap = 1.0f / cap;
        logits.mapInPlace(v -> (float) Math.tanh(v * invCap) * cap);
    }

    /** {@code ResidualAdd} — add the branch result back into the residual stream. */
    public static void residualAdd(FloatTensor stream, FloatTensor branch) {
        stream.addInPlace(branch);
    }

    /**
     * {@code Scale} — multiply every element by one scalar from the model configuration.
     *
     * <p>Granite's µP factors and Gemma's embedding scale. Naming it keeps the four places Granite
     * scales visible in its forward pass instead of hidden inside four other operations.
     */
    public static void scale(FloatTensor values, float factor) {
        values.mapInPlace(v -> v * factor);
    }

    /**
     * {@code SwiGLU} where the result is wanted in the <i>up</i> buffer rather than the gate.
     *
     * <p>Phi3's fused gate-up projection lands both halves in one buffer and then multiplies the up
     * half by the activated gate half, so its result is in the up half. Float multiplication is
     * commutative and exactly rounded, so this computes the same values as {@link #swiGLU} — what
     * differs is which buffer they end up in, and preserving that is what keeps the family
     * bit-identical without an extra copy.
     */
    public static void swiGLUIntoUp(FloatTensor gate, FloatTensor up) {
        gate.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
        up.multiplyInPlace(gate);
    }

    /**
     * {@code L2Norm} — scale one slice to unit length, in place.
     *
     * <p>Not {@link #rmsNormUnweighted}: that divides by the root <i>mean</i> square, a factor of
     * {@code √size} larger, and the two are not interchangeable. Qwen3.5's delta-net layers
     * normalize each convolved query and key head this way before the state update.
     *
     * @param values the buffer holding the slice
     * @param offset where the slice starts
     * @param size how many elements it spans
     * @param eps a floor on the divisor, so an all-zero slice yields zeros rather than NaN. A floor
     *     rather than a term added under the root: that is what {@code ggml_l2_norm} does, and the
     *     two differ by more than rounding once the norm approaches {@code eps}.
     */
    public static void l2Norm(FloatTensor values, int offset, int size, float eps) {
        float ss = values.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        final float inv = 1.0f / Math.max((float) Math.sqrt(ss), eps);
        values.mapWithIndexInPlace(offset, size, (value, index) -> inv * value);
    }

    /**
     * {@code CausalConv1d} — one step of a depthwise causal convolution, advancing its window.
     *
     * <p>Depthwise: each channel is convolved with its own taps and never mixes with another, so
     * this is {@code channels} independent dot products of length {@code kernel}, not a matrix
     * multiply. Causal: the taps run over this step and the {@code kernel - 1} before it, never
     * ahead.
     *
     * <p>Both the kernel and the window are <b>channel-major</b>: a channel's taps are contiguous.
     * That is how GGUF stores {@code ssm_conv1d} — its shape is {@code (kernel, channels)} with the
     * kernel as the fast axis, so element {@code c * kernel + t} is channel {@code c}'s tap {@code
     * t} — and matching it here means the weight is read as it lies. Taps run oldest first, so tap
     * {@code kernel - 1} is this step.
     *
     * <p>The window is state, and this operation owns advancing it. On return {@code window} holds
     * the last {@code kernel - 1} inputs including {@code input}, oldest first per channel, so the
     * next call continues the sequence. That is why the update is here and not in the caller: a
     * convolution whose window is advanced somewhere else is one refactor away from being advanced
     * twice.
     *
     * @param input this step's value per channel, {@code channels} long
     * @param weight the taps, {@code channels * kernel}, channel-major as GGUF stores them
     * @param window the retained inputs, {@code channels * (kernel - 1)}, channel-major
     * @param out the convolved result, {@code channels} long; may not alias {@code input}
     * @param channels how many independent channels
     * @param kernel how many taps, this step included
     */
    public static void causalConv1d(
            FloatTensor input,
            FloatTensor weight,
            FloatTensor window,
            FloatTensor out,
            int channels,
            int kernel) {
        final int history = kernel - 1;
        for (int c = 0; c < channels; c++) {
            final int wBase = c * kernel;
            final int hBase = c * history;
            final float x = input.getFloat(c);
            float sum = 0f;
            for (int t = 0; t < history; t++) {
                sum += weight.getFloat(wBase + t) * window.getFloat(hBase + t);
            }
            sum += weight.getFloat(wBase + history) * x;
            out.setFloat(c, sum);
            // Advance this channel's window: drop the oldest tap, append this step. Safe to do
            // here because a channel's taps have all been read by now.
            for (int t = 0; t + 1 < history; t++) {
                window.setFloat(hBase + t, window.getFloat(hBase + t + 1));
            }
            if (history > 0) {
                window.setFloat(hBase + history - 1, x);
            }
        }
    }

    /**
     * {@code DeltaRuleUpdate} — one step of the gated delta rule, for every value head.
     *
     * <p>The recurrent counterpart of {@link #attention}. Where attention scores a query against a
     * growing key/value store, this keeps one {@code stateDim × stateDim} matrix per head into
     * which the entire history has been summed, and updates it in place:
     *
     * <pre>
     *   S      *= decay[h]                      // forget, per head
     *   d       = (v[h] - Sᵀ·k[h]) * beta[h]    // what the state does not already predict
     *   S      += k[h] ⊗ d                      // write the correction
     *   out[h]  = Sᵀ·q[h]                       // read it back out
     * </pre>
     *
     * <p>The correction is the delta rule proper: the state is only moved by the part of the value
     * it does not already produce for this key, which is what keeps a fixed-size state from
     * saturating. {@code beta} is how much of that correction to apply.
     *
     * <p>Query and key heads may be fewer than value heads — this model has 16 of each against 48
     * value heads — so value head {@code h} reads key head {@code h % keyHeads}. <b>Modulo, not
     * division.</b> The reference repeats the key heads with a tiling repeat, which cycles {@code
     * 0,1,…,15,0,1,…} rather than blocking {@code 0,0,0,1,1,1,…}; the fused kernel states the same
     * mapping directly. Both orderings produce well-formed output, and the wrong one degrades
     * slowly with sequence length rather than failing outright.
     *
     * <p>State layout is {@code (h * stateDim + i) * stateDim + j}: row {@code i} indexes the key
     * dimension and column {@code j} the value dimension.
     *
     * @param q queries, {@code keyHeads * stateDim}, already normalized and scaled
     * @param k keys, {@code keyHeads * stateDim}, already normalized
     * @param v values, {@code valueHeads * stateDim}
     * @param decay per value head, already exponentiated
     * @param beta per value head, already through the logistic
     * @param state the retained matrices, updated in place
     * @param out the readout, {@code valueHeads * stateDim}
     * @param valueHeads how many value heads
     * @param keyHeads how many key heads there are to cycle through
     * @param stateDim the head width, equal for keys and values here
     */
    public static void deltaRuleUpdate(
            FloatTensor q,
            FloatTensor k,
            FloatTensor v,
            FloatTensor decay,
            FloatTensor beta,
            FloatTensor state,
            FloatTensor out,
            int valueHeads,
            int keyHeads,
            int stateDim) {
        final float[] correction = new float[stateDim];
        for (int h = 0; h < valueHeads; h++) {
            final int stateBase = h * stateDim * stateDim;
            final int kvBase = (h % keyHeads) * stateDim;
            final int vBase = h * stateDim;
            final float g = decay.getFloat(h);
            final float b = beta.getFloat(h);

            // Forget, and read what the state already predicts for this key, in one sweep: the
            // decayed state is what the prediction must be taken against.
            for (int j = 0; j < stateDim; j++) {
                correction[j] = 0f;
            }
            for (int i = 0; i < stateDim; i++) {
                final float ki = k.getFloat(kvBase + i);
                final int row = stateBase + i * stateDim;
                for (int j = 0; j < stateDim; j++) {
                    float decayed = state.getFloat(row + j) * g;
                    state.setFloat(row + j, decayed);
                    correction[j] += decayed * ki;
                }
            }

            // The delta: how far the value is from that prediction, scaled by beta.
            for (int j = 0; j < stateDim; j++) {
                correction[j] = (v.getFloat(vBase + j) - correction[j]) * b;
            }

            // Write the outer product, and read the state back out against the query in the same
            // sweep — the readout must see the updated state.
            for (int j = 0; j < stateDim; j++) {
                out.setFloat(vBase + j, 0f);
            }
            for (int i = 0; i < stateDim; i++) {
                final float ki = k.getFloat(kvBase + i);
                final float qi = q.getFloat(kvBase + i);
                final int row = stateBase + i * stateDim;
                for (int j = 0; j < stateDim; j++) {
                    float updated = state.getFloat(row + j) + ki * correction[j];
                    state.setFloat(row + j, updated);
                    out.setFloat(vBase + j, out.getFloat(vBase + j) + updated * qi);
                }
            }
        }
    }

    /**
     * {@code GatedNorm} — {@code rms_norm(x, weight) * silu(gate)}, per head.
     *
     * <p>Neither {@link #rmsNorm} nor {@link #swiGLU} alone: the normalization is per head over
     * {@code headDim} elements against a weight of that width shared by every head, and the gate is
     * applied to the normalized result rather than to a parallel projection.
     *
     * @param values the branch output, {@code heads * headDim}, normalized and gated in place
     * @param gate the {@code z} branch, the same shape, read through a SiLU
     * @param weight the learned scale, {@code headDim} long, shared across heads
     * @param heads how many heads
     * @param headDim the width normalization runs over
     */
    public static void gatedNorm(
            FloatTensor values,
            FloatTensor gate,
            FloatTensor weight,
            int heads,
            int headDim,
            float rmsNormEps) {
        for (int h = 0; h < heads; h++) {
            final int base = h * headDim;
            float ss = values.reduce(base, headDim, 0f, (acc, xi) -> acc + xi * xi);
            ss /= headDim;
            ss += rmsNormEps;
            final float inv = (float) (1.0 / Math.sqrt(ss));
            for (int i = 0; i < headDim; i++) {
                float z = gate.getFloat(base + i);
                float silu = z / (float) (1.0 + Math.exp(-z));
                values.setFloat(
                        base + i, weight.getFloat(i) * (inv * values.getFloat(base + i)) * silu);
            }
        }
    }

    /** The softplus {@code log(1 + exp(x))}, evaluated so a large argument does not overflow. */
    public static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }

    /**
     * {@code MoeRouter} — score every expert, then select the top {@code topK}.
     *
     * <p>Three semantics here are contract rather than implementation, because the selection order
     * becomes the accumulation order in {@link #weightedAccumulate} and floating-point addition is
     * not associative:
     *
     * <ul>
     *   <li>the softmax runs over <b>all</b> experts and precedes selection, and the selected
     *       weights are those probabilities <b>unrescaled</b>;
     *   <li>the scan is ascending with a strict {@code >}, so a <b>tie selects the lowest expert
     *       index</b>;
     *   <li>selection is destructive — a chosen expert's score is driven to negative infinity — so
     *       the {@code topK} results are distinct and come out in <b>descending weight order</b>.
     * </ul>
     */
    public static void moeRouter(
            FloatTensor input,
            FloatTensor routerWeight,
            FloatTensor scores,
            int[] selectedIds,
            float[] selectedWeights,
            int numberOfExperts,
            int topK,
            int dim) {
        routerWeight.matmul(input, scores, numberOfExperts, dim);
        scores.softmaxInPlace(0, numberOfExperts);

        for (int i = 0; i < topK; i++) {
            float best = Float.NEGATIVE_INFINITY;
            int index = -1;
            for (int j = 0; j < numberOfExperts; j++) {
                if (scores.getFloat(j) > best) {
                    best = scores.getFloat(j);
                    index = j;
                }
            }
            selectedIds[i] = index;
            selectedWeights[i] = best;
            scores.setFloat(index, Float.NEGATIVE_INFINITY);
        }
    }

    /**
     * {@code ExpertFeedForward} — one expert's gated feed-forward, indexed into stacked tensors.
     *
     * <p>The three weight tensors hold every expert; the index arithmetic is here rather than in a
     * sliced descriptor, which is what keeps the stacking layout out of the tensor vocabulary Same
     * gated shape as {@link #swiGLU}, over one expert's sub-matrices.
     */
    public static void expertFeedForward(
            FloatTensor input,
            int expert,
            FloatTensor gateWeights,
            FloatTensor upWeights,
            FloatTensor downWeights,
            FloatTensor hidden,
            FloatTensor hiddenUp,
            FloatTensor output,
            int expertHiddenDim,
            int dim) {
        int gateUpOffset = expert * expertHiddenDim * dim;
        int downOffset = expert * dim * expertHiddenDim;
        matmulExpert(gateWeights, gateUpOffset, input, hidden, expertHiddenDim, dim);
        matmulExpert(upWeights, gateUpOffset, input, hiddenUp, expertHiddenDim, dim);
        hidden.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
        hidden.multiplyInPlace(hiddenUp);
        matmulExpert(downWeights, downOffset, hidden, output, dim, expertHiddenDim);
    }

    /**
     * {@code WeightedAccumulate} — add a branch into the residual stream, scaled by one scalar.
     *
     * <p>In a mixture of experts this is the residual connection itself: there is no separate add
     * afterwards, so the order these calls are made in is the order the sum is formed.
     */
    public static void weightedAccumulate(
            FloatTensor stream, FloatTensor branch, float weight, int length) {
        stream.saxpyInPlace(0, branch, 0, length, weight);
    }

    /** The logistic sigmoid, for {@code WeightedAccumulate}'s {@code LOGISTIC} gate. */
    public static float logistic(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }

    /**
     * Like {@link FloatTensor#matmul}, but reads the weight matrix starting at element offset
     * {@code base} instead of 0 — so it can target ONE expert inside a stacked {@code [nExpert x d0
     * x d1]} tensor. {@code out[d0] = (d0 x d1 sub-matrix at base). in[d1]}.
     */
    static void matmulExpert(
            FloatTensor w, int base, FloatTensor in, FloatTensor out, int d0, int d1) {
        Parallel.parallelFor(0, d0, i -> out.setFloat(i, w.dot(base + i * d1, in, 0, d1)));
    }

    /**
     * Addressing, not arithmetic: split a fused QKV projection into its three parts.
     *
     * <p>Phi3 projects queries, keys and values with one weight matrix ({@code TensorRole}'s {@code
     * ATTENTION_QKV}) and then reads three slices out of the result. Those are copies with no
     * arithmetic in them, so they are not an operation — the same classification the key/value
     * append gets in {@link #attention}: a store into a program-fixed buffer is a binding-time
     * write, not work.
     */
    public static void splitFusedQkv(
            FloatTensor qkv,
            FloatTensor query,
            FloatTensor key,
            FloatTensor value,
            int queryWidth,
            int keyValueWidth) {
        qkv.copyTo(0, query, 0, queryWidth);
        qkv.copyTo(queryWidth, key, 0, keyValueWidth);
        qkv.copyTo(queryWidth + keyValueWidth, value, 0, keyValueWidth);
    }

    /**
     * {@code VocabProjection} — project the final hidden state onto the vocabulary.
     *
     * <p>The same arithmetic as {@link #matVec}, named separately because it is the operation a
     * phase skips: prefill runs every layer and stops before this one.
     */
    public static void vocabProjection(
            FloatTensor weight, FloatTensor in, FloatTensor logits, int vocabularySize, int dim) {
        weight.matmul(in, logits, vocabularySize, dim);
    }
}
