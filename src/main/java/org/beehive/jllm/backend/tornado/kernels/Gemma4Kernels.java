package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Custom GPU kernels for the Gemma 4 architecture.
 *
 * <p>Gemma 4's computation graph differs substantially from the "Llama-like" models the rest of the
 * {@code tornadovm.kernels} package targets: every layer carries its own Q/K-norm and a "sandwich"
 * of pre/post normalization around both attention and FFN, attention alternates between
 * sliding-window (local) and full (global) variants -- with different head dimensions and RoPE
 * tables -- some layers reuse an earlier layer's KV cache, the FFN uses a GeGLU activation, and
 * every layer additionally mixes in a per-layer embedding (PLE). None of the existing fused kernels
 * match this shape, so this class provides purpose-built (but otherwise unfused/modular)
 * replacements; see {@link org.beehive.jllm.backend.cpu.InferenceCore#forwardJavaGemma4} for the
 * reference computation each of these mirrors.
 */
// @formatter:off
public class Gemma4Kernels {

    /**
     * Materializes {@code out = weight * (rmsScale[0] * x)} -- i.e. RMSNorm with a learned scale,
     * written to a separate buffer.
     */
    public static void applyRmsNorm(
            KernelContext context,
            FloatArray out,
            FloatArray x,
            FloatArray weight,
            FloatArray rmsScale,
            int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            float scale = rmsScale.get(0);
            out.set(gid, weight.get(gid) * (scale * x.get(gid)));
        }
    }

    /**
     * {@code x[i] *= scaleTensor[0]} -- like {@link TransformerComputeKernels#scaleInPlace}, but
     * the (learned, per-layer) scale is read from a 1-element tensor at kernel time.
     */
    public static void scaleInPlaceFromTensor(
            KernelContext context, FloatArray x, FloatArray scaleTensor, int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            x.set(gid, x.get(gid) * scaleTensor.get(0));
        }
    }

    /**
     * {@code out[i] = (a[i] + b[i]) * scale} (used to merge the per-layer projection with the
     * per-layer token embedding).
     */
    public static void addAndScale(
            KernelContext context,
            FloatArray out,
            FloatArray a,
            FloatArray b,
            float scale,
            int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            out.set(gid, (a.get(gid) + b.get(gid)) * scale);
        }
    }

    /**
     * Sandwich-norm + residual: {@code x[i] += weight[i] * (rmsScale[0] * delta[i])}. Used for
     * post-attention-norm, post-FFN-norm, and the per-layer-embedding post-norm, each of which
     * normalizes a freshly computed branch output and adds it back onto the running residual.
     */
    public static void rmsNormApplyWithResidual(
            KernelContext context,
            FloatArray x,
            FloatArray delta,
            FloatArray weight,
            FloatArray rmsScale,
            int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            float scale = rmsScale.get(0);
            float normalized = weight.get(gid) * (scale * delta.get(gid));
            x.set(gid, x.get(gid) + normalized);
        }
    }

    /**
     * Per-head RMSNorm with a learned scale (Q-norm / K-norm): each workgroup normalizes one head
     * of {@code vec} in place, mirroring {@code rmsnorm(vec, vec, weight, h*headDim, headDim, eps)}
     * applied independently for every head {@code h}.
     */
    public static void rmsNormPerHead(
            KernelContext context,
            FloatArray vec,
            FloatArray weight,
            int nHeads,
            int headDim,
            int localMemSize,
            float rmsNormEps) {
        int headIdx = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        if (headIdx >= nHeads) {
            return;
        }
        int base = headIdx * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = vec.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            float normalized = ss * vec.get(base + i);
            vec.set(base + i, weight.get(i) * normalized);
        }
    }

    /**
     * Like {@link #rmsNormPerHead}, but without a learned scale (Gemma4 normalizes V with a plain,
     * weight-less RMSNorm).
     */
    public static void rmsNormPerHeadNoWeight(
            KernelContext context,
            FloatArray vec,
            int nHeads,
            int headDim,
            int localMemSize,
            float rmsNormEps) {
        int headIdx = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        if (headIdx >= nHeads) {
            return;
        }
        int base = headIdx * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = vec.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            vec.set(base + i, ss * vec.get(base + i));
        }
    }

    /**
     * NeoX-style RoPE rotation (split-half pairs, using precomputed cos/sin tables) for Q only --
     * used by layers that reuse an earlier layer's KV cache (so K is never computed/rotated here).
     * Launched on a 2D grid of (nHeads, headDim/2).
     */
    public static void ropeNeoxRotateQOnly(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int headDim) {
        int h = context.globalIdx;
        int ic = context.globalIdy;
        int half = headDim / 2;
        int pos = positionHolder.get(0);

        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int base = h * headDim;
        float v0 = q.get(base + ic);
        float v1 = q.get(base + ic + half);
        q.set(base + ic, v0 * fcr - v1 * fci);
        q.set(base + ic + half, v0 * fci + v1 * fcr);
    }

    /**
     * NeoX-style RoPE rotation for Q and K, fused with the KV-cache write (K rotated then cached, V
     * copied as-is) -- used by layers that own their KV cache. Launched on a 2D grid of (nHeads,
     * headDim/2); K/V handling is gated on {@code h < nHeadKv} (mirrors {@code
     * Qwen3Kernels.ropeRotationWithCacheCopy}'s {@code rotn} pattern for GQA).
     *
     * <p>{@code cacheBaseOffset} is the (possibly shared, see {@link
     * org.beehive.jllm.inference.state.Gemma4State#cacheLayerBaseOffset}) base element offset of
     * this layer's slot in the flat {@code keyCache}/{@code valueCache} buffers.
     */
    public static void ropeNeoxRotateAndCacheCopy(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeadKv,
            int headDim,
            int kvDim,
            int cacheBaseOffset) {

        int h = context.globalIdx;
        int ic = context.globalIdy;
        int half = headDim / 2;
        int pos = positionHolder.get(0);

        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        // Rotate Q (all heads)
        int qBase = h * headDim;
        float v0q = q.get(qBase + ic);
        float v1q = q.get(qBase + ic + half);
        q.set(qBase + ic, v0q * fcr - v1q * fci);
        q.set(qBase + ic + half, v0q * fci + v1q * fcr);

        // Rotate K and write rotated-K / raw-V into the cache (KV heads only)
        if (h < nHeadKv) {
            int kBase = h * headDim;
            float v0k = k.get(kBase + ic);
            float v1k = k.get(kBase + ic + half);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;
            k.set(kBase + ic, rotatedK0);
            k.set(kBase + ic + half, rotatedK1);

            int cacheOffset = cacheBaseOffset + pos * kvDim + h * headDim;
            keyCache.set(cacheOffset + ic, rotatedK0);
            keyCache.set(cacheOffset + ic + half, rotatedK1);
            valueCache.set(cacheOffset + ic, v.get(kBase + ic));
            valueCache.set(cacheOffset + ic + half, v.get(kBase + ic + half));
        }
    }

    /**
     * Split-KV attention, phase 2: combine, one thread per output element.
     *
     * <p>A Gemma 4 kernel rather than a change to {@link
     * TransformerComputeKernelsLayered#combineSplitKVAttention}, which Qwen 3.5 also runs. That one
     * maps a workgroup to a head, which for this family is eight workgroups on a device with far
     * more multiprocessors: Nsight Compute measured it at 2.1% of DRAM peak, 0.2% of compute and
     * essentially zero occupancy.
     *
     * <p>The parallelism available is {@code nHeads * headDim} output elements, each independently
     * computable, so this maps one thread to each. The per-head scalars -- the global maximum and
     * the denominator -- are then recomputed by every thread of that head from the {@code nSplits}
     * maxima and sums rather than shared through local memory. That is a few dozen redundant reads
     * per thread, and what it buys is no local array, no barrier, and a grid that is no longer one
     * workgroup per head.
     *
     * <p><b>Bit-identical to the shared kernel by construction.</b> The maximum is taken over the
     * same values in the same order, the denominator accumulates the same products in the same
     * order, and each output element accumulates its splits in the same order. Only where the
     * intermediate values live changes, so parity is expected to be unchanged to the last digit and
     * that is the contract this is tested against.
     */
    public static void combineSplitKVAttentionPerElement(
            KernelContext context,
            FloatArray att,
            FloatArray xb,
            int nHeads,
            int headDim,
            int nSplits) {

        int gid = context.globalIdx;
        if (gid >= nHeads * headDim) {
            return;
        }
        int h = gid / headDim;
        int d = gid - h * headDim;

        // Must match the COMPACT layout attentionWithSlidingWindowSplit writes: per head,
        // nSplits numerators of headDim, then nSplits maxima, then nSplits sums.
        int headBase = h * nSplits * (headDim + 2);
        int mBase = headBase + nSplits * headDim;
        int lBase = mBase + nSplits;

        float gMax = Float.NEGATIVE_INFINITY;
        for (int s = 0; s < nSplits; s++) {
            float ms = att.get(mBase + s);
            if (ms > gMax) {
                gMax = ms;
            }
        }

        float denom = 0.0f;
        float acc = 0.0f;
        for (int s = 0; s < nSplits; s++) {
            float ms = att.get(mBase + s);
            float f = (ms == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(ms - gMax);
            denom += f * att.get(lBase + s);
            acc += f * att.get(headBase + s * headDim + d);
        }
        float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        xb.set(h * headDim + d, acc * inv);
    }

    /**
     * Sliding-window attention, phase 1 of two: one workgroup per (head, split of the window).
     *
     * <p>The workgroup-per-head kernel parallelises across {@code headDim} and across positions
     * <i>within</i> a workgroup, but every lane still walks the whole window in the weighted sum,
     * so its cost grows with context depth. Measured on this family: 0.0164 ms per call at an
     * average depth of about 12, and 0.2031 ms at about 551, while the depth-independent
     * projections stayed flat. This phase cuts the window into {@code nSplits} slices and gives
     * each its own workgroup, so the work per workgroup stops growing once the slices do.
     *
     * <p>Each split emits an unnormalised online-softmax state — the numerators, its own maximum
     * and its own sum of exponentials — in the COMPACT layout {@link
     * TransformerComputeKernelsLayered#combineSplitKVAttention} already reads: per head, {@code
     * nSplits} numerators of {@code headDim}, then {@code nSplits} maxima, then {@code nSplits}
     * sums. An empty slice writes {@code -inf} and zero, which that combine already treats as
     * contributing nothing.
     *
     * <p>Scores still go through {@code wrapAtt} at their absolute position, so the slices write
     * disjoint ranges of it and no extra scratch is needed for them.
     */
    public static void attentionWithSlidingWindowSplit(
            KernelContext context,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray wrapAtt,
            FloatArray attSplit,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int cacheBaseOffset,
            int windowSize,
            int contextLength,
            int nSplits,
            int localMemSize) {

        int tid = context.localIdx;
        int group = context.groupIdx;
        int localSize = context.localGroupSizeX;

        int h = group / nSplits;
        int split = group - h * nSplits;
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int windowStart = Math.max(0, pos - windowSize + 1);
        int hOff = h * contextLength;
        int kvHeadIdx = h / kvMul;
        int qOffset = h * headDim;

        int total = pos - windowStart + 1;
        int chunk = (total + nSplits - 1) / nSplits;
        int from = windowStart + split * chunk;
        int to = Math.min(pos, from + chunk - 1);

        int headBase = h * nSplits * (headDim + 2);
        int mBase = headBase + nSplits * headDim;
        int lBase = mBase + nSplits;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);

        // An empty slice still has to write its state, or the combine reads whatever was there.
        if (from > to) {
            for (int d = tid; d < headDim; d += localSize) {
                attSplit.set(headBase + split * headDim + d, 0.0f);
            }
            if (tid == 0) {
                attSplit.set(mBase + split, Float.NEGATIVE_INFINITY);
                attSplit.set(lBase + split, 0.0f);
            }
            return;
        }

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = q.get(qOffset + i);
        }
        context.localBarrier();

        for (int t = from + tid; t <= to; t += localSize) {
            int keyOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
            float score = 0.0f;
            for (int i = 0; i < headDim; i++) {
                score += qShared[i] * keyCache.get(keyOffset + i);
            }
            // Gemma4 attention scaling = 1.0 (no 1/sqrt(headDim))
            wrapAtt.set(hOff + t, score);
        }
        context.localBarrier();

        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = from + tid; t <= to; t += localSize) {
            float v = wrapAtt.get(hOff + t);
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float sliceMax = reduce[0];
        context.localBarrier();

        float localSum = 0.0f;
        for (int t = from + tid; t <= to; t += localSize) {
            float e = TornadoMath.exp(wrapAtt.get(hOff + t) - sliceMax);
            wrapAtt.set(hOff + t, e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sliceSum = reduce[0];
        context.localBarrier();

        // Unnormalised numerators: the combine divides by the merged denominator.
        for (int d = tid; d < headDim; d += localSize) {
            float acc = 0.0f;
            for (int t = from; t <= to; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                acc += wrapAtt.get(hOff + t) * valueCache.get(valueOffset + d);
            }
            attSplit.set(headBase + split * headDim + d, acc);
        }
        if (tid == 0) {
            attSplit.set(mBase + split, sliceMax);
            attSplit.set(lBase + split, sliceSum);
        }
    }

    /**
     * Sliding-window attention, one workgroup per head, lanes parallel within it.
     *
     * <p>Replaces a {@code @Parallel} loop over heads. That loop was launched on {@code
     * createAttentionWorker(nHeads, headDim)} — 8 workgroups of 64 lanes — and the emitted CUDA
     * walked it as {@code for (i = blockIdx*blockDim + threadIdx; i < nHeads; ...)}, so eight of
     * the five hundred and twelve threads did every position and every head dimension and the rest
     * fell through. It was 52.1% of decode kernel time.
     *
     * <p><b>Two of the three phases keep their original summation order</b>, which is why this is a
     * smaller numerical change than a tree reduction everywhere: each lane still accumulates a
     * whole score across {@code headDim} serially, and still accumulates a whole output element
     * across the window serially. Only the softmax maximum and the sum of exponentials become tree
     * reductions, because those are the two quantities the whole workgroup shares.
     *
     * <p>The grid must launch exactly {@code nHeads} workgroups: {@code h} is the workgroup index,
     * and the early return is what the head-count guard becomes once the loop is gone.
     */
    public static void attentionWithSlidingWindowParallel(
            KernelContext context,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray xb,
            FloatArray wrapAtt,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int cacheBaseOffset,
            int windowSize,
            int contextLength,
            int localMemSize) {

        int tid = context.localIdx;
        int h = context.groupIdx;
        int localSize = context.localGroupSizeX;
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int windowStart = Math.max(0, pos - windowSize + 1);
        int hOff = h * contextLength;
        int kvHeadIdx = h / kvMul;
        int qOffset = h * headDim;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = q.get(qOffset + i);
        }
        context.localBarrier();

        // STEP 1: scores, one lane per position. The dot product stays serial over headDim, so
        // each score is the same sum in the same order the per-head loop produced.
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            int keyOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
            float score = 0.0f;
            for (int i = 0; i < headDim; i++) {
                score += qShared[i] * keyCache.get(keyOffset + i);
            }
            // Gemma4 attention scaling = 1.0 (no 1/sqrt(headDim))
            wrapAtt.set(hOff + t, score);
        }
        context.localBarrier();

        // STEP 2a: maximum over the window, as a tree.
        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float v = wrapAtt.get(hOff + t);
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float maxScore = reduce[0];
        context.localBarrier();

        // STEP 2b: exponentials and their sum, as a tree.
        float localSum = 0.0f;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float e = TornadoMath.exp(wrapAtt.get(hOff + t) - maxScore);
            wrapAtt.set(hOff + t, e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sum = reduce[0];
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        context.localBarrier();

        // Normalize in place, as the per-head loop did: the weighted sum below then reads exactly
        // the values it read before, rounded exactly once, and wrapAtt is left holding the same
        // thing either kernel leaves in it.
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            wrapAtt.set(hOff + t, wrapAtt.get(hOff + t) * normFactor);
        }
        context.localBarrier();

        // STEP 3: weighted sum of values, one lane per output element. The accumulation over the
        // window stays serial and in order, so this element is the same sum it was before.
        for (int i = tid; i < headDim; i += localSize) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum += wrapAtt.get(hOff + t) * valueCache.get(valueOffset + i);
            }
            xb.set(qOffset + i, weightedSum);
        }
    }

    private static void gemma4ProcessHead(
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray xb,
            FloatArray wrapAtt,
            int h,
            int headDim,
            int kvDim,
            int kvMul,
            int cacheBaseOffset,
            int pos,
            int windowStart,
            int contextLength) {

        // wrapAtt is sized (nHeads * contextLength); index by absolute time t with a per-head
        // stride of contextLength.
        int hOff = h * contextLength;
        int kvHeadIdx = h / kvMul;
        int qOffset = h * headDim;

        // STEP 1: scores for t in [windowStart, pos]
        for (int t = windowStart; t <= pos; t++) {
            int keyOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
            float score = 0.0f;
            for (int i = 0; i < headDim; i++) {
                score += q.get(qOffset + i) * keyCache.get(keyOffset + i);
            }
            // Gemma4 attention scaling = 1.0 (no 1/sqrt(headDim))
            wrapAtt.set(hOff + t, score);
        }

        // STEP 2: softmax over [windowStart, pos]
        float maxScore = wrapAtt.get(hOff + windowStart);
        for (int t = windowStart + 1; t <= pos; t++) {
            float val = wrapAtt.get(hOff + t);
            if (val > maxScore) {
                maxScore = val;
            }
        }
        float sum = 0.0f;
        for (int t = windowStart; t <= pos; t++) {
            int idx = hOff + t;
            float expScore = TornadoMath.exp(wrapAtt.get(idx) - maxScore);
            wrapAtt.set(idx, expScore);
            sum += expScore;
        }
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        for (int t = windowStart; t <= pos; t++) {
            int idx = hOff + t;
            wrapAtt.set(idx, wrapAtt.get(idx) * normFactor);
        }

        // STEP 3: weighted sum of values
        for (int i = 0; i < headDim; i++) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum += wrapAtt.get(hOff + t) * valueCache.get(valueOffset + i);
            }
            xb.set(h * headDim + i, weightedSum);
        }
    }

    /**
     * Fused GeGLU FFN gate/up projection: {@code hb[row] = gelu(W1[row]. xNorm) * (W3[row].
     * xNorm)}. Mirrors {@code TransformerComputeKernelsLayered.fusedRmsNormFFNGateUp} but (a) takes
     * an already-normalized input -- Gemma4 materializes the normalized branch separately via
     * {@link #applyRmsNorm} since the same normalized {@code xb} also feeds the attention QKV
     * projections -- and (b) uses GELU rather than SiLU (see {@link
     * TransformerComputeKernelsLayered#geluActivation}).
     */
    public static void fusedGateUpGeGLU(
            KernelContext context,
            FloatArray xNorm,
            FloatArray hb,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= hiddenDim) {
            return;
        }

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);
        int rowOffset = rowId * dim;

        // === W1 (gate) ===
        float sum1 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            sum1 += w1.get(rowOffset + j).getFloat32() * xNorm.get(j);
        }
        localSum[localId] = sum1;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];
        context.localBarrier();

        // === W3 (up) ===
        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            sum3 += w3.get(rowOffset + j).getFloat32() * xNorm.get(j);
        }
        localSum[localId] = sum3;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        if (localId == 0) {
            hb.set(rowId, TransformerComputeKernelsLayered.geluActivation(result1) * result3);
        }
    }

    /**
     * Q8_0 counterpart of {@link #fusedGateUpGeGLU}: identical GeGLU fusion, but the gate ({@code
     * w1}) and up ({@code w3}) weights are Q8_0-quantized byte arrays, dequantized on the fly by
     * {@link TransformerComputeKernelsLayered#matrixVectorRowMajorOptimizedQ8_0Byte}. One row per
     * workgroup.
     */
    public static void fusedGateUpGeGLUQ8(
            KernelContext context,
            FloatArray xNorm,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= hiddenDim) {
            return;
        }

        float sum1 =
                TransformerComputeKernelsLayered.matrixVectorRowMajorOptimizedQ8_0Byte(
                        context, localWorkGroupSize, xNorm, w1, dim);
        float sum3 =
                TransformerComputeKernelsLayered.matrixVectorRowMajorOptimizedQ8_0Byte(
                        context, localWorkGroupSize, xNorm, w3, dim);

        if (localId == 0) {
            hb.set(rowId, TransformerComputeKernelsLayered.geluActivation(sum1) * sum3);
        }
    }

    /** {@code gate[i] = gelu(gate[i]) * perLayerInputs[peOffset + i]} -- the PLE gating step. */
    public static void pleGateGeluMul(
            KernelContext context,
            FloatArray gate,
            FloatArray perLayerInputs,
            int peOffset,
            int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            float gated = TransformerComputeKernelsLayered.geluActivation(gate.get(gid));
            gate.set(gid, gated * perLayerInputs.get(peOffset + gid));
        }
    }

    /**
     * Per-segment scale + RMSNorm with a single shared learned scale, used for the per-layer
     * projection's normalization: {@code perLayerProjScratch} is laid out as {@code
     * [numLayers][segmentSize]}, and {@code weight} (size {@code segmentSize}) is reused
     * identically for every segment. One workgroup processes one segment (segment index = {@code
     * groupIdx}), mirroring {@code rmsnorm(scratch, scratch, perLayerProjNorm, l*segmentSize,
     * segmentSize, eps)} for every {@code l}.
     */
    public static void pleProjScaleAndNormalize(
            KernelContext context,
            FloatArray x,
            FloatArray weight,
            int segmentSize,
            int localMemSize,
            float preScale,
            float rmsNormEps) {
        int segIdx = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int base = segIdx * segmentSize;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0f;
        for (int i = localId; i < segmentSize; i += localSize) {
            float v = x.get(base + i) * preScale;
            x.set(base + i, v);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / segmentSize + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < segmentSize; i += localSize) {
            float normalized = ss * x.get(base + i);
            x.set(base + i, weight.get(i) * normalized);
        }
    }

    /** Final logit soft-capping: {@code logits[i] = softcap * tanh(logits[i] / softcap)}. */
    public static void applyLogitSoftcap(
            KernelContext context, FloatArray logits, float softcap, int size) {
        int gid = context.globalIdx;
        if (gid < size) {
            float v = logits.get(gid);
            logits.set(gid, TornadoMath.tanh(v / softcap) * softcap);
        }
    }
}
