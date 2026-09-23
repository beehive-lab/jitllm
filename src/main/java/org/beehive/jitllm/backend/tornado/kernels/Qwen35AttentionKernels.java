package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Device kernels for the {@code qwen35} attention layers — the one in four that attends.
 *
 * <p>Three things separate this from Qwen3's attention, and none of them is a different algorithm.
 * The query projection carries an output gate interleaved with the query; the rotary width is
 * smaller than the head; and the attention result is gated by a logistic before the output
 * projection. Everything else — the per-head query and key norms, the attention itself, the paged
 * key/value walk — is Qwen3's and is intended to be reused rather than restated.
 *
 * <p><b>One of those reuses is a claim, not yet a result.</b> {@code Qwen3Kernels.fusedQKRmsNorm}
 * is parameterized by head count and head width, so it should serve a 256-wide head as readily as
 * Qwen3's 128-wide one — but it reduces through local memory and barriers, so unlike the kernels
 * here it cannot be run on the host, and nothing has yet exercised it at this width. The device
 * gate has to settle it.
 *
 * <p>Written in the same shape as {@link Qwen35DeltaNetKernels}: every body is a static method
 * taking its lane index, with the kernel a two-line wrapper passing {@code context.globalIdx}. That
 * is what lets {@code Qwen35AttentionKernelParityTest} run each lane on the host against {@code
 * CpuOperations}. A body written directly against {@link KernelContext} could only be exercised by
 * running a model on a device, where a wrong index produces slightly wrong text rather than a
 * failure.
 */
public final class Qwen35AttentionKernels {

    private Qwen35AttentionKernels() {}

    // ---- the query/gate split ------------------------------------------------

    /**
     * One element of the fused query/gate projection, separated into its two halves.
     *
     * <p>{@code attn_q} is twice a query projection's width, holding per head a query slice then a
     * gate slice — element stride between heads is {@code 2 * headDim}. Downstream everything wants
     * contiguous heads: the per-head norm, the rotation and attention all address {@code head *
     * headDim}. Separating them once costs one copy and saves a stride parameter on three kernels.
     *
     * @param lane {@code head * headDim + element}, over the query's width
     */
    static void splitQueryGateLane(
            FloatArray fused, FloatArray query, FloatArray gate, int headDim, int lane) {
        int head = lane / headDim;
        int element = lane - head * headDim;
        int fusedBase = head * 2 * headDim;
        query.set(lane, fused.get(fusedBase + element));
        gate.set(lane, fused.get(fusedBase + headDim + element));
    }

    /** One lane per query element — {@code heads * headDim} of them. */
    public static void splitQueryGate(
            KernelContext context,
            FloatArray fused,
            FloatArray query,
            FloatArray gate,
            int heads,
            int headDim) {
        int lane = context.globalIdx;
        if (lane >= heads * headDim) {
            return;
        }
        splitQueryGateLane(fused, query, gate, headDim, lane);
    }

    // ---- partial rotary ------------------------------------------------------

    /**
     * One rotated pair of one head, over the first {@code rotaryDim} of that head only.
     *
     * <p>{@code rope.dimension_count} is stated independently of the head width — 64 against 256
     * here — so three quarters of every head passes through unrotated. That is a parameter of the
     * rotation, not a different scheme: the pairs are {@code (ic, ic + rotaryDim/2)} exactly as
     * NeoX pairs them, and the elements at or above {@code rotaryDim} are simply never addressed.
     *
     * <p>The frequencies are read from the precomputed tables rather than recomputed with {@code
     * pow} and {@code cos}. Faster, and it keeps the device bit-identical to the host, which reads
     * the same tables — computing them here in float where the loader computed them in double would
     * put a rounding difference at the front of every layer.
     *
     * <p>Keys are rotated only for the heads that have them, which is how grouped query attention
     * shows up here.
     *
     * @param lane {@code head * (rotaryDim / 2) + ic}
     */
    static void ropeNeoxPartialLane(
            FloatArray query,
            FloatArray key,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int position,
            int keyValueHeads,
            int headDim,
            int rotaryDim,
            int lane) {
        int half = rotaryDim / 2;
        int head = lane / half;
        int ic = lane - head * half;

        float fcr = freqCisReal.get(position * half + ic);
        float fci = freqCisImag.get(position * half + ic);

        int base = head * headDim;
        float q0 = query.get(base + ic);
        float q1 = query.get(base + ic + half);
        query.set(base + ic, q0 * fcr - q1 * fci);
        query.set(base + ic + half, q0 * fci + q1 * fcr);

        if (head < keyValueHeads) {
            float k0 = key.get(base + ic);
            float k1 = key.get(base + ic + half);
            key.set(base + ic, k0 * fcr - k1 * fci);
            key.set(base + ic + half, k0 * fci + k1 * fcr);
        }
    }

    /** One lane per rotated pair — {@code heads * rotaryDim / 2} of them. */
    public static void ropeNeoxPartial(
            KernelContext context,
            IntArray positionHolder,
            FloatArray query,
            FloatArray key,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int heads,
            int keyValueHeads,
            int headDim,
            int rotaryDim) {
        int lane = context.globalIdx;
        if (lane >= heads * (rotaryDim / 2)) {
            return;
        }
        ropeNeoxPartialLane(
                query,
                key,
                freqCisReal,
                freqCisImag,
                positionHolder.get(0),
                keyValueHeads,
                headDim,
                rotaryDim,
                lane);
    }

    // ---- key/value append ----------------------------------------------------

    /**
     * One element of this step's key and value, written into the paged store.
     *
     * <p>Separate from the rotation, where Llama and Qwen3 fuse the two. They can fuse because they
     * rotate the whole head, so the lanes that rotate are exactly the lanes that must be written.
     * Here the rotation covers a quarter of each head and the append covers all of it, so one
     * kernel would either leave three quarters of the cache unwritten or need lanes that do nothing
     * but copy. Two kernels, each with the lane count its own job needs.
     *
     * @param lane an element of the key/value vector, {@code 0 .. kvDim - 1}
     */
    static void appendKeyValueLane(
            FloatArray key,
            FloatArray value,
            FloatArray keyCache,
            FloatArray valueCache,
            IntArray blockTable,
            int position,
            int slot,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride,
            int lane) {
        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        slot,
                        position,
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);
        keyCache.set(cacheOffset + lane, key.get(lane));
        valueCache.set(cacheOffset + lane, value.get(lane));
    }

    /** One lane per key/value element. */
    public static void appendKeyValuePaged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray key,
            FloatArray value,
            FloatArray keyCache,
            FloatArray valueCache,
            IntArray blockTable,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride) {
        int lane = context.globalIdx;
        if (lane >= kvDim) {
            return;
        }
        appendKeyValueLane(
                key,
                value,
                keyCache,
                valueCache,
                blockTable,
                positionHolder.get(0),
                positionHolder.get(1),
                kvDim,
                layer,
                blockCfg,
                blockStride,
                lane);
    }

    /**
     * {@link #appendKeyValuePaged} into a half-precision store.
     *
     * <p>The entry is narrowed on the way in and widened on the way out; everything that consumes
     * it accumulates in FP32, so the only thing held in half precision is the store itself.
     */
    public static void appendKeyValueFP16Paged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray key,
            FloatArray value,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray blockTable,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride) {
        int lane = context.globalIdx;
        if (lane >= kvDim) {
            return;
        }
        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        positionHolder.get(1),
                        positionHolder.get(0),
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);
        keyCache.set(cacheOffset + lane, new HalfFloat(key.get(lane)));
        valueCache.set(cacheOffset + lane, new HalfFloat(value.get(lane)));
    }

    // ---- the output gate -----------------------------------------------------

    /**
     * One element of the attention result, scaled by the logistic of its gate.
     *
     * <p>A logistic, not a SiLU. {@code SwiGLU} multiplies by {@code x·σ(x)} and this multiplies by
     * {@code σ(x)}; reusing the SwiGLU kernel here would be an extra factor of the gate and output
     * that stays plausible.
     */
    static void applyOutputGateLane(FloatArray values, FloatArray gate, int lane) {
        float g = gate.get(lane);
        values.set(lane, values.get(lane) * (1.0f / (1.0f + TornadoMath.exp(-g))));
    }

    /** One lane per element of the attention result. */
    public static void applyOutputGate(
            KernelContext context, FloatArray values, FloatArray gate, int count) {
        int lane = context.globalIdx;
        if (lane >= count) {
            return;
        }
        applyOutputGateLane(values, gate, lane);
    }
}
