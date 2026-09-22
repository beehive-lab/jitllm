package org.beehive.jllm.backend.tornado.scheduling;

import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.runtime.backend.DeviceCapability;

// @formatter:off
/**
 * Whether decode attention runs the lane-cooperative kernel instead of the per-key one.
 *
 * <p>Support and preference are kept apart here for the same reason they are in {@link
 * Fp16GemvReductionPolicy}. The support half is {@link DeviceCapability#SHUFFLE_REDUCED_FP16_GEMV}:
 * the kernel reduces with {@code simdShuffleDown} and broadcasts with {@code simdBroadcastFirst},
 * so a backend that miscompiles the warp shuffle would compute a wrong answer, silently. OpenCL is
 * such a backend.
 *
 * <p>The shape half is not a preference at all but a hard precondition. The kernel assigns lane
 * {@code L} the head dimensions {@code 2L}, {@code 2L+1}, {@code 64+2L}, {@code 64+2L+1}, which is
 * a correct partition of exactly a 128-wide head across exactly 32 lanes. Anything else is not
 * slower, it is wrong, so the check is an equality and not a heuristic.
 *
 * <h2>Scope</h2>
 *
 * <p>Reached by the Qwen3 FP16 family's split-KV decode attention over an FP16 paged cache. Every
 * other family, every other quantisation, the FP32 key/value cache, the packed-half2 variant and
 * the non-split per-head kernel keep the kernel they had. Head widths other than 128 keep it too,
 * which is what leaves Llama-3.2-1B (64) and the Qwen3.5 family (256) exactly as they were.
 *
 * <p>Measured on one device, an RTX 5070 Ti (sm_120). The decision is a backend-level default
 * rather than a per-device measurement, and what it rests on is structural rather than incidental:
 * the kernel it replaces spends 34052 bytes of shared memory per block on a private 128-float
 * accumulator per thread, which caps it at six resident warps per SM of a possible 48, and reaches
 * every one of those floats through a bank all 32 lanes of the warp share.
 */
// @formatter:on
public final class LaneAttentionPolicy {

    /** The only head width the lane-cooperative kernel is written for. */
    public static final int SUPPORTED_HEAD_SIZE = 128;

    // @formatter:off
    /**
     * Warps per (head, split), which is how many independent key streams a block keeps in flight.
     *
     * <p>Not a free parameter; it was screened. One warp per block was measured first and is 1.65x
     * faster than the kernel it replaces at depth zero and 2.3x <b>slower</b> at depth 2048: with
     * 16 heads and 8 splits that is 128 warps on a 70-SM device, under two warps per SM, and every
     * warp's loop carries a dependent chain of load, reduce, score, load, accumulate, with nothing
     * to hide the KV-cache latency behind. Four, eight, sixteen and thirty-two were then measured,
     * and tg128 at depth 2048 reads 286.4, 351.0, <b>391.0</b> and 367.8 tok/s. Thirty-two
     * regresses because a 512-thread block at 34 registers stops fitting three to an SM.
     *
     * <p>Sixteen warps take the grid to 2048 warps and give each block sixteen independent key
     * streams, for 8320 bytes of shared memory in the end-of-block fold against the replaced
     * kernel's 34052 bytes in its inner loop.
     */
    // @formatter:on
    public static final int WARPS_PER_GROUP = 16;

    private LaneAttentionPolicy() {}

    /**
     * Whether the lane-cooperative kernel may be used for this head width on the active device.
     *
     * @param headSize the model's head width; anything but {@link #SUPPORTED_HEAD_SIZE} is refused
     */
    public static boolean laneCooperativeAttention(int headSize) {
        return headSize == SUPPORTED_HEAD_SIZE
                && TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV);
    }
}
