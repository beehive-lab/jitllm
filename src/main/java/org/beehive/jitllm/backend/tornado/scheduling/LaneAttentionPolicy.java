package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

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
 * <p>The shape half is not a preference at all but a hard precondition. There is one kernel per
 * head width: the 128-wide one assigns lane {@code L} the head dimensions {@code 2L}, {@code 2L+1},
 * {@code 64+2L}, {@code 64+2L+1}, the 96-wide one {@code 2L}, {@code 2L+1} and {@code 64+L}, the
 * 64-wide one {@code 2L} and {@code 2L+1}, each a correct partition of exactly that width across
 * exactly 32 lanes. Any other width is not slower, it is wrong, so the check is an equality and not
 * a heuristic. This class is the one place that maps a head width to its kernel and its grid.
 *
 * <h2>Scope</h2>
 *
 * <p>Reached by split-KV decode attention over an FP16 paged cache in the Llama (Q4_0, Q8_0, FP16)
 * Qwen3 (Q8_0, FP16), Qwen2 (Q8_0, FP16), Granite (Q8_0, through the scaled kernels) and Phi-3
 * (Q8_0, 96-wide heads) families. The FP32 key/value cache, the packed-half2 variant and the
 * non-split per-head kernel keep the kernels they had, as does the Qwen3.5 family (256-wide heads),
 * which has its own single-warp kernel on CUDA and its own lane-cooperative kernels on Metal
 * ({@link #laneCooperativeAttentionHead256}). The FP32-cache kernels ({@link
 * #laneCooperativeAttentionFp32Cache}) are a separate, Metal-only decision: Metal keeps the
 * key/value cache in FP32.
 *
 * <p>Measured on one device, an RTX 5070 Ti (sm_120). The decision is a backend-level default
 * rather than a per-device measurement, and what it rests on is structural rather than incidental:
 * the kernel it replaces spends 34052 bytes of shared memory per block on a private 128-float
 * accumulator per thread, which caps it at six resident warps per SM of a possible 48, and reaches
 * every one of those floats through a bank all 32 lanes of the warp share.
 */
// @formatter:on
public final class LaneAttentionPolicy {

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

    /** The 128-wide head the first lane-cooperative kernels were written for. */
    public static final int SUPPORTED_HEAD_SIZE = 128;

    /**
     * Warps per (head, split) for the 64-wide kernel. Half the work per key of a 128-wide head, so
     * fewer warps keep the group busy: four measured fastest of two, four and eight on an A10
     * (Llama-3.2-1B, tg128 at depths 0 and 680).
     */
    public static final int HEAD64_WARPS_PER_GROUP = 4;

    /**
     * Warps per (head, split) for the 96-wide kernel: eight measured fastest of four, eight and
     * sixteen on an A10 (Phi-3-mini, tg128 at depths 0 and 680).
     */
    public static final int HEAD96_WARPS_PER_GROUP = 8;

    /** The signature of a lane-cooperative split-KV kernel that also takes the score scale. */
    public interface ScaledKernel
            extends TornadoFunctions.Task16<
                    KernelContext,
                    FloatArray,
                    HalfFloatArray,
                    HalfFloatArray,
                    FloatArray,
                    Integer,
                    Integer,
                    Integer,
                    Integer,
                    IntArray,
                    Integer,
                    IntArray,
                    Integer,
                    Integer,
                    Integer,
                    Float> {}

    /** The signature every lane-cooperative split-KV kernel shares. */
    public interface Kernel
            extends TornadoFunctions.Task15<
                    KernelContext,
                    FloatArray,
                    HalfFloatArray,
                    HalfFloatArray,
                    FloatArray,
                    Integer,
                    Integer,
                    Integer,
                    Integer,
                    IntArray,
                    Integer,
                    IntArray,
                    Integer,
                    Integer,
                    Integer> {}

    // @formatter:off
    /**
     * Warps per (query token, head) in the batched-prefill lane-cooperative kernel. Prefill
     * launches one workgroup per token and head, so there is no shortage of workgroups to hide
     * latency with, and fewer warps per workgroup keep more of them resident. Measured on an M4
     * Pro, Qwen3-0.6B F16, batch 256 (pp512 / pp2048 tok/s): 1 warp 1671 / 681, 2 1947 / 1024, 4
     * 1961 / 1166, 8 1937 / 999, 16 1822 / 937.
     */
    // @formatter:on
    public static final int PREFILL_WARPS_PER_GROUP = 4;

    private LaneAttentionPolicy() {}

    /**
     * Whether a lane-cooperative kernel may be used for this head width on the active device: CUDA
     * for every width with a kernel, and Metal for the 128-wide one, where it is the FP16 cache's
     * split-KV decode attention.
     *
     * @param headSize the model's head width; anything but 64, 96 or 128 is refused
     */
    public static boolean laneCooperativeAttention(int headSize) {
        var capabilities = TornadoDevices.target().capabilities();
        if (capabilities.supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV)) {
            return headSize == 64 || headSize == 96 || headSize == 128;
        }
        // Metal: the FP32-cache twin already runs there on the same shuffles, and the 64-thread
        // per-key FP16 kernel needs 34052 bytes of threadgroup memory, past Metal's 32 KB. Only the
        // 128-wide kernel has been verified there.
        return headSize == SUPPORTED_HEAD_SIZE
                && capabilities.supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }

    /** The kernel for {@code headSize}, which {@link #laneCooperativeAttention} admitted. */
    public static Kernel kernel(int headSize) {
        return switch (headSize) {
            case 64 ->
                    TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16PagedLaneHead64;
            case 96 ->
                    TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16PagedLaneHead96;
            case 128 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead128;
            default ->
                    throw new IllegalArgumentException(
                            "no lane-cooperative attention kernel for head width " + headSize);
        };
    }

    /**
     * Whether a lane-cooperative kernel that takes its score scale exists for this head width: for
     * a family whose attention multiplier is not {@code 1/sqrt(headSize)}.
     */
    public static boolean scaledLaneCooperativeAttention(int headSize) {
        return (headSize == 64 || headSize == 128) && laneCooperativeAttention(headSize);
    }

    /**
     * The scaled kernel for {@code headSize}, which {@link #scaledLaneCooperativeAttention}
     * admitted.
     */
    public static ScaledKernel scaledKernel(int headSize) {
        return switch (headSize) {
            case 64 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead64Scaled;
            case 128 ->
                    TransformerPagedKvKernels
                            ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead128Scaled;
            default ->
                    throw new IllegalArgumentException(
                            "no scaled lane-cooperative attention kernel for head width "
                                    + headSize);
        };
    }

    /** The grid for {@code groups} (head, split) pairs of a {@code headSize}-wide head. */
    public static WorkerGrid worker(int groups, int headSize) {
        int warps =
                switch (headSize) {
                    case 64 -> HEAD64_WARPS_PER_GROUP;
                    case 96 -> HEAD96_WARPS_PER_GROUP;
                    default -> WARPS_PER_GROUP;
                };
        return WorkerGridFactory.createLaneAttentionWorker(groups, warps);
    }

    // @formatter:off
    /**
     * Whether split-KV decode attention over the FP32 cache runs the lane-cooperative kernel.
     *
     * <p>Granted where {@link DeviceCapability#SUBGROUP_SHUFFLE_32} holds (Metal), whose FP32 cache
     * otherwise runs {@code processHeadsFlashAttentionSplitKVPaged32}: one thread per key, each
     * reading a whole key row and folding a 128-wide accumulator through threadgroup memory for
     * every key. Same 128-wide precondition as {@link #laneCooperativeAttention}. CUDA's FP32 cache
     * keeps its kernel; it was not measured there.
     */
    // @formatter:on
    public static boolean laneCooperativeAttentionFp32Cache(int headSize) {
        return headSize == SUPPORTED_HEAD_SIZE
                && TornadoDevices.target()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }

    /** The narrower head width the FP32-cache lane-cooperative kernel also has a variant for. */
    public static final int NARROW_HEAD_SIZE = 64;

    /**
     * {@link #laneCooperativeAttentionFp32Cache} for a layer stack that installs the kernel
     * matching its head width: {@code processHeadsFlashAttentionSplitKVPagedLaneHead128} for 128,
     * {@code processHeadsFlashAttentionSplitKVPagedLaneHead64} for 64 (lane {@code L} owns
     * dimensions {@code 2L} and {@code 2L+1}). Any other width keeps its kernel.
     */
    public static boolean laneCooperativeAttentionFp32CacheAnyHead(int headSize) {
        return (headSize == SUPPORTED_HEAD_SIZE || headSize == NARROW_HEAD_SIZE)
                && TornadoDevices.target()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }

    /** The 256-wide head of the Qwen3.5 family, which has its own lane-cooperative kernels. */
    public static final int WIDE_HEAD_SIZE = 256;

    // @formatter:off
    /**
     * Warps per (head, split) for the 256-wide kernels. Screened on an Apple M4 Pro with the
     * Qwen3.5-0.8B shape (8 query heads over 2 key/value heads), kernel plus combine, FP32 cache,
     * depths 680 / 2000 (us): 8 splits x 4 warps 16.6 / 32.8, x 8 warps 16.9 / 32.0, x 16 warps
     * 24.7 / 45.8; 16 splits x 8 warps 19.0 / 32.7; FP16 cache at 2000, 8 x 8 23.2, 8 x 4 28.7, 16
     * x 8 26.0. The single-threadgroup-per-head kernel it replaces: 3708 / 11220 us. End to end
     * (tg128) every shape from 4 to 32 splits and 4 to 16 warps was within 2%.
     */
    // @formatter:on
    public static final int HEAD256_WARPS_PER_GROUP = 8;

    // @formatter:off
    /**
     * Splits per head for the 256-wide kernels, unless {@code jitllm.attention.splitKv.count} is
     * set; screened with {@link #HEAD256_WARPS_PER_GROUP}. At 8 query heads that is 64 groups of
     * 256 threads.
     */
    // @formatter:on
    public static final int HEAD256_SPLITS = 8;

    // @formatter:off
    /**
     * Whether decode attention over a 256-wide head runs the lane-cooperative split-KV kernels
     * {@code processHeadsFlashAttentionSplitKVPagedLaneHead256} (FP32 cache) and {@code
     * processHeadsFlashAttentionSplitKVFP16PagedLaneHead256} (FP16 cache).
     *
     * <p>Granted where {@link DeviceCapability#SUBGROUP_SHUFFLE_32} holds (Metal), whose decode
     * attention for this width was otherwise one threadgroup per query head walking every position
     * in 8-position tiles, a cost linear in the context with a handful of threadgroups in flight.
     * CUDA keeps its own 256-wide split-KV warp kernel and FP32-cache kernel.
     */
    // @formatter:on
    public static boolean laneCooperativeAttentionHead256(int headSize) {
        return headSize == WIDE_HEAD_SIZE
                && TornadoDevices.target()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }

    /**
     * Splits per head for the 256-wide kernels against a split scratch sized for {@code capacity}
     * splits: {@code jitllm.attention.splitKv.count} when set, {@link #HEAD256_SPLITS} otherwise,
     * never more than the capacity.
     */
    public static int head256Splits(int capacity) {
        int splits = Integer.getInteger("jitllm.attention.splitKv.count", HEAD256_SPLITS);
        return Math.max(1, Math.min(capacity, splits));
    }

    /** The grid for {@code groups} (head, split) pairs of the 256-wide kernels. */
    public static WorkerGrid head256Worker(int groups) {
        return WorkerGridFactory.createLaneAttentionWorker(groups, HEAD256_WARPS_PER_GROUP);
    }
}
