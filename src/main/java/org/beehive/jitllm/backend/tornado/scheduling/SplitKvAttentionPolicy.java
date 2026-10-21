package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;

// @formatter:off
/**
 * How many threads a workgroup of the FP32 split-KV attention kernel has.
 *
 * <p>Not a tuning knob but a fit. {@code processHeadsFlashAttentionSplitKVPaged} gives every thread
 * of its workgroup a private row of a 128-wide accumulator in local memory, so its local arrays
 * grow with the workgroup: 34052 bytes for 64 threads, 17284 for 32. Apple GPUs allow 32 KB per
 * threadgroup, and Metal refuses to build a pipeline that asks for more, which is why split-KV
 * attention used to be withheld from Metal altogether. A device whose local memory holds the
 * 64-thread arrays runs that kernel as before; one whose does not runs {@code
 * processHeadsFlashAttentionSplitKVPaged32}, the same arithmetic in workgroups of 32.
 *
 * <p>The kernel and its worker grid must agree: the 32-thread kernel on a 64-thread grid would
 * index past its local arrays. Both are therefore chosen from {@link #narrow()}.
 *
 * <p>Measured on an Apple M4 Pro, Qwen3-0.6B F16 with the FP32 cache: decode attention goes from
 * 38.5 to 4.9 ms per token at depth ~250 against the single-workgroup-per-head kernel Metal used
 * before, with greedy text identical.
 */
// @formatter:on
public final class SplitKvAttentionPolicy {

    /** Threads per workgroup where the local arrays fit: the kernel's original shape. */
    public static final int WIDE_GROUP = 64;

    /** Threads per workgroup where they do not. */
    public static final int NARROW_GROUP = 32;

    /** The head width the kernel's local arrays are sized for. */
    static final int MAX_HEAD_SIZE = 128;

    // @formatter:off
    /**
     * Splits per head where the 32-thread kernel runs, unless {@code
     * jitllm.attention.splitKv.count} is set. Each workgroup zeroes and folds a 32x128 accumulator
     * whatever its share of the cache, so on a short context more splits buy more fixed work than
     * parallelism. Measured on an M4 Pro, Qwen3-0.6B tg32 (tok/s) at depth 0 / 1024: F16 4 splits
     * 18.9 / 15.9, 8 splits 15.9 / 16.5, 16 splits 16.7 / 14.5; Q8_0 4 splits 20.0 / 17.9, 8 splits
     * 19.0 / 16.7, 16 splits 17.5 / 16.3.
     */
    // @formatter:on
    public static final int NARROW_SPLITS = 4;

    private SplitKvAttentionPolicy() {}

    /**
     * The local memory the FP32 split-KV kernel declares for a workgroup of {@code threads}: the
     * staged query, one accumulator row per thread, three per-thread scalars and a broadcast slot.
     */
    public static long localBytes(int threads) {
        return Float.BYTES * (MAX_HEAD_SIZE + (long) threads * MAX_HEAD_SIZE + 3L * threads + 1L);
    }

    /**
     * Threads per workgroup on a device with {@code localMemoryBytes} of local memory per
     * workgroup; 0 means the runtime did not say, which keeps the original shape.
     */
    public static int groupSize(long localMemoryBytes) {
        if (localMemoryBytes > 0 && localBytes(WIDE_GROUP) > localMemoryBytes) {
            return NARROW_GROUP;
        }
        return WIDE_GROUP;
    }

    /**
     * Splits per head for a split-KV scratch sized for {@code capacity} splits: {@link
     * #NARROW_SPLITS} where the 32-thread kernel runs and the count was not set explicitly, the
     * capacity otherwise.
     */
    public static int splits(int capacity) {
        if (System.getProperty("jitllm.attention.splitKv.count") != null || !narrow()) {
            return capacity;
        }
        return Math.min(capacity, NARROW_SPLITS);
    }

    /** Whether the active device runs the 32-thread kernel. */
    public static boolean narrow() {
        return groupSize(TornadoDevices.current().localMemoryBytes()) == NARROW_GROUP;
    }
}
