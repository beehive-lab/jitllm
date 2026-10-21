package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Threads per workgroup of the Q8_0 vocabulary projection, which runs one workgroup per vocabulary
 * row and reduces it through a shared-memory tree.
 *
 * <p>A preference, not a support question: the kernel computes the same result for any power-of-two
 * workgroup. The default, 256, comes from CUDA. On Apple GPUs it is the slowest choice measured:
 * with a 1024-wide row each thread loads four values and then spends eight barrier-separated steps
 * reducing them, for every one of the ~152k rows. Measured on an M4 Pro, Qwen3-0.6B Q8_0 decode
 * (tg64, tok/s): 256 threads 161.3, 128 179.1, 64 186.4, 32 189.0. Other backends keep 256, which
 * is what they were measured with.
 */
// @formatter:on
public final class LogitsWorkgroupPolicy {

    /** The workgroup the Q8_0 vocabulary projection was written and tuned for. */
    public static final int DEFAULT_Q8_THREADS = 256;

    /** One SIMD group: the fastest measured on Metal. */
    public static final int METAL_Q8_THREADS = 32;

    private LogitsWorkgroupPolicy() {}

    /** Threads per workgroup of the Q8_0 vocabulary projection on the active device. */
    public static int q8Threads() {
        return TornadoDevices.current().backend().equals(BackendId.METAL)
                ? METAL_Q8_THREADS
                : DEFAULT_Q8_THREADS;
    }
}
