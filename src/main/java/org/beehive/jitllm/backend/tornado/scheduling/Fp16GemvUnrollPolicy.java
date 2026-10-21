package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Whether the shuffle-reduced FP16 matrix-vector kernels run their unrolled variants, which keep
 * four independent loads in flight per lane.
 *
 * <p>A preference: both variants compute the same sums, in a different order. On an Apple GPU the
 * unrolled ones are faster: Qwen3-0.6B F16 decode on an M4 Pro went from 153.4 to 164.7 tok/s at
 * depth 0. On CUDA they are slower: Qwen3-0.6B F16 decode on an A100 drops from about 146 to 135
 * tok/s, so the other backends keep the original kernels.
 */
// @formatter:on
public final class Fp16GemvUnrollPolicy {

    private Fp16GemvUnrollPolicy() {}

    /** Whether the active device runs the unrolled FP16 matrix-vector kernels. */
    public static boolean unrolled() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }
}
