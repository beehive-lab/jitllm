package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Whether a layer's RMS sum of squares is reduced in one workgroup, which computes the scale
 * itself, instead of across workgroups followed by a separate {@code *_rms_finalize} task.
 *
 * <p>A preference: both shapes compute the same scale. The NVIDIA path reduces in one workgroup
 * already (see {@code AbstractTransformerLayerTaskGraphs.rmsReduceKernel}); elsewhere the extra
 * task per normalization stays, because the families were validated that way. On Metal, where
 * kernels are batched into shared command buffers and every task is a dispatch the GPU runs in
 * turn, dropping the two finalize tasks per layer is measurable for Qwen3: Qwen3-0.6B decode on an
 * M4 Pro (tg64, tok/s, depth 0) F16 159-161 to 164, Q8_0 203.7 to 207, greedy text identical. Only
 * the Qwen3 and Llama layers ask; the other families keep their validated shape.
 */
// @formatter:on
public final class RmsReductionPolicy {

    private RmsReductionPolicy() {}

    /** Whether the active device prefers the single-workgroup RMS reduction. */
    public static boolean singleWorkgroup() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }
}
