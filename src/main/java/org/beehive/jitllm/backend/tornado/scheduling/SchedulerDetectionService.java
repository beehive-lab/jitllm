package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

/** The device facts scheduling branches on. */
public class SchedulerDetectionService {

    /**
     * Whether the active device evaluates warp/sub-group shuffle reductions ({@code
     * KernelContext.simdShuffleDown}) correctly. CUDA does; the OpenCL backend compiles the shuffle
     * but produces incorrect results, so the warp-shuffle GEMV kernels must only run where this
     * holds (elsewhere the shared-memory GEMVs are used).
     */
    public static boolean isWarpShuffleSupported() {
        return TornadoDevices.current().capabilities().supports(DeviceCapability.WARP_SHUFFLE);
    }

    /**
     * Whether the active device's {@code KernelContext.simdShuffleDown} produces correct results
     * for the fused Q/K/V projection kernel's 32-lane butterfly reduction — verified on Metal, and
     * deliberately independent of {@link #isWarpShuffleSupported()} (CUDA's answer to a different
     * question). Where this is false, the fused QKV projection uses the shared-memory reduction
     * kernel instead ({@code fusedQKVMatmulX}), which works everywhere.
     */
    /** Whether packed FP16 pair arithmetic holds CPU parity here. */
    public static boolean isPackedHalf2MathSupported() {
        return TornadoDevices.current().capabilities().supports(DeviceCapability.PACKED_HALF2_MATH);
    }

    /**
     * Whether the shuffle-reducing FP16 matrix-vector kernels compute correct results on the active
     * device. A support question; see {@link DeviceCapability#SHUFFLE_REDUCED_FP16_GEMV}. Whether
     * to prefer them over their shared-memory twins is decided by {@link Fp16GemvReductionPolicy},
     * which is what the layers branch on.
     */
    public static boolean isShuffleReducedFp16GemvSupported() {
        return TornadoDevices.current()
                .capabilities()
                .supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV);
    }

    public static boolean isSubgroupShuffle32Supported() {
        return TornadoDevices.current()
                .capabilities()
                .supports(DeviceCapability.SUBGROUP_SHUFFLE_32);
    }

    /**
     * Whether the multi-workgroup split-KV flash-decoding attention kernel ({@code
     * processHeadsFlashAttentionSplitKV}) fails to JIT — true on Metal, where Qwen3 layers fall
     * back to the single-workgroup-per-head online-softmax kernel.
     *
     * <p>Kept under its old name because ~10 call sites read it as a flag; what it now asks is the
     * capability rather than the backend's identity.
     */
    public static boolean isMetalBackend() {
        return !TornadoDevices.current()
                .capabilities()
                .supports(DeviceCapability.SPLIT_KV_ATTENTION);
    }

    /**
     * The scheduler type is <b>not</b> a pure device fact: it is an NVIDIA-class device <i>and</i>
     * a model that is not Mistral. The device half is {@link DeviceCapability#SINGLE_PASS_RMS}; the
     * model half stays here, because a model is not something {@code TornadoDevices} can or should
     * know about.
     */
    public static SchedulerType determineSchedulerType(Model model) {
        boolean singlePassRms =
                TornadoDevices.current().capabilities().supports(DeviceCapability.SINGLE_PASS_RMS);
        boolean isNotMistral = model.getModelType() != ModelType.MISTRAL;
        return (singlePassRms && isNotMistral) ? SchedulerType.NVIDIA : SchedulerType.NON_NVIDIA;
    }
}
