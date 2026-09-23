package org.beehive.jllm.backend.tornado.scheduling;

import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.runtime.backend.DeviceCapability;

// @formatter:off
/**
 * Whether to reduce the decode-shaped FP16 matrix-vector kernels with a 32-lane warp butterfly
 * instead of a shared-memory tree.
 *
 * <p>This is the <b>preference</b>, and it is deliberately not a {@link DeviceCapability}. {@link
 * DeviceCapability#SHUFFLE_REDUCED_FP16_GEMV} says the shuffle-reducing kernels are correct on this
 * device; that is a fact about the instruction set and the backend's lowering. Whether running them
 * is faster is a fact about a <i>workload</i> on a <i>device</i>, which is a different question
 * with different evidence behind it, and putting it in the capability set made the two
 * indistinguishable.
 *
 * <h2>What this preference covers</h2>
 *
 * <p>Five kernels, all of them single-column (one token) FP16 matrix-vector products:
 *
 * <ul>
 *   <li>{@code fusedRmsNormQKVMatmulWarp} and {@code fusedRmsNormFFNGateUpWarp}, plus {@code
 *       matrixVectorGenericWithResidualSimd32} at the attention-output and FFN-down projections.
 *       These are selected by {@code Qwen3FP16FFNLayers}, so they reach the <b>Qwen3 FP16 family
 *       only</b> — every plan shape it builds (standard, prefill-decode and batched
 *       prefill-decode), in the decode half.
 *   <li>{@code matrixVectorGenericSimd32} at the vocabulary projection, selected by {@code
 *       LogitsFP16Layer}. That class is shared, so this one reaches <b>every FP16 family whose
 *       logits layer is the base class</b>: Llama, Mistral, Phi-3, Devstral, Qwen2 and Qwen3. It
 *       does <b>not</b> reach Granite or Gemma 4, whose logits layers override {@code
 *       setupLogitsTaskGraph} and install their own scaled or soft-capped projection.
 * </ul>
 *
 * <p>Backends: the preference is expressed on CUDA. Metal reaches the same kernels through {@link
 * DeviceCapability#SUBGROUP_SHUFFLE_32}, which is an older, correctness-motivated selection and is
 * not this decision. OpenCL cannot run these kernels at all.
 *
 * <h2>What the evidence is, and what it is not</h2>
 *
 * <p>Measured on <b>one device</b>: an RTX 5070 Ti (sm_120), Qwen3-0.6B FP16, tg128 at depth zero,
 * 344.1 tok/s with the shared-memory reduction against 393.9 with the shuffle on the four layer
 * kernels and 414.7 once the vocabulary projection joins them; 200.1, 214.9 and 221.0 at depth
 * 2048. Broadened across models on that device by an A/B over every FP16 family on the test machine
 * that reaches the shared logits layer, recorded in the campaign report.
 *
 * <p><b>It is not a per-device measurement.</b> The grant applies to every CUDA device and only one
 * has been measured. It is a backend-level default, justified by the mechanism rather than by a
 * survey: the butterfly removes three barriers and a per-thread shared-memory round trip while
 * leaving global memory traffic identical, so on a kernel that is bandwidth-bound it is a fixed
 * overhead removal rather than a trade. {@link DeviceCapability#WARP_SHUFFLE} carries a contrary
 * measurement from an RTX 5090 Laptop (Qwen3-1.7B, 141 to 134 tok/s); that measurement is about the
 * wider set of kernels that capability selects, was taken before decode grouped its layer graphs,
 * and is kept as a reason to treat this as a default that a future counter-measurement may narrow —
 * not as a contradiction of it.
 */
// @formatter:on
public final class Fp16GemvReductionPolicy {

    private Fp16GemvReductionPolicy() {}

    /**
     * Whether the decode-shaped FP16 matrix-vector kernels should use their shuffle-reducing
     * variants on the active device.
     *
     * <p>Support is necessary and is asked first, so a device that would compute the wrong answer
     * is never preferred into it.
     */
    public static boolean preferShuffleReduction() {
        return TornadoDevices.current()
                .capabilities()
                .supports(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV);
    }
}
