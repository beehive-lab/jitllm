package org.beehive.jitllm.runtime.backend;

import java.util.Locale;
import java.util.Objects;
import org.beehive.jitllm.api.Experimental;

/**
 * One thing a device can do that changes what is lowered onto it.
 *
 * <p>A value rather than an enum, for {@link BackendId}'s reason: a backend that gains a capability
 * should not require an edit in the layer above it. The constants below are the ones the tree
 * actually branches on today.
 *
 * <p>The bar for adding one, from {@code ProgramCacheKey}: <b>if changing it can change task count,
 * task names, kernels, grid entries or bindings, it belongs here</b> — and therefore in the cache
 * key. A capability that changes nothing observable does not need naming.
 */
@Experimental
public final class DeviceCapability {

    /**
     * Warp/sub-group shuffle reductions ({@code KernelContext.simdShuffleDown}) produce correct
     * results. No backend is granted this today: the OpenCL backend compiles the shuffle and
     * computes the wrong answer, and CUDA — which computes it correctly — was measured slower than
     * the shared-memory variants it would replace. {@code TornadoDevices.capabilitiesOf} carries
     * the numbers. The kernels stay because the decision is per-device and only one device has been
     * measured.
     */
    public static final DeviceCapability WARP_SHUFFLE = of("warp-shuffle");

    /**
     * The FP16 tensor-core kernel family runs: {@code mma.sync m16n8k16 f16->f32} with {@code
     * ldmatrix} and {@code cp.async} staging ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore},
     * {@code asyncCopyToLocal}). CUDA on a device of compute capability 8.0 or newer: {@code
     * cp.async} is Ampere's, and TornadoVM lowers the intrinsics nowhere else. A CUDA device below
     * that keeps the scalar kernels.
     */
    public static final DeviceCapability TENSOR_CORE_MMA = of("tensor-core-mma");

    /**
     * The int8 tensor-core kernel family runs: {@code mma.sync m16n8k32 s8.s8.s32} with the same
     * staging as {@link #TENSOR_CORE_MMA}, plus in-kernel reads of the int32 accumulators. The same
     * device threshold (compute capability 8.0) — granted separately because it is a separate
     * instruction and a separate arithmetic, and a kernel family asks for exactly what it needs.
     */
    public static final DeviceCapability INT8_TENSOR_CORE_MMA = of("int8-tensor-core-mma");

    /**
     * The multi-workgroup split-KV flash-decoding attention kernel JITs. Metal fails to, so Qwen3
     * falls back to the single-workgroup-per-head online-softmax kernel there.
     */
    public static final DeviceCapability SPLIT_KV_ATTENTION = of("split-kv-attention");

    /**
     * Root-mean-square normalization completes in one pass. Where it does not, lowering emits an
     * extra {@code *_rms_finalize} task per block — the same program, a different task set, which
     * is exactly why capabilities are in the cache key.
     */
    public static final DeviceCapability SINGLE_PASS_RMS = of("single-pass-rms");

    /**
     * The device lowers {@code QuantizationUtils.dp4a_packed} to a packed four-way integer
     * dot-product instruction <b>and</b> computes {@code KernelContext.simdShuffleDown} correctly.
     *
     * <p>The Java body of {@code dp4a_packed} is a correct scalar fallback, so a device without
     * this still computes the right answer — it just has no reason to prefer the packed path over
     * the floating-point one.
     *
     * <p><b>The second half of that contract is not incidental.</b> Every packed kernel reduces
     * with a warp-shuffle butterfly rather than a shared-memory tree, so this one grant selects
     * both the instruction and the reduction. It is granted on CUDA alone, where both are verified;
     * the OpenCL backend lowers {@code dp4a_packed} and would compute the dot products correctly,
     * but miscompiles the shuffle, so granting this there would produce wrong answers even though
     * the instruction half holds. A backend that wants the packed arithmetic without the shuffle
     * needs the kernels to carry a shared-memory reduction again — it is not a matter of adding a
     * grant. See {@link #WARP_SHUFFLE}, which is a different question about the floating-point
     * kernels and is granted nowhere.
     */
    public static final DeviceCapability PACKED_INTEGER_DOT = of("packed-integer-dot");

    /**
     * A 32-wide subgroup butterfly reduction over {@code KernelContext.simdShuffleDown} produces
     * correct results for the fused Q/K/V projection kernel family.
     *
     * <p><b>Deliberately narrower than {@link #WARP_SHUFFLE}.</b> That capability is CUDA's
     * shuffle-reduction correctness, verified wrong on OpenCL and never measured on Metal at all —
     * granting it to Metal would silently change every other call site gated on it (Qwen3's GEMV
     * kernel selection among them), none of which this capability's verification covers. This one
     * names exactly what was measured: {@code fusedQKVMatmulXSimd32}'s five-step 32-lane butterfly
     * (shuffle widths 16, 8, 4, 2, 1) against a CPU reference, isolated in its own minimal task
     * graph, with no rounding ambiguity in the inputs — exact agreement, no poisoned output
     * remaining, on Metal (Apple Pro, TornadoVM 5.2.0-jdk21). Not evaluated on OpenCL or CUDA,
     * where {@link #WARP_SHUFFLE} already answers the equivalent question for the kernels gated on
     * it. A capability that changes nothing observable does not need naming — this one selects
     * between {@code fusedQKVMatmulX} (shared-memory reduction, works everywhere) and {@code
     * fusedQKVMatmulXSimd32} (32-lane shuffle reduction) for the QKV projection task, so it belongs
     * here by this file's own bar.
     */
    public static final DeviceCapability SUBGROUP_SHUFFLE_32 = of("subgroup-shuffle-32");

    /**
     * The doubly-rounded FP16 QKV projection is accurate enough on this device.
     *
     * <p>{@code fusedQKVMatmulX} multiplies a packed FP16 weight pair by a packed FP16 activation
     * pair, so the product is rounded to FP16 before the FP32 accumulator sees it — once per term,
     * over a projection row, in a direction that does not cancel. Withheld on OpenCL, where that
     * cost the Llama-shaped FP16 families their CPU parity (worst relative L2 0.0109 against CUDA's
     * 0.0011 for the identical kernel); where it is withheld, the projection widens each pair
     * before multiplying instead.
     *
     * <p><b>Narrower than the name suggests.</b> Other kernels multiply packed pairs and hold
     * parity on OpenCL — {@code fusedRmsNormFFNGateUp}, which Qwen3 shares, is one. What is
     * specific to the QKV projection is that <i>both</i> operands are FP16: the weights, and an
     * activation {@code mapContextWithQuantize} has already rounded to FP16. Read this as a
     * statement about that combination, not about packed arithmetic generally. Metal keeps the
     * packed path and has not been measured against this question.
     */
    // @formatter:off
    /**
     * The shuffle-reducing FP16 matrix-vector kernels compute correct results on this device.
     *
     * <p><b>This is a support claim, not a preference.</b> It says the 32-lane butterfly in {@code
     * fusedRmsNormQKVMatmulWarp}, {@code fusedRmsNormFFNGateUpWarp}, {@code
     * matrixVectorGenericWithResidualSimd32} and {@code matrixVectorGenericSimd32} is lowered and
     * evaluated correctly here — nothing about whether running them is a good idea. Whether to
     * prefer them over their shared-memory twins is a workload question, and it lives in {@link
     * org.beehive.jitllm.backend.tornado.scheduling.Fp16GemvReductionPolicy}, which is what the
     * layers actually branch on.
     *
     * <p>Granted on CUDA, where {@code simdShuffleDown} is correct and where the CPU-parity gates
     * run these kernels against a host reference for every FP16 family. Withheld on OpenCL, whose
     * backend compiles the shuffle and produces wrong answers. Metal makes the same support claim
     * through {@link #SUBGROUP_SHUFFLE_32}, which predates this one and covers its own verified
     * subset.
     *
     * <p><b>Deliberately not {@link #WARP_SHUFFLE}</b>, which conflates the same support question
     * with a preference for a wider set of kernels — including Q8_0 paths that carry a contrary
     * measurement — and which several unrelated call sites branch on. Separating them is what lets
     * this grant be a plain statement of correctness.
     */
    // @formatter:on
    public static final DeviceCapability SHUFFLE_REDUCED_FP16_GEMV =
            of("shuffle-reduced-fp16-gemv");

    public static final DeviceCapability PACKED_HALF2_MATH = of("packed-half2-math");

    private final String name;

    private DeviceCapability(String name) {
        this.name = name;
    }

    public static DeviceCapability of(String name) {
        Objects.requireNonNull(name, "name");
        String canonical = name.trim().toLowerCase(Locale.ROOT);
        if (canonical.isEmpty()) {
            throw new IllegalArgumentException("a capability name must not be blank");
        }
        return new DeviceCapability(canonical);
    }

    public String name() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DeviceCapability that && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return name;
    }
}
