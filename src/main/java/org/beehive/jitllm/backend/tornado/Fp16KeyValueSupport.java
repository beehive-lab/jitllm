package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Which configurations really store and read the key/value cache in half precision.
 *
 * <p>FP16 is the default cache, and a configuration whose kernels do not implement it is refused,
 * never quietly run on FP32. Accepting the setting is not evidence: before this rule existed a
 * dozen paths allocated the FP16 arrays and then wrote and read FP32 ones, reporting FP16 all the
 * same, and one — Q8_0 batched prefill — handed decode an FP16 cache nothing had written.
 *
 * <p>The supported set is what has an FP16 writer and reader on the same path, end to end, and a
 * numerical comparison against FP32 behind it. Everything else is refused with the way out: {@code
 * --fp32-kv-cache}, or {@link StorageOptions#fp32()} from Java.
 */
public final class Fp16KeyValueSupport {

    private Fp16KeyValueSupport() {}

    /**
     * The resolved facts the answer depends on.
     *
     * @param architecture the model family's architecture id
     * @param weights the weight representation as loaded, after any materialization — a Q4_K file
     *     loaded as Q8_0 runs the Q8_0 layers
     * @param mode the execution mode the policy selects
     * @param backend the backend the plan runs on; {@link BackendId#CPU} for the host path
     * @param nvidiaScheduler whether the NVIDIA-class decode layers are selected
     * @param tensorCores whether batched prefill takes the tensor-core MMA layers
     * @param nvidiaDevice whether the device is NVIDIA-class — a device fact, unlike the scheduler,
     *     which also depends on the model (Mistral keeps the non-NVIDIA scheduler)
     */
    public record Combination(
            String architecture,
            DataType weights,
            ExecutionMode mode,
            BackendId backend,
            boolean nvidiaScheduler,
            boolean tensorCores,
            boolean nvidiaDevice) {

        @Override
        public String toString() {
            return architecture + " / " + weights + " / " + mode + " on " + backend;
        }
    }

    /** Why FP16 cannot be used here, or empty when it can. */
    public static Optional<String> unsupported(Combination c) {
        if (BackendId.CPU.equals(c.backend())) {
            // Every family reaches the host cache through FloatTensor, in every mode.
            return Optional.empty();
        }
        if (BackendId.METAL.equals(c.backend())) {
            return metal(c);
        }
        // CUDA, and OpenCL on an NVIDIA-class device, where the FP16 kernels were measured; Metal
        // above.
        boolean verifiedBackend =
                BackendId.CUDA.equals(c.backend())
                        || (BackendId.OPENCL.equals(c.backend()) && c.nvidiaDevice());
        if (!verifiedBackend) {
            return Optional.of(
                    "the "
                            + c.backend()
                            + " kernels have no verified FP16 key/value path"
                            + (BackendId.OPENCL.equals(c.backend())
                                    ? " on this (non-NVIDIA) device"
                                    : ""));
        }
        Optional<String> noBatchedPrefill = BatchPrefillSupport.unsupported(c);
        if (noBatchedPrefill.isPresent()) {
            // Not a key/value question: the plan cannot be built with either cache.
            return noBatchedPrefill;
        }
        return TornadoPlanRegistry.provider(c.architecture())
                .map(provider -> provider.fp16KeyValueUnsupported(c))
                .orElseGet(
                        () ->
                                Optional.of(
                                        "the " + c.architecture() + " layers keep an FP32 cache"));
    }

    /**
     * The FP16 answer of a family whose single-token and batched-prefill layers write the FP16
     * cache and whose sequential prefill/decode plan does not (Qwen2 and its distillations): F16
     * and Q8_0 weights on the NVIDIA-class decode layers.
     */
    public static Optional<String> singleTokenAndBatchedNvidiaLayers(Combination c) {
        return c.mode() == ExecutionMode.BATCH_PREFILL_DECODE
                ? nvidiaDecodeLayers(c, false)
                : singleTokenNvidiaLayers(c);
    }

    // @formatter:off
    /**
     * The FP16 answer of the NVIDIA-class decode layers shared by Llama and Qwen3: F16 and Q8_0
     * weights, and Q4_0 where {@code q4_0} says the family's Q4_0 layers write FP16 too. Every
     * mode, except that Q4_0 has no sequential prefill/decode plan.
     */
    // @formatter:on
    public static Optional<String> nvidiaDecodeLayers(Combination c, boolean q4_0) {
        if (!c.nvidiaScheduler()) {
            return Optional.of("the non-NVIDIA decode layers keep an FP32 cache");
        }
        boolean q8 = c.weights() == DataType.Q8_0;
        boolean f16 = c.weights() == DataType.F16;
        boolean q4 = q4_0 && c.weights() == DataType.Q4_0;
        if (!f16 && !q8 && !q4) {
            return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
        }
        return switch (c.mode()) {
            case STANDARD -> Optional.empty();
            case PREFILL_DECODE ->
                    q4
                            ? Optional.of("Q4_0 has no sequential prefill/decode plan")
                            : Optional.empty();
            case BATCH_PREFILL_DECODE -> Optional.empty();
        };
    }

    /**
     * The FP16 answer of a family with single-token plans only, on the NVIDIA-class decode layers:
     * F16 and Q8_0 weights.
     */
    public static Optional<String> singleTokenNvidiaLayers(Combination c) {
        if (!c.nvidiaScheduler()) {
            return Optional.of("the non-NVIDIA decode layers keep an FP32 cache");
        }
        if (c.weights() != DataType.F16 && c.weights() != DataType.Q8_0) {
            return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
        }
        return c.mode() == ExecutionMode.STANDARD
                ? Optional.empty()
                : Optional.of(c.architecture() + " has single-token plans only");
    }

    // @formatter:off
    /**
     * The Metal rows: each (family, weights, mode) measured against FP32 on an Apple M4 Pro with
     * the {@code GpuFp16Kv*AccelTest} classes. Decode attention over the FP16 cache takes the
     * lane-cooperative split-KV kernel for 128-wide heads (the 64-thread per-key kernel needs more
     * than Metal's 32 KB of threadgroup memory), its 256-wide twin for Qwen3.5, and the per-head
     * flash kernels otherwise; batched prefill takes the FP16 SIMD-group attention kernels. Rows
     * without a model file here stay refused: Gemma 4 Q4_0 and the families' sequential
     * prefill/decode outside Llama, Qwen3 and Qwen3.5.
     */
    // @formatter:on
    private static Optional<String> metal(Combination c) {
        Optional<String> noBatchedPrefill = BatchPrefillSupport.unsupported(c);
        if (noBatchedPrefill.isPresent()) {
            return noBatchedPrefill;
        }
        boolean f16OrQ8 = c.weights() == DataType.F16 || c.weights() == DataType.Q8_0;
        switch (c.architecture()) {
            case "qwen35" -> {
                return c.weights() == DataType.Q4_0 || c.weights() == DataType.Q8_0
                        ? Optional.empty()
                        : Optional.of(
                                "the qwen35 " + c.weights() + " layers are not verified on Metal");
            }
            case "gemma4" -> {
                return c.weights() == DataType.Q8_0
                        ? Optional.empty()
                        : Optional.of(
                                "the gemma4 "
                                        + c.weights()
                                        + " FP16 cache is not verified on Metal");
            }
            case "llama", "qwen3" -> {
                boolean q4Llama = c.weights() == DataType.Q4_0 && c.architecture().equals("llama");
                if (!f16OrQ8 && !q4Llama) {
                    return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
                }
                return q4Llama && c.mode() == ExecutionMode.PREFILL_DECODE
                        ? Optional.of("Q4_0 has no sequential prefill/decode plan")
                        : Optional.empty();
            }
            case "mistral", "qwen2", "deepseek-r1-distill-qwen", "phi3", "granite" -> {
                if (!f16OrQ8) {
                    return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
                }
                return c.mode() == ExecutionMode.PREFILL_DECODE
                        ? Optional.of(
                                c.architecture()
                                        + " has no verified FP16 sequential prefill/decode path on"
                                        + " Metal")
                        : Optional.empty();
            }
            default -> {
                return Optional.of("the " + c.architecture() + " layers keep an FP32 cache");
            }
        }
    }

    /**
     * Refuses an FP16 cache the resolved configuration does not implement.
     *
     * <p>Called before any cache is allocated or plan built, with what is known then: the loaded
     * model, the policy a session will execute, and the storage it asked for.
     *
     * @throws UnsupportedOperationException from {@link BatchPrefillSupport#require} when the
     *     batched-prefill plan cannot be built at all, with either cache, so the FP32 setting would
     *     not help
     * @throws IllegalArgumentException naming the combination, the reason, and the FP32 setting
     */
    public static void require(
            Model model, ExecutionPolicy policy, StorageOptions storage, boolean gpu) {
        if (!storage.usesFp16KeyValueCache() || !gpu) {
            // FP32 is always available, and every CPU family has an FP16 host cache.
            return;
        }
        Combination combination = resolve(model, policy, gpu);
        // A plan that cannot be built at all is refused as that, not as a cache the FP32 setting
        // would rescue.
        BatchPrefillSupport.require(model, policy, gpu);
        unsupported(combination)
                .ifPresent(
                        reason -> {
                            throw new IllegalArgumentException(refusal(combination, reason));
                        });
    }

    /** The refusal: the combination, why, and how to ask for the FP32 cache instead. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "an FP16 key/value cache (the default) is not supported for "
                        + combination
                        + ": "
                        + reason
                        + ". Run with --fp32-kv-cache; from Java, load with"
                        + " ModelOptions.builder().storageOptions(StorageOptions.fp32()), or start"
                        + " the JVM with -D"
                        + StorageOptions.FP32_PROPERTY
                        + "=true");
    }

    /** The combination this model and policy resolve to on the current device. */
    public static Combination resolve(Model model, ExecutionPolicy policy, boolean gpu) {
        ExecutionMode mode = executionMode(policy);
        if (!gpu) {
            return new Combination(
                    model.architectureId().toString(),
                    model.weights().dataType(),
                    mode,
                    BackendId.CPU,
                    false,
                    false,
                    false);
        }
        return new Combination(
                model.architectureId().toString(),
                model.weights().dataType(),
                mode,
                TornadoDevices.current().backend(),
                SchedulerDetectionService.determineSchedulerType(model) == SchedulerType.NVIDIA,
                TensorCoreSupport.isTensorCoreCapableBackend(),
                nvidiaDevice());
    }

    /** Whether the current device is NVIDIA-class, whatever the model. */
    public static boolean nvidiaDevice() {
        return TornadoDevices.target()
                .capabilities()
                .supports(org.beehive.jitllm.runtime.backend.DeviceCapability.SINGLE_PASS_RMS);
    }

    private static ExecutionMode executionMode(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return ExecutionMode.STANDARD;
        }
        return policy.prefillBatchSize() > 1
                ? ExecutionMode.BATCH_PREFILL_DECODE
                : ExecutionMode.PREFILL_DECODE;
    }
}
