package org.beehive.jllm.backend.tornado;

import java.util.Optional;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.policy.StorageOptions;
import org.beehive.jllm.runtime.tensor.DataType;

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
     */
    public record Combination(
            String architecture,
            DataType weights,
            ExecutionMode mode,
            BackendId backend,
            boolean nvidiaScheduler,
            boolean tensorCores) {

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
        if (!BackendId.CUDA.equals(c.backend())) {
            return Optional.of(
                    "the " + c.backend() + " kernels have no verified FP16 key/value path");
        }
        switch (c.architecture()) {
            case "qwen35" -> {
                // FP16 writers and readers in every mode, including batched prefill.
                return Optional.empty();
            }
            case "llama", "qwen3" -> {
                if (!c.nvidiaScheduler()) {
                    return Optional.of("the non-NVIDIA decode layers keep an FP32 cache");
                }
                boolean q8 = c.weights() == DataType.Q8_0;
                boolean f16 = c.weights() == DataType.F16;
                boolean q4Llama = c.weights() == DataType.Q4_0 && c.architecture().equals("llama");
                if (!f16 && !(q8 && c.architecture().equals("llama")) && !q4Llama) {
                    return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
                }
                return switch (c.mode()) {
                    case STANDARD -> Optional.empty();
                    case PREFILL_DECODE ->
                            q8
                                    ? Optional.empty()
                                    : Optional.of(
                                            "the "
                                                    + c.weights()
                                                    + " sequential prefill/decode layers keep an"
                                                    + " FP32 cache");
                    case BATCH_PREFILL_DECODE ->
                            q4Llama
                                    ? Optional.of("Q4_0 has no batched prefill")
                                    : c.tensorCores()
                                            ? Optional.empty()
                                            : Optional.of(
                                                    "batched prefill without tensor-core MMA"
                                                            + " writes an FP32 cache");
                };
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
     * @throws IllegalArgumentException naming the combination, the reason, and the FP32 setting
     */
    public static void require(
            Model model, ExecutionPolicy policy, StorageOptions storage, boolean gpu) {
        if (!storage.usesFp16KeyValueCache() || !gpu) {
            // FP32 is always available, and every CPU family has an FP16 host cache.
            return;
        }
        Combination combination = resolve(model, policy, gpu);
        unsupported(combination)
                .ifPresent(
                        reason -> {
                            throw new IllegalArgumentException(
                                    DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                                            "an FP16 key/value cache (the default) is not supported"
                                                    + " for "
                                                    + combination
                                                    + ": "
                                                    + reason
                                                    + ". Run with --fp32-kv-cache, or load with"
                                                    + " ModelOptions.builder().storageOptions("
                                                    + "StorageOptions.fp32()) from Java"));
                        });
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
                    false);
        }
        return new Combination(
                model.architectureId().toString(),
                model.weights().dataType(),
                mode,
                TornadoDevices.current().backend(),
                SchedulerDetectionService.determineSchedulerType(model) == SchedulerType.NVIDIA,
                TensorCoreSupport.isTensorCoreCapableBackend());
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
