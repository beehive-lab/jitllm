package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/**
 * Where a family's batched prefill ({@code --batch-prefill-size N}) cannot be built, whatever the
 * key/value cache precision.
 *
 * <p>Refused by name before any plan is built, rather than left to fail inside TornadoVM's compiler
 * or its sketcher halfway through the first generation. Each family's plan provider says where its
 * own batched prefill has holes ({@code TornadoPlanProvider.batchPrefillUnsupported}); this class
 * names no family.
 *
 * <p>Single-token decode and sequential prefill/decode run on every backend.
 */
public final class BatchPrefillSupport {

    private BatchPrefillSupport() {}

    /** Why this combination's batched prefill cannot be built, or empty when it can. */
    public static Optional<String> unsupported(Combination c) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(c.architecture())
                .flatMap(provider -> provider.batchPrefillUnsupported(c));
    }

    /** Why {@code model}'s batched prefill cannot be built on the current device, if it cannot. */
    public static Optional<String> unsupportedOnCurrentDevice(Model model) {
        return unsupported(
                new Combination(
                        model.architectureId().toString(),
                        model.weights().dataType(),
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend(),
                        false,
                        TensorCoreSupport.isTensorCoreCapableBackend(),
                        Fp16KeyValueSupport.nvidiaDevice()));
    }

    /**
     * Refuses a batched prefill this configuration cannot build.
     *
     * @throws UnsupportedOperationException naming the combination, the reason, and the way out
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!gpu) {
            return;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, gpu);
        unsupported(combination)
                .or(() -> needsNativeProjections(combination, policy))
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(refusal(combination, reason));
                        });
    }

    /** What the family's batched prefill needs from {@code policy} that it does not ask for. */
    static Optional<String> needsNativeProjections(Combination c, ExecutionPolicy policy) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(c.architecture())
                .flatMap(provider -> provider.batchPrefillNeeds(c, policy));
    }

    /** The refusal: the combination, why, and that it holds for either cache. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "batched prefill is not supported for "
                        + combination
                        + ": "
                        + reason
                        + ". This holds for either key/value cache precision. Drop"
                        + " --batch-prefill-size (the single-token plan runs this model here),"
                        + " or leave prefill batching off in the execution policy");
    }

    /**
     * The families whose batched prefill runs the Metal SIMD-group kernels; see {@link
     * #defaultFor}.
     */
    static final java.util.Set<String> METAL_BATCHED_FAMILIES =
            java.util.Set.of(
                    "qwen3",
                    "llama",
                    "qwen2",
                    "deepseek-r1-distill-qwen",
                    "mistral",
                    "phi3",
                    "granite",
                    "qwen35",
                    "gemma4");

    /** The batched-prefill chunk every accelerator backend runs by default. */
    public static final int DEFAULT_PREFILL_BATCH = 512;

    // @formatter:off
    /**
     * The policy a model runs when its caller did not choose one: batched prefill on every
     * accelerator backend where the family's batched prefill can be built, the given policy
     * otherwise.
     *
     * <p>Every backend prefills in chunks of {@value #DEFAULT_PREFILL_BATCH}. On Metal, the
     * families in {@link #METAL_BATCHED_FAMILIES}, in F16 and Q8_0, prefill an order of magnitude
     * faster in batches than one token at a time on an Apple GPU (M4 Pro, pp512 in chunks of 256:
     * Qwen3-0.6B F16 151 to 4171 tok/s; Llama-3.2-1B F16 40 to 2372, Q8_0 45 to 2075; Qwen3.5-0.8B
     * Q4_0 60 to 2057; Gemma-4-E2B Q8_0 68 to 1067); greedy text is identical. Every chunk reads
     * all the weights once and replays every layer graph, so a wider chunk spreads that fixed cost
     * over more tokens: against 512, chunks of 128 halve prefill on an A100 (Mistral-7B Q8_0 pp512
     * 2343 against 5206 tok/s) and chunks of 256 cost 5-6% on an M4 Pro. Q4_0 is batched by default
     * for llama and qwen35 only. Other families keep single-token prefill on Metal until their
     * batched kernels are tuned there. A model split across devices ({@code --devices}) keeps
     * single-token prefill: not every family splits its batched prefill. Only a single-token policy
     * is changed, so an explicit choice is never overridden.
     */
    // @formatter:on
    public static ExecutionPolicy defaultFor(Model model, ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.SINGLE_TOKEN
                || org.beehive.jitllm.runtime.backend.DeviceSplit.fromSystemProperties()
                        .isPresent()) {
            return policy;
        }
        BackendId backend =
                org.beehive.jitllm.backend.tornado.device.TornadoDevices.current().backend();
        if (BackendId.CPU.equals(backend)) {
            return policy;
        }
        boolean metal = BackendId.METAL.equals(backend);
        if (metal && !metalBatchedByDefault(model)) {
            return policy;
        }
        if (unsupportedOnCurrentDevice(model).isPresent()) {
            return policy;
        }
        return ExecutionPolicy.from(policy)
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(DEFAULT_PREFILL_BATCH)
                .build();
    }

    /** Whether Metal's batched kernels are tuned for this family and weight type. */
    private static boolean metalBatchedByDefault(Model model) {
        if (!METAL_BATCHED_FAMILIES.contains(model.architectureId().toString())) {
            return false;
        }
        var type = model.weights().dataType();
        boolean q4Batched =
                type == org.beehive.jitllm.runtime.tensor.DataType.Q4_0
                        && java.util.Set.of("llama", "qwen35")
                                .contains(model.architectureId().toString());
        return type == org.beehive.jitllm.runtime.tensor.DataType.F16
                || type == org.beehive.jitllm.runtime.tensor.DataType.Q8_0
                || q4Batched;
    }
}
