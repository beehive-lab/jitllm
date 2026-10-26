package org.beehive.jitllm.api;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Set;
import java.util.stream.Collectors;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.memory.TornadoMemoryModel;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanProvider;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Builds a {@link MemoryPlan} from a model file, and refuses a load that cannot fit.
 *
 * <p>Package-private machinery behind {@link LocalModels#preflight}. It exists as its own type so
 * the facade stays a facade: this is where the backend is asked, and {@code LocalModels} does not
 * name {@code TornadoMemoryModel} or a device.
 */
final class MemoryPreflight {

    private MemoryPreflight() {}

    /** The predicted plan for this file and these options. Reads descriptors, not tensor data. */
    static MemoryPlan plan(Path modelFile, int contextLength, ModelOptions options)
            throws IOException {
        return plan(modelFile, contextLength, options, options.executionPolicy(), true);
    }

    /**
     * The plan for {@code policy}, with or without the family's batched-prefill workspace: without
     * it, what everything else costs, which the residency of decoded projections is decided
     * against.
     */
    private static MemoryPlan plan(
            Path modelFile,
            int contextLength,
            ModelOptions options,
            ExecutionPolicy policy,
            boolean withWorkspace)
            throws IOException {
        // loadWeights = false, useTornadovm = false: the configuration comes from metadata and no
        // tensor is materialized, on the host or the device. That is what makes this a *pre*flight.
        var model = ModelLoader.loadModel(modelFile, contextLength, false, false);
        Configuration config = model.configuration();
        // Descriptors only. The loader owns the format types (Rule 4 permits it there and forbids
        // them here), and what comes back is a neutral footprint.
        //
        // The footprint has to know which representations this family keeps as they are, or it
        // predicts every quantized weight at its Q8_0 size — nearly double for a 4-bit file, which
        // is the difference between refusing a 27B model on a 24 GB device and running it.
        var weights =
                ModelLoader.weightFootprint(
                        modelFile,
                        org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry
                                .nativeDeviceTypes(model.architectureId()));
        return TornadoMemoryModel.predict(
                weights,
                config,
                policy,
                TornadoDevices.current(),
                configuredBudgetBytes(),
                keyValueReservation(model, modelFile, options),
                withWorkspace
                        ? workspace(model, modelFile, options, policy)
                        : java.util.Optional.empty(),
                packedRepackScratchBytes(modelFile, model.architectureId(), policy));
    }

    /**
     * The scratch a load under {@code policy} repacks its packed projections through, or 0 when it
     * packs none. The family names what its loader packs, and {@link PackedWeights} answers whether
     * packing is on for this policy and device, as it does for the load itself.
     */
    private static long packedRepackScratchBytes(
            Path modelFile, ArchitectureId architecture, ExecutionPolicy policy)
            throws IOException {
        Set<DataType> packed =
                PackedWeights.loading(
                        policy,
                        () ->
                                TornadoPlanRegistry.provider(architecture)
                                        .map(TornadoPlanProvider::packedProjectionTypes)
                                        .orElse(Set.of())
                                        .stream()
                                        .filter(
                                                type ->
                                                        switch (type) {
                                                            case Q8_0 -> PackedWeights.packQ8(true);
                                                            case Q4_0 -> PackedWeights.packQ4(true);
                                                            default -> false;
                                                        })
                                        .collect(Collectors.toSet()));
        return ModelLoader.packedProjectionBytes(modelFile, packed);
    }

    private static org.beehive.jitllm.runtime.memory.KeyValueReservation keyValueReservation(
            org.beehive.jitllm.model.Model model, Path modelFile, ModelOptions options) {
        return new org.beehive.jitllm.runtime.memory.KeyValueReservation(
                options.maxConcurrentSessions(),
                DelegatingModel.attachesSharedPool(
                        model,
                        options.storageOptions(),
                        org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection.mayHandle(
                                model.architectureId(),
                                loadedWeightType(modelFile),
                                options.executionPolicy())),
                org.beehive.jitllm.runtime.memory.KeyValueReservation.BLOCK_SIZE_TOKENS,
                options.storageOptions().usesFp16KeyValueCache());
    }

    /** What the family's batched prefill layers would allocate per session, if anything. */
    private static java.util.Optional<
                    org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace>
            workspace(
                    org.beehive.jitllm.model.Model model,
                    Path modelFile,
                    ModelOptions options,
                    ExecutionPolicy policy) {
        var weightType = loadedWeightType(modelFile);
        if (weightType == null || policy.prefillBatchSize() <= 1) {
            return java.util.Optional.empty();
        }
        return org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry.provider(
                        model.architectureId())
                .flatMap(
                        provider ->
                                provider.batchPrefillWorkspace(
                                        weightType,
                                        model.configuration(),
                                        policy.prefillBatchSize(),
                                        policy,
                                        options.storageOptions().usesFp16KeyValueCache()));
    }

    // @formatter:off
    /**
     * {@code policy} with where its batched prefill keeps decoded projections settled for this
     * load: resident only when one copy per session fits beside everything else predicted,
     * otherwise one per-chunk scratch. The layers then build what was predicted.
     *
     * <p>When even the per-chunk scratch cannot fit a budget the rest of the plan predicts exactly,
     * the load is refused here, before any device allocation. A policy that pins the residency is
     * kept as it is.
     *
     * @throws InsufficientDeviceMemoryException when neither representation fits
     */
    // @formatter:on
    static ExecutionPolicy resolveDecodedProjections(
            Path modelFile, int contextLength, ModelOptions options, ExecutionPolicy policy) {
        try {
            var model = ModelLoader.loadModel(modelFile, contextLength, false, false);
            var workspace = workspace(model, modelFile, options, policy);
            if (workspace.isEmpty() || !workspace.get().decodesProjections()) {
                return policy;
            }
            MemoryPlan others = plan(modelFile, contextLength, options, policy, false);
            long budget = configuredBudgetBytes();
            int sessions = options.maxConcurrentSessions();
            if (budget > 0
                    && others.confidence() == MemoryPlan.Confidence.EXACT
                    && others.predictedBudgetBytes() + workspace.get().perChunkBytes() * sessions
                            > budget) {
                throw new InsufficientDeviceMemoryException(
                        plan(
                                modelFile,
                                contextLength,
                                options,
                                withDecodedProjections(
                                        policy, ExecutionPolicy.DecodedProjections.PER_CHUNK),
                                true));
            }
            if (policy.decodedProjections() != ExecutionPolicy.DecodedProjections.BACKEND) {
                return policy;
            }
            boolean resident =
                    TornadoMemoryModel.residentDecodedProjections(
                            policy,
                            workspace.get(),
                            others.predictedBudgetBytes(),
                            budget,
                            sessions);
            return withDecodedProjections(
                    policy,
                    resident
                            ? ExecutionPolicy.DecodedProjections.RESIDENT
                            : ExecutionPolicy.DecodedProjections.PER_CHUNK);
        } catch (IOException | RuntimeException unpredictable) {
            if (unpredictable instanceof InsufficientDeviceMemoryException refused) {
                throw refused;
            }
            // As the preflight: a prediction that cannot be made leaves the backend's own rule.
            return policy;
        }
    }

    private static ExecutionPolicy withDecodedProjections(
            ExecutionPolicy policy, ExecutionPolicy.DecodedProjections decodedProjections) {
        return ExecutionPolicy.from(policy).decodedProjections(decodedProjections).build();
    }

    /**
     * The weight type a load materializes, when the file's type determines it: F16 and Q8_0 are
     * loaded as they are. Anything else may be converted, so it is left unknown rather than
     * guessed.
     */
    private static org.beehive.jitllm.runtime.tensor.DataType loadedWeightType(Path modelFile) {
        try {
            return switch (org.beehive.jitllm.format.GgufModelFacts.read(modelFile).quant()) {
                case "F16" -> org.beehive.jitllm.runtime.tensor.DataType.F16;
                case "Q8_0" -> org.beehive.jitllm.runtime.tensor.DataType.Q8_0;
                default -> null;
            };
        } catch (IOException | RuntimeException unreadable) {
            return null;
        }
    }

    /**
     * Fails a known-over-capacity load before the first device allocation.
     *
     * <p>Silent when the budget is unknown or the plan's confidence is not exact. **Refusing on an
     * unreliable prediction would be worse than not predicting**: a conservative estimate that
     * happens to exceed the budget would block a load that would in fact have run, and the caller
     * has no way to overrule it. An under-confident plan is reported by {@link
     * LocalModels#preflight}, where a person can read it, rather than enforced here.
     *
     * <p>Except where even its lower bound does not fit: a plan whose buffers, each counted once,
     * already exceed the budget is refused at any known confidence ({@link
     * MemoryPlan#exceedsConfiguredBudgetAtLowerBound}). That is what refuses a Metal load — capped
     * at {@code CONSERVATIVE} — that could only die part-allocated, such as a 7B Q8_0 model whose
     * MLX prefill adds a second form of its projections beside them.
     */
    static void refuseIfOverCapacity(Path modelFile, int contextLength, ModelOptions options) {
        MemoryPlan plan;
        try {
            plan = plan(modelFile, contextLength, options);
        } catch (IOException | RuntimeException e) {
            // A preflight that cannot be computed must not stop a load that might succeed. The
            // load's own error is the better diagnostic in that case.
            return;
        }
        boolean exactOver =
                plan.confidence() == MemoryPlan.Confidence.EXACT && !plan.fitsConfiguredBudget();
        if (!exactOver && !plan.exceedsConfiguredBudgetAtLowerBound()) {
            return;
        }
        throw new InsufficientDeviceMemoryException(plan);
    }

    /**
     * The backend's configured budget, or 0 when it is not set.
     *
     * <p>Read from the same property the backend charges against, so the preflight and the
     * allocator are talking about one number. The budget is per device, so a model split across
     * several devices gets it once per device.
     */
    private static long configuredBudgetBytes() {
        return perDeviceBudgetBytes()
                * org.beehive.jitllm.runtime.backend.DeviceSplit.requestedDeviceCount();
    }

    private static long perDeviceBudgetBytes() {
        String configured = System.getProperty("tornado.device.memory");
        if (configured == null || configured.isBlank()) {
            return 0;
        }
        try {
            String value = configured.trim().toUpperCase(java.util.Locale.ROOT);
            if (value.endsWith("B") && value.length() > 2) {
                int prefix = "KMGTPE".indexOf(value.charAt(value.length() - 2));
                if (prefix >= 0) {
                    long unit = (long) Math.pow(1024, prefix + 1);
                    return Long.parseLong(value.substring(0, value.length() - 2)) * unit;
                }
                return Long.parseLong(value.substring(0, value.length() - 1));
            }
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            // The backend rejects this value too, and its own message is the clearer one.
            return 0;
        }
    }
}
