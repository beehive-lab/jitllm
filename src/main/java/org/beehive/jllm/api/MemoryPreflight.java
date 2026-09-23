package org.beehive.jllm.api;

import java.io.IOException;
import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.backend.tornado.memory.TornadoMemoryModel;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.memory.MemoryPlan;

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
                        org.beehive.jllm.backend.tornado.plan.TornadoPlanRegistry.nativeDeviceTypes(
                                model.architectureId()));
        return TornadoMemoryModel.predict(
                weights,
                config,
                options.executionPolicy(),
                TornadoDevices.current(),
                configuredBudgetBytes(),
                new org.beehive.jllm.runtime.memory.KeyValueReservation(
                        options.maxConcurrentSessions(),
                        DelegatingModel.attachesSharedPool(
                                model,
                                options.storageOptions(),
                                org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection
                                        .mayHandle(
                                                model.architectureId(),
                                                loadedWeightType(modelFile),
                                                options.executionPolicy())),
                        org.beehive.jllm.runtime.memory.KeyValueReservation.BLOCK_SIZE_TOKENS,
                        options.storageOptions().usesFp16KeyValueCache()));
    }

    /**
     * The weight type a load materializes, when the file's type determines it: F16 and Q8_0 are
     * loaded as they are. Anything else may be converted, so it is left unknown rather than
     * guessed.
     */
    private static org.beehive.jllm.runtime.tensor.DataType loadedWeightType(Path modelFile) {
        try {
            return switch (org.beehive.jllm.format.GgufModelFacts.read(modelFile).quant()) {
                case "F16" -> org.beehive.jllm.runtime.tensor.DataType.F16;
                case "Q8_0" -> org.beehive.jllm.runtime.tensor.DataType.Q8_0;
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
        if (plan.confidence() != MemoryPlan.Confidence.EXACT || plan.fitsConfiguredBudget()) {
            return;
        }
        throw new InsufficientDeviceMemoryException(plan);
    }

    /**
     * The backend's configured budget, or 0 when it is not set.
     *
     * <p>Read from the same property the backend charges against, so the preflight and the
     * allocator are talking about one number.
     */
    private static long configuredBudgetBytes() {
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
