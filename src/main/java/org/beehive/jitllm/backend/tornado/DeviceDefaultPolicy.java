package org.beehive.jitllm.backend.tornado;

import java.util.List;
import java.util.function.Predicate;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;

/**
 * The policy a GPU load runs when the caller left it to the device: batched prefill where the
 * family builds it ({@link BatchPrefillSupport#defaultFor}), on native libraries where they are
 * implemented and installed ({@link NativeLibrarySupport#defaultFor}).
 *
 * <p>One place for every entry point that loads a model for the GPU, the facade and the bench
 * alike, so that both measure the same configuration when given the same flags.
 */
public final class DeviceDefaultPolicy {

    private DeviceDefaultPolicy() {}

    /**
     * Resolves the device's default over {@code chosen}. A default this configuration then refuses
     * is dropped rather than reported: nobody asked for it, and {@code chosen} runs instead.
     *
     * @param model the loaded model
     * @param chosen the policy the caller's flags and properties give
     * @param defaultBatching whether the prefill strategy was left unchosen
     * @param defaultNativeLibraries whether native libraries were neither asked for nor declined
     * @param storage the key/value storage the load uses
     */
    public static ExecutionPolicy resolve(
            Model model,
            ExecutionPolicy chosen,
            boolean defaultBatching,
            boolean defaultNativeLibraries,
            StorageOptions storage) {
        return resolve(
                model, chosen, defaultBatching, defaultNativeLibraries, storage, policy -> true);
    }

    /**
     * {@link #resolve(Model, ExecutionPolicy, boolean, boolean, StorageOptions)}, skipping a
     * candidate {@code fits} says the memory budget cannot hold: the model then runs batched on the
     * JIT kernels rather than running out of memory on the native libraries' copies.
     */
    public static ExecutionPolicy resolve(
            Model model,
            ExecutionPolicy chosen,
            boolean defaultBatching,
            boolean defaultNativeLibraries,
            StorageOptions storage,
            Predicate<ExecutionPolicy> fits) {
        ExecutionPolicy batched =
                defaultBatching ? BatchPrefillSupport.defaultFor(model, chosen) : chosen;
        ExecutionPolicy accelerated =
                defaultNativeLibraries ? NativeLibrarySupport.defaultFor(model, batched) : batched;
        for (ExecutionPolicy candidate : List.of(accelerated, batched)) {
            if (candidate.equals(chosen)) {
                break;
            }
            java.util.Optional<String> refused = refusal(model, candidate, storage);
            if (refused.isEmpty() && !fits.test(candidate)) {
                refused = java.util.Optional.of("predicted not to fit the device memory budget");
            }
            if (refused.isEmpty()) {
                return candidate;
            }
            if (candidate == accelerated
                    && accelerated.nativeLibraries()
                    && !batched.nativeLibraries()) {
                // The native default was on and a plan check refused it: say so, or the run
                // continues on the JIT kernels with nothing to tell the two apart.
                LOGGER.log(
                        System.Logger.Level.INFO,
                        "Native libraries are off for this model: " + refused.get());
            }
        }
        return chosen;
    }

    private static final System.Logger LOGGER =
            System.getLogger(DeviceDefaultPolicy.class.getName());

    /**
     * Why the plan checks every GPU load runs refuse {@code policy}, or empty if they accept it.
     */
    private static java.util.Optional<String> refusal(
            Model model, ExecutionPolicy policy, StorageOptions storage) {
        try {
            PlanSupport.require(model, policy, storage, true);
            return java.util.Optional.empty();
        } catch (UnsupportedOperationException | IllegalArgumentException refused) {
            // The FP16 cache check refuses with IllegalArgumentException, the others with
            // UnsupportedOperationException.
            return java.util.Optional.of(String.valueOf(refused.getMessage()));
        }
    }
}
