package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Finds the plan provider for an architecture — discovery, index and validation only.
 *
 * <p>Holds no family name, for the reason {@code TornadoBackendSupport} holds none: a table of
 * families is a switch with different syntax, and the addition test is written to catch either.
 *
 * <p>{@link #create} returns {@code null} for an architecture with no provider, which is how {@code
 * ForwardPlanFactory} migrates one slice at a time without the matrix changing under it. That is
 * deliberately a different answer from "unsupported": a family whose provider exists and whose mode
 * is not supported gets the existing error, unchanged.
 */
public final class TornadoPlanRegistry {

    private TornadoPlanRegistry() {}

    public static List<TornadoPlanProvider> discover() {
        return discover(Thread.currentThread().getContextClassLoader());
    }

    public static List<TornadoPlanProvider> discover(ClassLoader classLoader) {
        List<TornadoPlanProvider> providers = new ArrayList<>();
        ServiceLoader.load(TornadoPlanProvider.class, classLoader).forEach(providers::add);
        providers.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        return providers;
    }

    /**
     * @throws IllegalStateException if two providers claim one identity, naming both classes
     */
    static Map<ArchitectureId, TornadoPlanProvider> index(List<TornadoPlanProvider> providers) {
        Map<ArchitectureId, TornadoPlanProvider> byId = new LinkedHashMap<>();
        for (TornadoPlanProvider provider : providers) {
            TornadoPlanProvider previous = byId.put(provider.architecture(), provider);
            if (previous != null) {
                List<String> both =
                        new ArrayList<>(
                                List.of(
                                        previous.getClass().getName(),
                                        provider.getClass().getName()));
                both.sort(Comparator.naturalOrder());
                throw new IllegalStateException(
                        "Two plan providers claim '"
                                + provider.architecture()
                                + "' on the tornado backend: "
                                + String.join(", ", both)
                                + ". Exactly one must.");
            }
        }
        return byId;
    }

    private static final class Index {
        private static final Map<ArchitectureId, TornadoPlanProvider> BY_ID = index(discover());
    }

    /** The registered identities, for a message that must say what is available. */
    public static String registeredNames() {
        return Index.BY_ID.isEmpty()
                ? "nothing"
                : Index.BY_ID.keySet().stream()
                        .map(ArchitectureId::name)
                        .sorted()
                        .collect(java.util.stream.Collectors.joining(", "));
    }

    /**
     * What representation each tensor of a file of this architecture will occupy on the device: the
     * provider's own {@code deviceRetention}, which by default keeps its {@code nativeTensorTypes}.
     * That is separate from {@code supportedDataTypes} because that one admits a plan for a
     * model-wide representation, and a mixed model holds several: reading admission as retention
     * under-predicts every tensor whose representation is not the model's.
     *
     * <p>Read by the memory preflight, which would otherwise predict every quantized weight at its
     * Q8_0 size and refuse a configuration that fits. An architecture with no provider retains
     * nothing, which is the right answer for it: nothing will build it a plan either.
     */
    public static DeviceRetention deviceRetention(
            ArchitectureId architecture,
            Map<String, DataType> sourceTypes,
            int layers,
            BackendId backend) {
        TornadoPlanProvider provider = Index.BY_ID.get(architecture);
        return provider == null
                ? DeviceRetention.converting()
                : provider.deviceRetention(sourceTypes, layers, backend);
    }

    /** The provider registered for {@code architecture}, if any. */
    public static Optional<TornadoPlanProvider> provider(ArchitectureId architecture) {
        return Optional.ofNullable(Index.BY_ID.get(architecture));
    }

    /** Which architectures have migrated to a registered plan provider. */
    public static java.util.Set<ArchitectureId> registered() {
        return new java.util.LinkedHashSet<>(Index.BY_ID.keySet());
    }

    /**
     * The plan for a migrated architecture.
     *
     * <p>It does <b>not</b> mean "unsupported". A registered provider that does not support the
     * dtype or the mode throws the named error here rather than returning empty — falling through
     * to the family switch would have made a deliberate refusal indistinguishable from a family
     * nobody had migrated, and the switch would then have answered for a family that had already
     * moved. That is the Qwen2-MoE shape: registered, {@code Q8_0} only, and {@code F16} is a
     * refusal rather than a gap.
     *
     * <p>The unsupported messages are the factory's own, word for word, so a caller cannot tell
     * whether a family has migrated by reading its error.
     */
    static Optional<ForwardPlan> create(
            DataType quantization, ExecutionMode mode, State state, Model model) {
        return components(quantization, mode, state, model)
                .map(
                        components ->
                                switch (mode) {
                                    case STANDARD ->
                                            new SingleTokenForwardPlan(
                                                    model, components, state.workspace.prepacker);
                                    case PREFILL_DECODE ->
                                            new PrefillDecodeForwardPlan(
                                                    model,
                                                    (PrefillDecodeForwardPlanComponents) components,
                                                    state.workspace.prepacker);
                                    case BATCH_PREFILL_DECODE ->
                                            new BatchPrefillDecodeForwardPlan(
                                                    model,
                                                    (BatchPrefillDecodeForwardPlanComponents)
                                                            components,
                                                    state.executionPolicy().prefillBatchSize(),
                                                    state.workspace.prepacker);
                                });
    }

    /**
     * Where a native-library path exists on {@code backend}, for a message that must say so: "llama
     * Q4_0, ...".
     */
    public static String nativeLibraryPaths(BackendId backend) {
        return Index.BY_ID.values().stream()
                .filter(provider -> !provider.nativeLibraryWeights(backend).isEmpty())
                .map(
                        provider ->
                                provider.architecture().name()
                                        + " "
                                        + provider.nativeLibraryWeights(backend).stream()
                                                .sorted()
                                                .map(DataType::name)
                                                .collect(java.util.stream.Collectors.joining("/")))
                .sorted()
                .collect(java.util.stream.Collectors.joining(", "));
    }

    /** The provider registered for the architecture named {@code architecture}, if any. */
    public static Optional<TornadoPlanProvider> provider(String architecture) {
        return Optional.ofNullable(Index.BY_ID.get(ArchitectureId.of(architecture)));
    }

    /** The provider registered for {@code model}'s architecture, if it has migrated. */
    static Optional<TornadoPlanProvider> provider(Model model) {
        return Optional.ofNullable(Index.BY_ID.get(model.architectureId()));
    }

    /**
     * Why {@code model}'s provider declines {@code weights} or {@code mode}, or empty when it
     * supports both or no provider is registered. Needs no state, so admission asks it at load.
     */
    public static Optional<String> unsupported(Model model, DataType weights, ExecutionMode mode) {
        TornadoPlanProvider provider = Index.BY_ID.get(model.architectureId());
        if (provider == null) {
            return Optional.empty();
        }
        if (!provider.supportedDataTypes().contains(weights)) {
            return Optional.of(weights + " not supported for model: " + model.getModelType());
        }
        if (!provider.supportedModes().contains(mode)) {
            return Optional.of(
                    mode + " not yet supported for " + model.getModelType() + " + " + weights);
        }
        return Optional.empty();
    }

    /**
     * The components for a migrated architecture in {@code mode}, admitted as {@link #create}
     * admits them: an unsupported representation or mode is refused by name.
     */
    static Optional<SingleTokenForwardPlanComponents> components(
            DataType quantization, ExecutionMode mode, State state, Model model) {
        TornadoPlanProvider provider = Index.BY_ID.get(model.architectureId());
        if (provider == null) {
            return Optional.empty();
        }
        unsupported(model, quantization, mode)
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(reason);
                        });

        // Decided before the components build their layer graphs, which read it.
        state.workspace.prepacking =
                model.weights() instanceof TornadoWeights weights
                        ? WeightPrepacking.forPlan(provider, weights, state.executionPolicy())
                        : WeightPrepacking.NONE;
        SingleTokenForwardPlanComponents components =
                provider.components(quantization, state, model);
        // A provider may support a mode for one representation and not for another — Llama has
        // prefill and batch kernels for Q8_0 and F16 but only single-token ones for Q4_0. Without
        // this the plan's cast fails with a ClassCastException naming two internal interfaces,
        // where the contract of this method is that an unsupported combination is refused by name.
        if (mode == ExecutionMode.PREFILL_DECODE
                        && !(components instanceof PrefillDecodeForwardPlanComponents)
                || mode == ExecutionMode.BATCH_PREFILL_DECODE
                        && !(components instanceof BatchPrefillDecodeForwardPlanComponents)) {
            throw new UnsupportedOperationException(
                    mode + " not yet supported for " + model.getModelType() + " + " + quantization);
        }
        return Optional.of(components);
    }
}
