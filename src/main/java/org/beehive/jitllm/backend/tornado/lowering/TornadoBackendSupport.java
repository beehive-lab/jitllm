package org.beehive.jitllm.backend.tornado.lowering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Which architectures this backend lowers — <b>discovery, index and validation, and nothing
 * else</b>.
 *
 * <h2>No architecture names live here</h2>
 *
 * <p>Providers are sorted by class name before indexing, and the index refuses duplicates outright,
 * so neither a resolution nor an error message can depend on the order the classpath happened to
 * offer them in. That is not a nicety: a wrong-architecture lowering produces plausible output
 * rather than a crash, so "it worked on my build" is exactly the failure mode to design out.
 *
 * <p>What this class does <b>not</b> say is what a caller should do when an answer is "no". Falling
 * back to the legacy plan is selection policy, and it lives in {@link LoweredPlanSelection} while
 * that path exists.
 */
public final class TornadoBackendSupport {

    /** This backend's name, for messages that must say which backend declined. */
    public static final String BACKEND = "tornado";

    private TornadoBackendSupport() {}

    /** Every provider on the classpath, in a stable order. */
    public static List<TornadoLoweringProvider> discover() {
        return discover(Thread.currentThread().getContextClassLoader());
    }

    public static List<TornadoLoweringProvider> discover(ClassLoader classLoader) {
        List<TornadoLoweringProvider> providers = new ArrayList<>();
        ServiceLoader.load(TornadoLoweringProvider.class, classLoader).forEach(providers::add);
        providers.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        return providers;
    }

    /**
     * Indexes providers by identity.
     *
     * @throws IllegalStateException if two providers claim one identity, naming <b>both classes</b>
     */
    static Map<ArchitectureId, TornadoLoweringProvider> index(
            List<TornadoLoweringProvider> providers) {
        Map<ArchitectureId, TornadoLoweringProvider> byId = new LinkedHashMap<>();
        for (TornadoLoweringProvider provider : providers) {
            TornadoLoweringProvider previous = byId.put(provider.architecture(), provider);
            if (previous != null) {
                // Sorted by class name so this message is the same whichever order they arrived in.
                List<String> both =
                        new ArrayList<>(
                                List.of(
                                        previous.getClass().getName(),
                                        provider.getClass().getName()));
                both.sort(Comparator.naturalOrder());
                throw new IllegalStateException(
                        DiagnosticCode.PROVIDER_DUPLICATE.prefix()
                                + "Two lowering providers claim '"
                                + provider.architecture()
                                + "' on the "
                                + BACKEND
                                + " backend: "
                                + String.join(", ", both)
                                + ". Exactly one must; an identity is how everything downstream refers to a"
                                + " lowering, so two answers is not something to resolve by ordering.");
            }
        }
        return byId;
    }

    /** Discovered once: {@code ServiceLoader} walks the classpath, and the answer cannot change. */
    private static final class Index {
        private static final Map<ArchitectureId, TornadoLoweringProvider> BY_ID = index(discover());
    }

    /** Whether this backend can run the triple. */
    public static boolean supports(ArchitectureId id, DataType weights, ExecutionMode mode) {
        return supports(Index.BY_ID, id, weights, mode);
    }

    static boolean supports(
            Map<ArchitectureId, TornadoLoweringProvider> index,
            ArchitectureId id,
            DataType weights,
            ExecutionMode mode) {
        TornadoLoweringProvider provider = index.get(id);
        return provider != null
                && provider.supportedDataTypes().contains(weights)
                && provider.supportedModes().contains(mode);
    }

    /**
     * The lowering for a triple this backend supports.
     *
     * @throws UnsupportedOperationException naming <b>architecture, dtype, mode and backend</b> —
     *     all four, because "unsupported" without them sends the reader looking in the wrong place,
     *     and the three facts are not interchangeable
     */
    public static FamilyLowering lowering(
            ArchitectureId id,
            DataType weights,
            ExecutionMode mode,
            CompileOptions options,
            DeviceCapabilities capabilities) {
        return lowering(Index.BY_ID, id, weights, mode, options, capabilities);
    }

    static FamilyLowering lowering(
            Map<ArchitectureId, TornadoLoweringProvider> index,
            ArchitectureId id,
            DataType weights,
            ExecutionMode mode,
            CompileOptions options,
            DeviceCapabilities capabilities) {
        TornadoLoweringProvider provider = index.get(id);
        if (provider == null) {
            throw new UnsupportedOperationException(
                    "the "
                            + BACKEND
                            + " backend has no lowering for"
                            + " architecture '"
                            + id
                            + "' ("
                            + mode
                            + ", "
                            + weights
                            + "). It lowers: "
                            + names(index.keySet())
                            + ".");
        }
        if (!provider.supportedDataTypes().contains(weights)) {
            throw new UnsupportedOperationException(
                    weights
                            + " not supported for '"
                            + id
                            + "' + "
                            + mode
                            + " on the "
                            + BACKEND
                            + " backend; it lowers "
                            + sorted(provider.supportedDataTypes()));
        }
        if (!provider.supportedModes().contains(mode)) {
            throw new UnsupportedOperationException(
                    mode
                            + " not supported for '"
                            + id
                            + "' + "
                            + weights
                            + " on the "
                            + BACKEND
                            + " backend; it lowers "
                            + sorted(provider.supportedModes()));
        }
        return provider.create(options, capabilities);
    }

    /** Which architectures this backend lowers. */
    public static Set<ArchitectureId> registered() {
        return new java.util.LinkedHashSet<>(Index.BY_ID.keySet());
    }

    private static String names(Set<ArchitectureId> ids) {
        return ids.isEmpty()
                ? "nothing"
                : ids.stream().map(ArchitectureId::name).sorted().collect(Collectors.joining(", "));
    }

    private static String sorted(Set<? extends Enum<?>> values) {
        return new TreeSet<>(values.stream().map(Enum::name).toList())
                .stream().collect(Collectors.joining(", "));
    }
}
