package org.beehive.jitllm.runtime.backend;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Discovers the {@link DeviceResolver}, once. Same policy as {@code KvStorageFactories}, for the
 * same reason ['s sibling on the resolution side]: at most one, or a named configuration error — no
 * priority, no fallback chain, no registry.
 *
 * <p><b>Unlike {@code KvStorageFactories}, zero is a valid answer here</b>, not a configuration
 * error: a {@code backend.cpu}-only build registers no {@link DeviceResolver} at all, and that is
 * the normal shape of a CPU-only build, not something lost from a shaded jar. Absence only becomes
 * an error at the caller that needed a resolver and did not get one — see {@code
 * LocalModels#verifyAcceleratorHonoured}.
 */
public final class DeviceResolvers {

    private DeviceResolvers() {}

    /** The one discovered resolver, or empty when none is registered. */
    public static Optional<DeviceResolver> discovered() {
        return fromDiscovered(Holder.RESOLVERS);
    }

    /**
     * The selection policy, decoupled from {@code ServiceLoader} I/O so it is testable with a fixed
     * list — no test needs to install a competing {@code META-INF/services} provider to exercise
     * the duplicate-rejection path deterministically.
     *
     * @throws IllegalStateException if more than one resolver is given
     */
    static Optional<DeviceResolver> fromDiscovered(List<DeviceResolver> resolvers) {
        if (resolvers.size() > 1) {
            throw new IllegalStateException(
                    "more than one device resolver was discovered: "
                            + resolvers.stream().map(r -> r.getClass().getName()).toList()
                            + ". Exactly one backend must own device resolution; there is no priority order"
                            + " to break the tie with, deliberately");
        }
        return resolvers.isEmpty() ? Optional.empty() : Optional.of(resolvers.get(0));
    }

    /** Visible for tests. */
    static List<DeviceResolver> loaded() {
        return Holder.RESOLVERS;
    }

    private static final class Holder {
        private static final List<DeviceResolver> RESOLVERS = load();

        private static List<DeviceResolver> load() {
            List<DeviceResolver> resolvers = new ArrayList<>();
            ServiceLoader.load(DeviceResolver.class).forEach(resolvers::add);
            // Deterministic by implementation class name, so a diagnostic listing them reads the
            // same on every run.
            resolvers.sort(Comparator.comparing(r -> r.getClass().getName()));
            return List.copyOf(resolvers);
        }
    }
}
