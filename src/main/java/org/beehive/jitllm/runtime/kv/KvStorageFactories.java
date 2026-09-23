package org.beehive.jllm.runtime.kv;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Discovers the {@link KvStorageFactory}, once.
 *
 * <p>Same policy as {@code KvFieldBinders}, for the same reason: exactly one, or an error. No
 * backend priority, no fallback chain, no registry — there is one backend, and a mechanism built
 * for several would be built without a second one to check it against.
 *
 * <p><b>Resolve before allocating.</b> A caller that can fall back on an allocation failure must
 * call {@link #single()} <i>outside</i> its {@code catch}, so that "no backend can make a pool" and
 * "this device has no room for one" stay different answers. They are: the first is a configuration
 * error and the second is capacity.
 */
public final class KvStorageFactories {

    private KvStorageFactories() {}

    /**
     * The one factory.
     *
     * @throws IllegalStateException if none is discovered, or more than one is
     */
    public static KvStorageFactory single() {
        List<KvStorageFactory> factories = Holder.FACTORIES;
        if (factories.size() == 1) {
            return factories.get(0);
        }
        if (factories.isEmpty()) {
            throw new IllegalStateException(
                    "no backend key/value storage factory was discovered,"
                            + " so no shared pool can be allocated. This is a configuration error, not a"
                            + " capacity one: it usually means the "
                            + KvStorageFactory.class.getName()
                            + " service file was lost when the jar was shaded");
        }
        throw new IllegalStateException(
                "more than one backend key/value storage factory was"
                        + " discovered: "
                        + factories.stream().map(f -> f.getClass().getName()).toList()
                        + ". Exactly one backend must own pool allocation; there is no priority order to"
                        + " break the tie with, deliberately");
    }

    /** Visible for tests. */
    static List<KvStorageFactory> discovered() {
        return Holder.FACTORIES;
    }

    private static final class Holder {
        private static final List<KvStorageFactory> FACTORIES = load();

        private static List<KvStorageFactory> load() {
            List<KvStorageFactory> factories = new ArrayList<>();
            ServiceLoader.load(KvStorageFactory.class).forEach(factories::add);
            // Deterministic by implementation class name, so a diagnostic listing them reads the
            // same on every run.
            factories.sort(Comparator.comparing(f -> f.getClass().getName()));
            return List.copyOf(factories);
        }
    }
}
