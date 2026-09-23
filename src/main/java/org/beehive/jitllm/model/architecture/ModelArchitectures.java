package org.beehive.jllm.model.architecture;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.stream.Collectors;
import org.beehive.jllm.runtime.model.ArchitectureId;

/**
 * Finds the architecture computation for an identity.
 *
 * <p>Discovery is {@link ServiceLoader}, so an architecture is added by adding a file and a service
 * line — <b>never by editing a list here</b>. This class holds no family names, and a test asserts
 * that adding one requires touching no shared file.
 *
 * <p>Sorted by class name before use, so the order two implementations are seen in is a property of
 * their names rather than of how the jar was built. Nothing should depend on that order — the
 * duplicate check below exists precisely so nothing can.
 */
public final class ModelArchitectures {

    private ModelArchitectures() {}

    /** Every architecture on the classpath, in a stable order. */
    public static List<ModelArchitecture> discover() {
        return discover(Thread.currentThread().getContextClassLoader());
    }

    public static List<ModelArchitecture> discover(ClassLoader classLoader) {
        List<ModelArchitecture> architectures = new ArrayList<>();
        ServiceLoader.load(ModelArchitecture.class, classLoader).forEach(architectures::add);
        architectures.sort(Comparator.comparing(a -> a.getClass().getName()));
        return architectures;
    }

    /**
     * The one architecture that computes {@code id}.
     *
     * <p><b>Both failures are deterministic and name what is wrong.</b> Two implementations
     * claiming one identity fails naming <i>both classes</i> — not whichever the classpath offered
     * first, because a silently-picked architecture computes plausible nonsense rather than
     * crashing. A missing one fails by identity, listing what is registered, because "no
     * architecture for qwen2-moe" is actionable and "null" is not.
     *
     * @throws IllegalStateException if no architecture, or more than one, claims the identity
     */
    public static ModelArchitecture select(
            ArchitectureId id, List<ModelArchitecture> architectures) {
        Map<ArchitectureId, List<ModelArchitecture>> byId = new LinkedHashMap<>();
        for (ModelArchitecture architecture : architectures) {
            byId.computeIfAbsent(architecture.id(), key -> new ArrayList<>()).add(architecture);
        }

        for (var entry : byId.entrySet()) {
            if (entry.getValue().size() > 1) {
                throw new IllegalStateException(
                        "Several architectures claim '"
                                + entry.getKey()
                                + "': "
                                + entry.getValue().stream()
                                        .map(a -> a.getClass().getName())
                                        .collect(Collectors.joining(", "))
                                + ". Exactly one must; an identity is how everything downstream refers to a"
                                + " computation, so two answers is not something to resolve by ordering.");
            }
        }

        List<ModelArchitecture> claiming = byId.getOrDefault(id, List.of());
        if (claiming.isEmpty()) {
            throw new IllegalStateException(
                    "No architecture computes '"
                            + id
                            + "'. Registered: "
                            + (byId.isEmpty()
                                    ? "none"
                                    : byId.keySet().stream()
                                            .map(ArchitectureId::name)
                                            .sorted()
                                            .collect(Collectors.joining(", ")))
                            + ". A model can still load and run on the legacy path — this says only that"
                            + " nothing describes it as a program.");
        }
        return claiming.getFirst();
    }

    /** As {@link #select(ArchitectureId, List)}, over everything on the classpath. */
    public static ModelArchitecture select(ArchitectureId id) {
        return select(id, discover());
    }

    /** Whether an architecture computation exists, without throwing when it does not. */
    public static boolean isDescribed(ArchitectureId id, List<ModelArchitecture> architectures) {
        return architectures.stream().anyMatch(a -> a.id().equals(id));
    }
}
