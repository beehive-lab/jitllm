package org.beehive.jllm.backend.cpu;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;
import org.beehive.jllm.inference.ForwardPass;
import org.beehive.jllm.runtime.model.ArchitectureId;

/**
 * Resolves the host forward pass for an architecture, once.
 *
 * <p>Discovery happens at class initialization — at most once per application class loader — so it
 * is not on any execution path, let alone the per-token one. A resolved {@link ForwardPass} is held
 * by the caller for the life of the model.
 *
 * <h2>Absence is an error, not a fallback</h2>
 *
 * <p>An architecture with no provider is a configuration problem, and the likeliest cause is a
 * shaded jar that lost the service file. Falling back to some default routine would run the wrong
 * arithmetic for the family and produce plausible tokens, which is the worst way for this to fail.
 * Duplicates fail too, naming both: there is no priority order to break the tie with, deliberately.
 */
public final class CpuForwardPasses {

    private CpuForwardPasses() {}

    /**
     * @throws IllegalStateException if no provider serves {@code architecture}, or more than one
     *     does
     */
    public static ForwardPass forArchitecture(ArchitectureId architecture) {
        return select(Holder.PROVIDERS, architecture);
    }

    /**
     * Resolution against a stated provider list.
     *
     * <p>Package-private so the cases the real service file cannot produce — no provider, two
     * providers — are tested against <b>this</b> code rather than a copy of it in a test.
     */
    static ForwardPass select(List<CpuForwardProvider> providers, ArchitectureId architecture) {
        List<CpuForwardProvider> serving = new ArrayList<>(1);
        for (CpuForwardProvider provider : providers) {
            if (provider.architecture().equals(architecture)) {
                serving.add(provider);
            }
        }
        if (serving.size() == 1) {
            return serving.get(0).create();
        }
        if (serving.isEmpty()) {
            throw new IllegalStateException(
                    "no CPU forward pass is registered for architecture '"
                            + architecture
                            + "'; registered: "
                            + architectures(providers)
                            + ". Adding an architecture means adding a "
                            + CpuForwardProvider.class.getName()
                            + " and a service line; an empty list usually means the service file was lost"
                            + " when the jar was shaded");
        }
        throw new IllegalStateException(
                "architecture '"
                        + architecture
                        + "' is served by more than"
                        + " one CPU forward pass: "
                        + names(serving)
                        + ". Exactly one provider must own an"
                        + " architecture; there is no priority order to break the tie with, deliberately");
    }

    /** Visible for tests. */
    static List<CpuForwardProvider> discovered() {
        return Holder.PROVIDERS;
    }

    private static String architectures(List<CpuForwardProvider> providers) {
        return providers.stream()
                .map(p -> p.architecture().toString())
                .sorted()
                .toList()
                .toString();
    }

    private static String names(List<CpuForwardProvider> providers) {
        return providers.stream().map(p -> p.getClass().getName()).toList().toString();
    }

    private static final class Holder {
        private static final List<CpuForwardProvider> PROVIDERS = load();

        private static List<CpuForwardProvider> load() {
            List<CpuForwardProvider> providers = new ArrayList<>();
            ServiceLoader.load(CpuForwardProvider.class).forEach(providers::add);
            // Ordered by implementation class name so a diagnostic listing them, and any tie the
            // duplicate check reports, reads the same on every run and every machine.
            providers.sort(Comparator.comparing(p -> p.getClass().getName()));
            return List.copyOf(providers);
        }
    }
}
