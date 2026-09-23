package org.beehive.jitllm.model.provider;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;
import java.util.stream.Collectors;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * Finds the provider for a source.
 *
 * <p>Discovery is {@link ServiceLoader}, so a provider is added by adding a file, not by editing a
 * list. Selection is the part worth being careful about.
 */
public final class ModelProviders {

    private ModelProviders() {}

    /**
     * Every provider on the classpath, in a stable order — by class name, not by classpath order.
     */
    public static List<ModelProvider> discover() {
        return discover(ModelProviders.class.getClassLoader());
    }

    public static List<ModelProvider> discover(ClassLoader classLoader) {
        List<ModelProvider> providers = new ArrayList<>();
        ServiceLoader.load(ModelProvider.class, classLoader).forEach(providers::add);
        providers.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        return providers;
    }

    /**
     * The one provider that handles {@code source}.
     *
     * <p><b>Ambiguity is an error, not a race.</b> If two providers claim the same source, the load
     * fails naming both, rather than taking whichever the classpath happened to offer first. That
     * is the difference between a bug that reproduces and one that depends on how the jar was built
     * — and a wrong-family load produces plausible nonsense, not a crash, so it is the kind of
     * wrong that goes unnoticed.
     *
     * @throws IllegalStateException if no provider or more than one claims the source
     */
    public static ModelProvider select(ModelSource source, List<ModelProvider> providers) {
        List<ModelProvider> claiming =
                providers.stream().filter(provider -> provider.supports(source)).toList();

        if (claiming.isEmpty()) {
            throw new IllegalStateException(
                    DiagnosticCode.PROVIDER_MISSING.prefix()
                            + "No model provider recognizes "
                            + source
                            + ". general.architecture="
                            + source.metadata().get("general.architecture")
                            + ", general.name="
                            + source.metadata().get("general.name"));
        }
        if (claiming.size() > 1) {
            throw new IllegalStateException(
                    DiagnosticCode.PROVIDER_DUPLICATE.prefix()
                            + "Several providers claim "
                            + source
                            + ": "
                            + claiming.stream()
                                    .map(ModelProvider::name)
                                    .collect(Collectors.joining(", "))
                            + ". Exactly one must; this is a defect in their supports(...) checks, not"
                            + " something to resolve by ordering.");
        }
        return claiming.get(0);
    }

    /** Convenience: discover, then select. */
    public static ModelProvider select(ModelSource source) {
        return select(source, discover());
    }
}
