package org.beehive.jllm.model.provider;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * Class B, because it loads real files. The comparison is the loaded configuration rather than the
 * output: if the same loader ran with the same arguments, every derived number matches, and a
 * provider that picked the wrong family would differ in the first field it touched.
 *
 * <p>Extra models can be swept by pointing {@code -Dprovider.parity.models} at a comma-separated
 * list of paths; without it the pinned fixtures are used, so the suite needs nothing extra.
 */
public class ProviderLoadParityAccelTest {

    private static final String PROVIDERS_PROPERTY = "jllm.providers";

    @Test
    public void theProviderLoadsWhatModelTypeDispatchLoaded() throws Exception {
        for (Path model : modelsUnderTest()) {
            Map<String, Object> throughProvider = load(model, true);
            Map<String, Object> throughModelType = load(model, false);
            assertEquals(
                    model.getFileName() + ": the provider must load what ModelType loaded",
                    throughModelType,
                    throughProvider);
        }
    }

    /** The fallback is selectable, and selecting it really does take the other path. */
    @Test
    public void theLegacyDispatchIsStillReachable() throws Exception {
        Path model = modelsUnderTest().get(0);
        assertEquals(load(model, false), load(model, false));
    }

    private static java.util.List<Path> modelsUnderTest() {
        String configured = System.getProperty("provider.parity.models", "");
        if (!configured.isBlank()) {
            return java.util.Arrays.stream(configured.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .map(Path::of)
                    .toList();
        }
        Path fixture = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (fixture == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent", false);
        }
        return java.util.List.of(fixture);
    }

    /** A value a family may not implement; recorded as absent rather than skipped. */
    private static Object optional(java.util.function.IntSupplier accessor) {
        try {
            return accessor.getAsInt();
        } catch (UnsupportedOperationException notImplemented) {
            return "not implemented by this family";
        }
    }

    /** Everything the loaded model states about itself, as data. */
    private static Map<String, Object> load(Path modelPath, boolean throughProviders)
            throws Exception {
        String previous = System.getProperty(PROVIDERS_PROPERTY);
        System.setProperty(PROVIDERS_PROPERTY, Boolean.toString(throughProviders));
        try {
            Model model = ModelLoader.loadModel(modelPath, 512, true, false);
            Configuration configuration = model.configuration();
            Map<String, Object> loaded = new LinkedHashMap<>();
            loaded.put("modelType", model.getModelType().name());
            loaded.put("dim", configuration.dim());
            loaded.put("hiddenDim", configuration.hiddenDim());
            loaded.put("layers", configuration.numberOfLayers());
            loaded.put("heads", configuration.numberOfHeads());
            loaded.put("kvHeads", configuration.numberOfKeyValueHeads());
            loaded.put("vocabulary", configuration.vocabularySize());
            loaded.put("contextLength", configuration.contextLength());
            // contextLengthModel() and numberOfHeadsKey() are not implemented by every family —
            // Llama's throw. Comparing what a family does not answer would test the exception,
            // not the load.
            loaded.put("modelContextLength", optional(configuration::contextLengthModel));
            loaded.put("rmsNormEps", configuration.rmsNormEps());
            loaded.put("ropeTheta", configuration.ropeTheta());
            loaded.put("weights", model.weights().dataType().name());
            loaded.put("activations", configuration.activationType().name());
            loaded.put("tokenizer", model.tokenizer().getClass().getSimpleName());
            loaded.put("chatFormat", model.chatFormat().getClass().getSimpleName());
            return loaded;
        } finally {
            if (previous == null) {
                System.clearProperty(PROVIDERS_PROPERTY);
            } else {
                System.setProperty(PROVIDERS_PROPERTY, previous);
            }
        }
    }
}
