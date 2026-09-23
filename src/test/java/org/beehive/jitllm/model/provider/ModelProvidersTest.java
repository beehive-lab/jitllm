package org.beehive.jitllm.model.provider;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.junit.Test;

public class ModelProvidersTest {

    private static ModelSource source(String architecture) {
        return ModelSource.ofMetadata(
                Path.of("synthetic.gguf"),
                Map.of(
                        "general.architecture",
                        architecture,
                        "general.name",
                        architecture + " test"));
    }

    /** A provider is added by adding a file on the classpath, not by editing a list. */
    @Test
    public void providersAreDiscoveredFromTheClasspath() {
        List<ModelProvider> discovered = ModelProviders.discover();
        assertTrue(
                "the test providers must be discovered: " + discovered,
                discovered.stream().anyMatch(p -> p instanceof TestProviders.FixtureProvider));
        assertTrue(
                discovered.stream().anyMatch(p -> p instanceof TestProviders.OtherFixtureProvider));
    }

    /** Ordering is by class name, so it does not depend on how the jar was assembled. */
    @Test
    public void discoveryOrderIsStable() {
        List<String> first = ModelProviders.discover().stream().map(ModelProvider::name).toList();
        List<String> second = ModelProviders.discover().stream().map(ModelProvider::name).toList();
        assertEquals(first, second);
    }

    @Test
    public void supportsSelectsTheProviderForTheSource() {
        ModelProvider selected = ModelProviders.select(source("fixture"));
        assertTrue(selected.toString(), selected instanceof TestProviders.FixtureProvider);
        assertEquals(ArchitectureId.of("fixture"), selected.architecture(source("fixture")));

        assertTrue(
                ModelProviders.select(source("other-fixture"))
                        instanceof TestProviders.OtherFixtureProvider);
    }

    /** An unrecognized model is an error that says what it saw, not a wrong-family load. */
    @Test
    public void anUnrecognizedSourceFailsWithWhatItSaw() {
        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> ModelProviders.select(source("no-such-architecture")));
        assertTrue(failure.getMessage(), failure.getMessage().contains("no-such-architecture"));
    }

    /**
     * Two providers claiming one source is a defect in their checks. Resolving it by order would
     * make the wrong-family load depend on how the classpath was built — reproducible for one
     * person and not another.
     */
    @Test
    public void ambiguityIsRefusedRatherThanResolvedByOrder() {
        ModelProvider greedy =
                new ModelProvider() {
                    @Override
                    public boolean supports(ModelSource s) {
                        return true;
                    }

                    @Override
                    public ArchitectureId architecture(ModelSource s) {
                        return ArchitectureId.of("greedy");
                    }

                    @Override
                    public Model load(
                            ModelSource s, org.beehive.jitllm.runtime.backend.BackendId b, int c) {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public String name() {
                        return "GreedyProvider";
                    }
                };

        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                ModelProviders.select(
                                        source("fixture"),
                                        List.of(new TestProviders.FixtureProvider(), greedy)));
        assertTrue(failure.getMessage(), failure.getMessage().contains("FixtureProvider"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("GreedyProvider"));
    }

    /** Recognition sees metadata, not tensors — a metadata-only source proves it cannot cheat. */
    @Test
    public void recognitionNeedsNoFile() {
        ModelSource metadataOnly = source("fixture");
        assertFalse(metadataOnly.isLoadable());
        assertTrue(new TestProviders.FixtureProvider().supports(metadataOnly));
        assertThrows(IllegalStateException.class, metadataOnly::gguf);
    }

    /**
     * It existed only because {@code BackendId} did not: inventing a placeholder would have put a
     * name in an SPI signature before the thing it names was designed (D4). The type exists now, so
     * the adapter does not.
     */
    @Test
    public void theProviderSpiTakesABackendIdentityAndAContextLength() {
        java.lang.reflect.Method load =
                java.util.Arrays.stream(ModelProvider.class.getMethods())
                        .filter(m -> m.getName().equals("load"))
                        .findFirst()
                        .orElseThrow();

        assertEquals(3, load.getParameterCount());
        assertEquals(
                org.beehive.jitllm.runtime.backend.BackendId.class, load.getParameterTypes()[1]);
        assertEquals(
                "a negative context length still means the model's own",
                int.class,
                load.getParameterTypes()[2]);

        assertTrue(
                "LoadTarget must be gone, not merely unused",
                java.util.Arrays.stream(ModelProvider.class.getMethods())
                        .flatMap(m -> java.util.Arrays.stream(m.getParameterTypes()))
                        .noneMatch(t -> t.getSimpleName().equals("LoadTarget")));
    }

    @Test
    public void anArchitectureIdentityIsAValueAndIsNormalized() {
        assertEquals(ArchitectureId.of("llama"), ArchitectureId.of("LLaMA"));
        assertEquals("llama", ArchitectureId.of(" llama ").name());
        assertThrows(IllegalArgumentException.class, () -> ArchitectureId.of("  "));
    }
}
