package org.beehive.jitllm.model.provider;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.format.ModelSource;
import org.junit.Test;

/**
 * The metadata below is what the local corpus actually contains, which is the point: Mistral ships
 * as {@code arch=llama} and the DeepSeek distill as {@code arch=qwen2}, so an architecture-only
 * rule would load two families through the wrong loader, and a name-only rule would depend on words
 * in a field anyone can set.
 */
public class FamilyRecognitionTest {

    private static ModelSource source(String architecture, String name) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (architecture != null) {
            metadata.put("general.architecture", architecture);
        }
        if (name != null) {
            metadata.put("general.name", name);
        }
        return ModelSource.ofMetadata(Path.of("m.gguf"), metadata);
    }

    private static List<ModelProvider> families() {
        return ModelProviders.discover().stream()
                .filter(
                        provider ->
                                !(provider instanceof TestProviders.FixtureProvider)
                                        && !(provider
                                                instanceof TestProviders.OtherFixtureProvider))
                .toList();
    }

    private static String architectureOf(ModelSource source) {
        return ModelProviders.select(source, families()).architecture(source).name();
    }

    /** Exactly the metadata pairs the corpus carries. */
    @Test
    public void everyModelInTheCorpusIsRecognized() {
        assertEquals("llama", architectureOf(source("llama", "Llama 3.2 1B Instruct")));
        assertEquals(
                "mistral",
                architectureOf(source("llama", "models--mistralai--Mistral-7B-Instruct-v0.3")));
        assertEquals("qwen2", architectureOf(source("qwen2", "Qwen2.5 0.5B Instruct")));
        assertEquals("qwen3", architectureOf(source("qwen3", "Qwen3 0.6B")));
        assertEquals("phi3", architectureOf(source("phi3", "Phi3")));
        assertEquals("granite", architectureOf(source("granite", "Granite 3.2 2b Instruct")));
        assertEquals(
                "deepseek-r1-distill-qwen",
                architectureOf(source("qwen2", "DeepSeek R1 Distill Qwen 1.5B")));
    }

    /**
     * The declared architecture decides first. A doctored name cannot move a file to another family
     * — which is what the old substring chain allowed, since it never read the architecture at all.
     */
    @Test
    public void aDoctoredNameCannotChangeTheArchitecture() {
        assertEquals("qwen3", architectureOf(source("qwen3", "Llama 3.2 1B Instruct")));
        assertEquals("granite", architectureOf(source("granite", "mistral phi3 qwen2")));
        assertEquals("llama", architectureOf(source("llama", "no idea what this is")));
    }

    /** Within one declared architecture, the name is the only discriminator — and it is used. */
    @Test
    public void theNameStillSeparatesFamiliesThatShareAnArchitecture() {
        assertEquals("devstral", architectureOf(source("llama", "Devstral Small 2507")));
        assertEquals("mistral", architectureOf(source("llama", "Mistral 7B Instruct v0.3")));
        assertEquals("llama", architectureOf(source("llama", "Llama 3.1 8B Instruct")));
        assertEquals("qwen2", architectureOf(source("qwen2", "Qwen2.5 1.5B Instruct")));
    }

    /**
     * Metal parity task 12 — Devstral's newer builds declare {@code mistral3}, not {@code llama}.
     *
     * <p>The metadata pair here is verbatim from the real fixture ({@code
     * Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf}: {@code general.architecture=mistral3},
     * {@code general.name=Devstral-Small-2-24B-Instruct-2512}), which reached no provider at all
     * and failed with {@code [GPUL-MO2]} until this case existed — even though {@code
     * DevstralModelLoader} already reads the {@code mistral3.*} block, YaRN scaling included.
     */
    @Test
    public void devstralIsRecognizedWhenItDeclaresMistral3() {
        assertEquals(
                "devstral",
                architectureOf(source("mistral3", "Devstral-Small-2-24B-Instruct-2512")));
        // Both spellings of the same family stay the same family.
        assertEquals("devstral", architectureOf(source("mistral3", "Devstral Small 2 24B")));
        assertEquals("devstral", architectureOf(source("llama", "Devstral Small 2507")));
    }

    /**
     * {@code mistral3} is Mistral's architecture identifier, not Devstral's, and no provider here
     * claims plain {@code mistral3}. Such a file must stay recognized as itself and claimed by
     * nobody — rejected with a named diagnostic rather than quietly loaded as Devstral, whose
     * tokenizer and chat format it does not share.
     */
    @Test
    public void aNonDevstralMistral3IsStillUnsupported() {
        ModelSource source = source("mistral3", "Mistral Small 3 24B Instruct");
        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> ModelProviders.select(source, families()));
        assertTrue(failure.getMessage(), failure.getMessage().contains("mistral3"));
    }

    /** Files from tools that omit general.architecture keep loading, by the old rules. */
    @Test
    public void aFileWithNoDeclaredArchitectureFallsBackToItsName() {
        assertEquals("llama", architectureOf(source(null, "Llama 3.2 1B Instruct")));
        assertEquals("granite", architectureOf(source(null, "Granite 3.2 2b")));
        assertEquals("phi3", architectureOf(source(null, "Phi-3 mini 4k instruct")));
    }

    @Test
    public void graniteIsStillRecognizedByItsMetadataAlone() {
        ModelSource unnamed =
                ModelSource.ofMetadata(Path.of("m.gguf"), Map.of("granite.block_count", 40));
        assertEquals("granite", architectureOf(unnamed));
    }

    /** One provider per file, by construction: they all ask the same question. */
    @Test
    public void noTwoFamiliesClaimTheSameFile() {
        List<ModelSource> corpus =
                List.of(
                        source("llama", "Llama 3.2 1B Instruct"),
                        source("llama", "models--mistralai--Mistral-7B-Instruct-v0.3"),
                        source("llama", "Devstral Small 2507"),
                        source("qwen2", "Qwen2.5 0.5B Instruct"),
                        source("qwen2", "DeepSeek R1 Distill Qwen 1.5B"),
                        source("qwen3", "Qwen3 0.6B"),
                        source("phi3", "Phi3"),
                        source("granite", "Granite 4.0 1b"));
        for (ModelSource source : corpus) {
            assertEquals(
                    source.metadata() + " must be claimed by exactly one provider",
                    1,
                    families().stream().filter(p -> p.supports(source)).count());
        }
    }

    @Test
    public void anUnsupportedModelFailsWithTheMetadataItSaw() {
        ModelSource unknown = source("mamba", "Mamba 2.8B");
        assertEquals(0, families().stream().filter(p -> p.supports(unknown)).count());

        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> ModelProviders.select(unknown, families()));
        assertTrue(failure.getMessage(), failure.getMessage().contains("mamba"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("Mamba 2.8B"));
    }

    /** A doctored name on an unknown architecture must not rescue it into a wrong family. */
    @Test
    public void aDoctoredNameOnAnUnknownArchitectureIsStillUnsupported() {
        ModelSource doctored = source("mamba", "Llama 3.2 1B Instruct");
        assertEquals(
                "naming it Llama must not make it load as Llama",
                0,
                families().stream().filter(p -> p.supports(doctored)).count());
        assertThrows(
                IllegalStateException.class, () -> ModelProviders.select(doctored, families()));
    }
}
