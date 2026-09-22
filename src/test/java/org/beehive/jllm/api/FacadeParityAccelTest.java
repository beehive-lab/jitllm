package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.Options;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * Class B: runs under {@code -Paccel-tests} with the pinned fixture, skipping explicitly when it is
 * absent. Both paths are exercised — CPU, and GPU when {@code use.tornadovm} selects it.
 *
 * <p>The comparison is against {@code Model.runInstructOnceLangChain4J}, the existing entry point,
 * with equivalent settings and greedy sampling. Greedy matters: it makes the comparison a statement
 * about the code rather than about a shared seed.
 */
public class FacadeParityAccelTest {

    private static final String PROMPT =
            "Explain what a matrix multiplication is in one paragraph.";

    /**
     * The same total budget on both sides. The existing path's {@code maxTokens} counts the prompt
     * too, and the facade caps its request at the session's context length, so equal numbers here
     * mean equal budgets — an unequal budget would show up as one side stopping mid-sentence and
     * read like a delegation bug.
     */
    private static final int MAX_TOKENS = 512;

    private static final String GPU_PROPERTY = "use.tornadovm";

    @Test
    public void theFacadeProducesWhatTheExistingPathProduces_cpu() throws Exception {
        assertFacadeMatchesLegacy(false);
    }

    @Test
    public void theFacadeProducesWhatTheExistingPathProduces_gpu() throws Exception {
        assertFacadeMatchesLegacy(true);
    }

    @Test
    public void theFacadeReportsTheDtypesThatExecute() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(512).build())) {
            assertEquals(
                    "the Q8_0 fixture's weights execute as Q8_0",
                    java.util.Optional.of(org.beehive.jllm.runtime.tensor.DataType.Q8_0),
                    model.info().weightType());
            assertEquals(org.beehive.jllm.runtime.tensor.DataType.Q8_0, model.info().computeType());
        } finally {
            restore(previous);
        }
    }

    @Test
    public void preparingGpuSessionDoesNotChangeGenerationOrPosition() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(128).build())) {
            GenerationRequest request =
                    GenerationRequest.builder()
                            .prompt(PROMPT)
                            .temperature(0)
                            .maxNewTokens(16)
                            .build();
            String lazyText;
            try (GenerationSession lazy = ((TextGenerationModel) model).newSession()) {
                lazyText = lazy.generate(request).text();
            }
            try (GenerationSession prepared = ((TextGenerationModel) model).newSession()) {
                var info = prepared.prepare();
                assertNotEquals("CPU", info.backend());
                assertEquals(info, prepared.prepare());
                assertEquals(0, prepared.position());
                assertEquals(lazyText, prepared.generate(request).text());
                int position = prepared.position();
                prepared.prepare();
                assertEquals(position, prepared.position());
                prepared.reset();
                prepared.prepare();
                assertEquals(0, prepared.position());
                assertEquals(lazyText, prepared.generate(request).text());
            }
        } finally {
            restore(previous);
        }
    }

    private void assertFacadeMatchesLegacy(boolean gpu) throws Exception {
        Path modelPath = fixtureOrSkip();

        String legacy = legacyText(modelPath, gpu);
        String facade = facadeText(modelPath, gpu);

        assertEquals(
                (gpu ? "GPU" : "CPU") + ": the facade must produce the existing path's text",
                legacy,
                facade);
        assertTrue("the comparison is worthless if both are empty", legacy.length() > 0);
    }

    /** The facade, driven exactly as a user would drive it. */
    private static String facadeText(Path modelPath, boolean gpu) throws Exception {
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, Boolean.toString(gpu));
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(512).build())) {
            TextGenerationModel generator = (TextGenerationModel) model;
            try (GenerationSession session = generator.newSession()) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .prompt(PROMPT)
                                        .maxNewTokens(MAX_TOKENS)
                                        .temperature(0) // greedy: no seed dependence
                                        .seed(42)
                                        .build());
                assertTrue("the session must have advanced", session.position() > 0);
                assertNotEquals(0, result.generatedTokens());
                return result.text();
            }
        } finally {
            restore(previous);
        }
    }

    /** The path that exists today, with the same settings. */
    private static String legacyText(Path modelPath, boolean gpu) throws Exception {
        Model model = ModelLoader.loadModel(modelPath, 512, true, gpu);
        Options options =
                new Options(
                        modelPath, PROMPT, null, null, false, 0.0f, 0.95f, 42L, 512, true, false,
                        gpu, false, 1);
        Sampler sampler =
                Sampler.selectSampler(model.configuration().vocabularySize(), 0.0f, 0.95f, 42L);
        List<String> streamed = new ArrayList<>();
        String text =
                org.beehive.jllm.generation.ModelGeneration.runInstructOnceLangChain4J(
                        model, sampler, options, streamed::add);
        return text;
    }

    private static Path fixtureOrSkip() {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent: fixture " + Fixture.LLAMA_3_2_1B_Q8_0.fileName, false);
        }
        return modelPath;
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(GPU_PROPERTY);
        } else {
            System.setProperty(GPU_PROPERTY, previous);
        }
    }
}
