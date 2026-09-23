package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.api.GenerationRequest;
import org.beehive.jllm.api.GenerationResult;
import org.beehive.jllm.api.GenerationSession;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.api.TextGenerationModel;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * An earlier run of this slice's gate looked green while the opt-in had <b>never reached the
 * JVM</b>: a {@code -DargLine} override had replaced the accel profile's own arguments, so the
 * property was absent and every session quietly took the legacy path. Eighteen tests passed and
 * proved nothing about the lowering.
 *
 * <p>So this test does not read the property back, and does not trust a log. It reads {@link
 * LoweredPlanSelection#loweredPlanCount()} before and after, and <b>fails if the count did not
 * move</b>. A fallback to legacy construction is therefore a failure, not a silent pass.
 *
 * <p>It was outside it for a measured reason: every GPU session built its own plan holding its own
 * device copy of the weights, so adding one more model load to the shared JVM made an unrelated
 * test — {@code FacadeParityAccelTest} — fail with {@code TornadoOutOfMemory}. Sharing removed the
 * duplication, and the suite forks one JVM per class, which is what actually returns device memory:
 * a closed session's buffers go back to TornadoVM's buffer provider, not to the driver, and the
 * provider recycles them only under budget pressure.
 *
 * <h2>What this slice does and does not claim</h2>
 */
public class LoweredLlamaFp16AccelTest {

    private static final int MAX_TOKENS = 32;
    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String PROMPT = "Explain what a lighthouse does, in two sentences.";

    /**
     * The lowered path executes, and produces token-identical output to the legacy path.
     *
     * <p>Both halves matter. The count assertion proves the lowering ran; the comparison proves it
     * ran <i>correctly</i>. Either alone would be satisfiable by a path that did nothing useful.
     */
    @Test
    public void theLoweredPathRunsAndMatchesTheLegacyPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        // Set before loading: the shared key/value pool a lowered domain requires is attached at
        // model construction, so a run that enables lowering only afterwards is correctly refused
        // by the eligibility veto. Clearing the property per generation still selects the legacy
        // path, since selection is read per call.
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(256).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;

            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
            long beforeLegacy = LoweredPlanSelection.loweredPlanCount();
            String legacy = generateOnce(generator);
            assertEquals(
                    "the legacy path must not reach the lowering",
                    beforeLegacy,
                    LoweredPlanSelection.loweredPlanCount());

            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
            long beforeLowered = LoweredPlanSelection.loweredPlanCount();
            String lowered = generateOnce(generator);
            assertTrue(
                    "the lowered path did not run — execution fell back to legacy construction,"
                            + " which is exactly the false green this assertion exists to catch",
                    LoweredPlanSelection.loweredPlanCount() > beforeLowered);

            assertEquals(
                    "the lowered path must produce what the legacy path produces", legacy, lowered);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    private static String generateOnce(TextGenerationModel generator) throws Exception {
        try (GenerationSession session = generator.newSession()) {
            GenerationResult result =
                    session.generate(
                            GenerationRequest.builder()
                                    .prompt(PROMPT)
                                    .maxNewTokens(MAX_TOKENS)
                                    .temperature(0f)
                                    .seed(1234L)
                                    .build());
            return result.text();
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
