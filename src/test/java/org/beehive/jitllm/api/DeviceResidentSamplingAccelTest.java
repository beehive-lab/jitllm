package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * {@code ResolvedPolicyReachesThePlanAccelTest.bothResidenciesReachThePlan} already proves the
 * policy reaches the task graph (the {@code logits.argmax_sample} task is present only when
 * requested). What it cannot prove is that the on-device argmax computes the right token and that
 * {@code state.workspace.deviceSampledToken()} correctly carries it back to the host across a full
 * multi-step generation — {@link org.beehive.jitllm.golden.GoldenCapture} deliberately forbids {@code
 * deviceSample=true} ({@code assertHostLogitsAvailable()}) because its capture hook needs the
 * host-visible logits row, so this is the only place that exercises it end to end.
 */
public class DeviceResidentSamplingAccelTest {

    private static final int MAX_TOKENS = 64;

    @Test
    public void deviceResidentArgmaxMatchesHostResidentArgmax() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty("use.tornadovm");
        String previousDeviceSample = System.getProperty("jitllm.deviceSample");
        System.setProperty("use.tornadovm", "true");
        try {
            System.clearProperty("jitllm.deviceSample");
            String hostText = generate(model);

            System.setProperty("jitllm.deviceSample", "true");
            String deviceText = generate(model);

            assertTrue(
                    "device-resident sampling must produce a non-empty continuation",
                    !deviceText.isBlank());
            assertEquals(
                    "host- and device-resident argmax sampling must pick the identical greedy"
                            + " token sequence — a difference means the on-device argmax result did"
                            + " not cross the invocation boundary correctly",
                    hostText,
                    deviceText);
        } finally {
            restore("use.tornadovm", previousGpu);
            restore("jitllm.deviceSample", previousDeviceSample);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static String generate(Path model) throws Exception {
        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(512).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .prompt(
                                                "Explain what a matrix multiplication is in one paragraph.")
                                        .maxNewTokens(MAX_TOKENS)
                                        .temperature(0.0f)
                                        .build());
                return result.text();
            }
        }
    }
}
