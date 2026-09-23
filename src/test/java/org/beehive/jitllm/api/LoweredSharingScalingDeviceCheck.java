package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Split out of {@link LoweredSharedWorkspaceAccelTest}, which asserts sharing. These are
 * <b>measurements</b>: eight legacy sessions hold eight device copies of the weights — 17 GiB on
 * the machine this was recorded on — and the point of the sweep is exactly that cost. They are
 * therefore <b>deliberately outside the automatic accel suite</b>, and not because they are flaky
 * or slow: a test whose purpose is to consume the device cannot share one with anything else.
 *
 * <p>This is a different reason from the one that kept the lowered correctness checks out until
 * sharing landed. Those rejoined the suite; these will not, and should not.
 *
 * <p><b>One measurement per JVM.</b> Running the class whole fails, and the reason is the subject
 * itself: a closed session's device memory goes back to TornadoVM's buffer provider but not to the
 * driver, and the provider recycles it only under budget pressure — so the second measurement in a
 * JVM starts with the first one's sessions still resident. Each method passes on its own:
 *
 * <pre>
 *   source ~/TornadoVM/setvars.sh &amp;&amp; mvn verify -Paccel-tests \
 *       -Dtest=LoweredSharingScalingDeviceCheck#scaling_lowered_8 -DfailIfNoTests=false
 * </pre>
 *
 * <p>Verified 2026-09-01: all five green run one at a time, three of five red run together. That is
 * a property of the measurement, not a defect in it.
 *
 * <p>Recorded 2026-08-31, peak device memory less a 79 MiB idle baseline: legacy 3131 / 8339 /
 * 17335 MiB at 2 / 4 / 8 sessions; lowered flat at 3243 MiB. The crossover is between two and four
 * sessions — at two, the pool sized for eight makes sharing 112 MiB <i>worse</i>.
 */
public class LoweredSharingScalingDeviceCheck {

    private static final String GPU_PROPERTY = "use.tornadovm";

    /**
     * The paired legacy measurement: the same two sessions, the same model and context, without
     * lowering.
     */
    @Test
    public void twoLegacySessionsForComparison() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(256)
                                .maxConcurrentSessions(2)
                                .build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            long before = LoweredPlanSelection.loweredPlanCount();
            try (GenerationSession first = generator.newSession();
                    GenerationSession second = generator.newSession()) {
                warm(first);
                warm(second);
            }
            assertEquals(
                    "this measurement must be of the legacy path",
                    before,
                    LoweredPlanSelection.loweredPlanCount());
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    // crossover measurement: how the two paths scale with session count

    @Test
    public void scaling_legacy_4() throws Exception {
        manySessions(4, false);
    }

    @Test
    public void scaling_lowered_4() throws Exception {
        manySessions(4, true);
    }

    @Test
    public void scaling_legacy_8() throws Exception {
        manySessions(8, false);
    }

    @Test
    public void scaling_lowered_8() throws Exception {
        manySessions(8, true);
    }

    /**
     * Opens {@code count} sessions at once and warms each, so peak device memory reflects them all
     * being live.
     *
     * <p>The point of the sweep: the lowered path pays a fixed pool cost and the legacy path pays a
     * per-session one, so where they cross is a measurement rather than an argument. At two
     * sessions the fixed cost wins and sharing is more expensive.
     *
     * <p>The shared pool is sized for eight sessions, so eight is the last count where the
     * comparison is meaningful without resizing it.
     */
    private void manySessions(int count, boolean lowered) throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        if (lowered) {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "on");
        } else {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
        }
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(256)
                                .maxConcurrentSessions(count)
                                .build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            long before = LoweredPlanSelection.loweredPlanCount();
            GenerationSession[] sessions = new GenerationSession[count];
            try {
                for (int i = 0; i < count; i++) {
                    sessions[i] = generator.newSession();
                    warm(sessions[i]);
                }
                if (lowered) {
                    assertTrue(
                            "the lowered path did not run",
                            LoweredPlanSelection.loweredPlanCount() > before);
                    assertEquals(
                            "all sessions must share one compiled program; keys held: "
                                    + ((DelegatingModel) loaded).compiledProgramKeys(),
                            1,
                            ((DelegatingModel) loaded).compiledProgramCount());
                } else {
                    assertEquals(
                            "this measurement must be of the legacy path",
                            before,
                            LoweredPlanSelection.loweredPlanCount());
                }
            } finally {
                for (GenerationSession session : sessions) {
                    if (session != null) {
                        session.close();
                    }
                }
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    private static void warm(GenerationSession session) {
        session.generate(
                GenerationRequest.builder()
                        .prompt("Hi")
                        .maxNewTokens(4)
                        .temperature(0f)
                        .seed(1L)
                        .build());
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
