package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * The claim is <b>allocation</b>: a second lowered session must construct no second device
 * workspace and no second key/value store. Object identity is what is actually being claimed, so it
 * is what is compared — counting bytes would be indirect and noisy.
 *
 * <p>In the automatic suite. The session-count sweep that measures what sharing costs and saves
 * lives in {@code LoweredSharingScalingDeviceCheck}, outside the suite: eight legacy sessions hold
 * eight device copies of the weights, and a test whose purpose is to consume the device cannot
 * share a JVM with anything else.
 */
public class LoweredSharedWorkspaceAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    /** Acceptance 7 says "over 100+ decode steps"; this is that number. */
    private static final int MIN_DECODE_STEPS = 100;

    /** Acceptance 1: one workspace, one domain, one compiled program, two sessions. */
    @Test
    public void twoLoweredSessionsShareOneWorkspaceAndOneProgram() throws Exception {
        withLoweredModel(
                (model, handle) -> {
                    try (GenerationSession first = model.newSession()) {
                        warm(first);
                        var workspace = handle.loweredWorkspace();
                        var program = handle.loweredProgram();
                        var domain = handle.loweredDomain();
                        assertNotNull("the first lowered session must build the domain", workspace);
                        assertNotNull(program);
                        assertNotNull(domain);

                        try (GenerationSession second = model.newSession()) {
                            warm(second);
                            assertSame(
                                    "a second lowered session must not construct another workspace",
                                    workspace,
                                    handle.loweredWorkspace());
                            assertSame(
                                    "nor another compiled program",
                                    program,
                                    handle.loweredProgram());
                            assertSame(
                                    "nor another binding domain", domain, handle.loweredDomain());
                            assertEquals(
                                    "one compiled program for both sessions",
                                    1,
                                    handle.compiledProgramCount());
                        }
                    }
                });
    }

    /** Acceptance 4: the workspace's {@code latestToken} is not read by the lowered path. */
    @Test
    public void poisoningTheWorkspacesLatestTokenChangesNothing() throws Exception {
        withLoweredModel(
                (model, handle) -> {
                    String clean;
                    try (GenerationSession session = model.newSession()) {
                        clean = generate(session);
                    }
                    try (GenerationSession session = model.newSession()) {
                        handle.loweredWorkspace().latestToken = 999_999;
                        String poisoned = generate(session);
                        assertEquals(
                                "the lowered path must not read the workspace's latestToken",
                                clean,
                                poisoned);
                    }
                });
    }

    /** Acceptance 3 and 5: interleaved sessions keep independent histories; close isolates. */
    @Test
    public void sessionsRetainIndependentHistoriesAcrossCloseAndReset() throws Exception {
        withLoweredModel(
                (model, handle) -> {
                    try (GenerationSession survivor = model.newSession()) {
                        String before = generate(survivor);

                        GenerationSession other = model.newSession();
                        warm(other);
                        other.close();

                        assertNotNull(
                                "closing a borrower must not release the workspace",
                                handle.loweredWorkspace());
                        assertNotNull("nor the shared program", handle.loweredProgram());

                        survivor.reset();
                        String after = generate(survivor);
                        assertEquals(
                                "a reset session must ingest exactly as a new one does",
                                before,
                                after);
                    }
                });
    }

    /**
     * Acceptance 7: the workspace, the compiled program and the binding domain a session borrows
     * are the <b>same objects</b> after more than a hundred decode steps as they were at the first.
     *
     * <p>What this is really testing is that nothing in the decode loop reconstructs the domain. A
     * per-token rebuild would still produce correct tokens — the plan is deterministic — so
     * correctness tests cannot see it; only identity can. The plan count is asserted unchanged for
     * the same reason: it moves once, when the domain is first built, and never again.
     *
     * <p>Its own context length, because 100+ tokens do not fit the 256 the sharing tests use, and
     * a run that silently truncated would assert over fewer steps than it claims.
     */
    @Test
    public void identityIsStableOverAHundredDecodeSteps() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(1024).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            DelegatingModel handle = (DelegatingModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                int steps =
                        session.generate(
                                        GenerationRequest.builder()
                                                .prompt(
                                                        "Write a detailed paragraph about the history of lighthouses.")
                                                .maxNewTokens(64)
                                                .temperature(0f)
                                                .seed(11L)
                                                .build())
                                .generatedTokens();

                var workspace = handle.loweredWorkspace();
                var program = handle.loweredProgram();
                var domain = handle.loweredDomain();
                assertNotNull(
                        "the lowered domain must exist after the first generation", workspace);
                long planCount = LoweredPlanSelection.loweredPlanCount();

                // Successive turns on one session, so the count is decode steps rather than
                // sessions: a single request can stop at an end-of-text token well before its
                // budget, which would leave the claim unverified rather than failed.
                for (int turn = 0; turn < 40 && steps < MIN_DECODE_STEPS; turn++) {
                    steps +=
                            session.generate(
                                            GenerationRequest.builder()
                                                    .prompt(
                                                            "Write another detailed paragraph on the same subject.")
                                                    .maxNewTokens(64)
                                                    .temperature(0f)
                                                    .seed(11L + turn)
                                                    .build())
                                    .generatedTokens();
                }

                assertTrue(
                        "fewer than "
                                + MIN_DECODE_STEPS
                                + " decode steps ran ("
                                + steps
                                + "), so identity was not observed over the claimed span",
                        steps >= MIN_DECODE_STEPS);
                assertSame(
                        "the workspace was rebuilt during decoding",
                        workspace,
                        handle.loweredWorkspace());
                assertSame(
                        "the compiled program was rebuilt during decoding",
                        program,
                        handle.loweredProgram());
                assertSame(
                        "the binding domain was rebuilt during decoding",
                        domain,
                        handle.loweredDomain());
                assertEquals("one compiled program throughout", 1, handle.compiledProgramCount());
                assertEquals(
                        "a plan was constructed after the first — that is a per-token rebuild",
                        planCount,
                        LoweredPlanSelection.loweredPlanCount());
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    /**
     * Two Q8_0 sessions resolve to one compiled program exactly as two F16 ones do. Worth its own
     * case rather than trusting the F16 result: the representation is now part of the signature,
     * and therefore of the cache key, so a mistake there would show as two entries for one model.
     */
    @Test
    public void twoQ8_0SessionsShareOneProgram() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(256)
                                .maxConcurrentSessions(2)
                                .build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            DelegatingModel handle = (DelegatingModel) loaded;
            long before = LoweredPlanSelection.loweredPlanCount();
            try (GenerationSession first = generator.newSession()) {
                warm(first);
                assertTrue(
                        "the lowered path did not run, so this proves nothing about Q8_0",
                        LoweredPlanSelection.loweredPlanCount() > before);
                var workspace = handle.loweredWorkspace();
                var program = handle.loweredProgram();
                try (GenerationSession second = generator.newSession()) {
                    warm(second);
                    assertSame(
                            "a second Q8_0 session must not construct another workspace",
                            workspace,
                            handle.loweredWorkspace());
                    assertSame("nor another compiled program", program, handle.loweredProgram());
                    assertEquals(
                            "one compiled program for both Q8_0 sessions; keys held: "
                                    + handle.compiledProgramKeys(),
                            1,
                            handle.compiledProgramCount());
                }
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    /**
     * <b>What must not happen is the third option:</b> a shared workspace and compiled program over
     * session-private key/value arrays, where one session reads another's cache. Falling back
     * shares nothing, so it cannot produce that.
     */
    @Test
    public void aFamilyWithoutSharedStorageFallsBackInsteadOfFailing() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN2_5_0_5B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN2_5_0_5B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
        // FP32: this is about the shared-pool fallback, and Qwen2's GPU layers keep an FP32
        // cache, so the FP16 default is refused for them.
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(256)
                                .storageOptions(
                                        org.beehive.jllm.runtime.policy.StorageOptions.fp32())
                                .build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            long before = LoweredPlanSelection.loweredPlanCount();
            String text;
            try (GenerationSession session = generator.newSession()) {
                text = generate(session);
            }
            assertTrue(
                    "the session still lowers — the branch is in the plan factory, which the"
                            + " legacy runtime uses too",
                    LoweredPlanSelection.loweredPlanCount() > before);
            assertNull(
                    "but no binding domain is built, so nothing is shared",
                    ((DelegatingModel) loaded).loweredWorkspace());
            assertEquals(
                    "and no compiled program is cached for sharing",
                    0,
                    ((DelegatingModel) loaded).compiledProgramCount());
            assertNotNull("and it must still generate", text);
            assertTrue(
                    "a fallback that produced nothing would be a failure wearing a fallback's"
                            + " name",
                    text.length() > 0);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    /**
     * The default single session is enough for the lowered path: its pool has one slot and the
     * scratch block, and the slot is reusable once the session closes. Under {@code auto}, which is
     * what every caller that sets nothing gets.
     */
    @Test
    public void theDefaultSingleSessionLowersAndReopens() throws Exception {
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
                LocalModels.load(model, ModelOptions.builder().contextLength(8192).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            long before = LoweredPlanSelection.loweredPlanCount();
            String first;
            try (GenerationSession session = generator.newSession()) {
                first = generate(session);
            }
            assertTrue(
                    "auto did not lower the qualified tuple",
                    LoweredPlanSelection.loweredPlanCount() > before);
            try (GenerationSession reopened = generator.newSession()) {
                assertEquals(
                        "the reopened session must reuse the slot and reproduce the output",
                        first,
                        generate(reopened));
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    // harness

    private interface Body {
        void run(TextGenerationModel model, DelegatingModel handle) throws Exception;
    }

    private static void withLoweredModel(Body body) throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(256)
                                .maxConcurrentSessions(2)
                                .build())) {
            long before = LoweredPlanSelection.loweredPlanCount();
            body.run((TextGenerationModel) loaded, (DelegatingModel) loaded);
            assertTrue(
                    "the lowered path did not run — this test would prove nothing about sharing",
                    LoweredPlanSelection.loweredPlanCount() > before);
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

    private static String generate(GenerationSession session) {
        return session.generate(
                        GenerationRequest.builder()
                                .prompt("Name one colour.")
                                .maxNewTokens(12)
                                .temperature(0f)
                                .seed(7L)
                                .build())
                .text();
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
