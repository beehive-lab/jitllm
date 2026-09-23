package org.beehive.jllm.api;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * A generation call allocates two different things: a <b>fixed</b> amount per call — tokenizing the
 * prompt, the chat template, the result — and a <b>per-token</b> amount inside the decode loop.
 * Only the second is this milestone's subject, and a single measurement cannot tell them apart: at
 * 32 tokens the fixed cost is a quarter of the total and would drown the difference being looked
 * for.
 *
 * <p>So the measurement is taken at <b>two token counts</b> and the per-token cost read off as the
 * slope, {@code (bytes(n2) - bytes(n1)) / (n2 - n1)}, which cancels the intercept exactly. Measured
 * on this machine: about **8 MiB fixed per call** and **650 KiB per token**, and the claim under
 * test is that the lowered path's slope matches the legacy path's.
 *
 * <p><b>Warm-up matters, and getting it wrong produced a false result once.</b> The first
 * measurement taken in a JVM reads ~9% lower than every later one, whichever path it is —
 * interpreting that as "the lowered path allocates 10% more" is exactly the mistake this comment
 * exists to prevent. Both paths are therefore exercised once before anything is recorded.
 *
 * <p>The 650 KiB itself is <b>not</b> the invocation boundary's: it is the same on the legacy path,
 * which has no boundary at all, and it survives a decode loop whose Java code allocates nothing
 * vocabulary-sized. It is TornadoVM's per-execution bookkeeping. The boundary adds one {@code
 * Result} record per token and no array.
 *
 * <h2>It was seen to fail, and what it cannot see</h2>
 *
 * <p>A vocabulary-sized {@code float[]} added to {@code SharedWorkspacePlan.invoke} moves the
 * lowered slope from 627 KiB to 1140 KiB — exactly the 513 KiB the array costs — and this test
 * fails. That control was run, and reverted.
 *
 * <p>The <b>first</b> attempt at that control did not fail, and the reason is a real limit of the
 * method: an allocation that does not escape is scalarized by the JIT and never happens. So this
 * test proves that no allocation <i>survives optimization</i> per token, which is what matters for
 * a decode loop — not that no {@code new} appears in the source.
 */
public class DecodeAllocationAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    /** Two token counts far enough apart that the slope is not noise. */
    private static final int FEW = 32;

    private static final int MANY = 128;

    /**
     * The tolerance on the slope comparison. Generous on purpose: the claim is "the lowered path
     * adds no per-token allocation", and a 10% band still fails an added vocabulary-sized array,
     * which would be a 79% step at this vocabulary size.
     */
    private static final double TOLERANCE = 0.10;

    @Test
    public void theLoweredPathAllocatesNoMorePerTokenThanTheLegacyPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            // Both paths warmed before anything is recorded — see the class javadoc.
            allocatedBytes(model, false, FEW);
            allocatedBytes(model, true, FEW);

            double legacy = perTokenBytes(model, false);
            double lowered = perTokenBytes(model, true);

            System.out.println(
                    "per-token allocation: legacy "
                            + Math.round(legacy)
                            + " B, lowered "
                            + Math.round(lowered)
                            + " B");

            assertTrue(
                    "both slopes must be positive and plausible, or the measurement is broken"
                            + " rather than informative: legacy "
                            + legacy
                            + ", lowered "
                            + lowered,
                    legacy > 1024 && lowered > 1024);
            assertTrue(
                    "the lowered path allocates "
                            + Math.round(lowered)
                            + " bytes per token"
                            + " against the legacy path's "
                            + Math.round(legacy)
                            + " — an"
                            + " invocation that binds must not allocate",
                    lowered <= legacy * (1 + TOLERANCE));
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    /** The slope: allocation attributable to a token, with the per-call intercept cancelled. */
    private static double perTokenBytes(Path model, boolean lowered) throws Exception {
        Measurement few = allocatedBytes(model, lowered, FEW);
        Measurement many = allocatedBytes(model, lowered, MANY);
        assertTrue(
                "generation stopped early, so the two points are not "
                        + FEW
                        + " and "
                        + MANY
                        + " tokens apart: "
                        + few.tokens
                        + " and "
                        + many.tokens,
                many.tokens - few.tokens >= (MANY - FEW) / 2);
        return (double) (many.bytes - few.bytes) / (many.tokens - few.tokens);
    }

    private record Measurement(long bytes, int tokens) {}

    private static Measurement allocatedBytes(Path model, boolean lowered, int tokens)
            throws Exception {
        if (lowered) {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "on");
        } else {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
        }
        ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        long before1 = LoweredPlanSelection.loweredPlanCount();
        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(512).build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                generate(session, 8); // plan construction, compilation, first upload
                if (lowered) {
                    assertTrue(
                            "the lowered path did not run, so this measurement is of the legacy"
                                    + " path under another name",
                            LoweredPlanSelection.loweredPlanCount() > before1);
                }
                long id = Thread.currentThread().threadId();
                long before = threads.getThreadAllocatedBytes(id);
                int produced = generate(session, tokens);
                long after = threads.getThreadAllocatedBytes(id);
                return new Measurement(after - before, produced);
            }
        }
    }

    private static int generate(GenerationSession session, int tokens) {
        return session.generate(
                        GenerationRequest.builder()
                                .prompt(
                                        "Write a detailed paragraph about the history of lighthouses.")
                                .maxNewTokens(tokens)
                                .temperature(0f)
                                .seed(3L)
                                .build())
                .generatedTokens();
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
