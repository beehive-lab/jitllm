package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;

// @formatter:off
/**
 * Does a reset actually return a {@code qwen35} session to the state it started in — on the
 * <b>device</b>?
 *
 * <p>The lifecycle tests ask the same question of the public surface and cannot answer it: both
 * pass unchanged when the reset clears only the host arrays, because their prompt is short and its
 * one-word greedy answer survives a polluted recurrent state. That is measured, not assumed. This
 * case is built to fail instead:
 *
 * <ol>
 *   <li>generate from a fresh session and keep every logits row;
 *   <li>generate a <b>different, longer</b> sequence, which advances the convolution windows and
 *       the delta-net matrices and leaves them holding it;
 *   <li>reset;
 *   <li>generate the first sequence again and compare the rows element for element.
 * </ol>
 *
 * <p>Equality has to be exact. Both runs issue the same kernels over the same weights at the same
 * positions, so the only thing that can differ is the recurrent state carried in, which is the
 * subject. Step 2 is asserted to have moved the logits, so a reset that did nothing at all cannot
 * pass by making steps 1 and 4 trivially identical.
 *
 * <p>This is also the property the benchmark depends on: {@code JitllmBench.runTest} calls every
 * repetition an independent sequence from position zero, and that is only true if the recurrent
 * state starts each one at zero on the device.
 */
// @formatter:on
abstract class Qwen35SequenceReset {

    /** Rows to compare. Long enough for the recurrence to matter, short enough to stay quick. */
    private static final int TOKENS = 24;

    private static final String PROMPT = "What is the capital of France? Answer in one word.";

    /** Deliberately unlike the subject prompt, so the state it leaves behind is unlike zero. */
    private static final String POLLUTING_PROMPT =
            "Write out the first twelve prime numbers, then explain in detail why the sieve of"
                    + " Eratosthenes finds them, and describe its running time.";

    private static final int POLLUTING_TOKENS = 48;

    /**
     * What this run's own plan had to dispatch for its attention. Nothing here; the half-precision
     * key/value subclasses override it, because split-KV is selected on that path alone.
     */
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {}

    void assertResetRestoresTheSequence(int prefillBatchSize) throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] environment absent — no TornadoVM device");
            assumeTrue("environment absent", false);
        }
        GoldenCapture.assertHostLogitsAvailable();

        Model model = ModelLoader.loadModel(modelPath, GoldenCapture.CONTEXT_LENGTH, true, true);
        State state =
                prefillBatchSize > 1
                        ? State.withPrefillBatchSize(prefillBatchSize, model::createNewState)
                        : model.createNewState();
        int seedToken = state.latestToken;

        TornadoVMMasterPlan plan = null;
        try {
            plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            verifyAttentionDispatch(plan);

            List<float[]> fresh = generate(model, state, plan, PROMPT, TOKENS);
            assertFalse("nothing was generated, so the case proves nothing", fresh.isEmpty());
            assertTrue("too few rows to say anything: " + fresh.size(), fresh.size() >= TOKENS - 4);

            // Leave the recurrence holding something else.
            state.latestToken = seedToken;
            List<float[]> polluting =
                    generate(model, state, plan, POLLUTING_PROMPT, POLLUTING_TOKENS);
            assertNotEquals(
                    "the polluting sequence produced the same first row as the subject, so it"
                            + " cannot be said to have changed the recurrent state",
                    fresh.get(0)[0],
                    polluting.get(0)[0]);

            plan.resetSequenceState();
            state.latestToken = seedToken;
            List<float[]> afterReset = generate(model, state, plan, PROMPT, TOKENS);

            assertEquals("rows after the reset", fresh.size(), afterReset.size());
            for (int r = 0; r < fresh.size(); r++) {
                float[] before = fresh.get(r);
                float[] after = afterReset.get(r);
                assertEquals("vocabulary length, row " + r, before.length, after.length);
                for (int i = 0; i < before.length; i++) {
                    if (before[i] != after[i]) {
                        // Reported with context: which row, which logit, and how far off, because
                        // "not equal" alone does not distinguish a stale recurrent state from a
                        // stale key/value entry.
                        assertTrue(
                                "reset did not restore the sequence: row "
                                        + r
                                        + " logit "
                                        + i
                                        + " was "
                                        + before[i]
                                        + " on a fresh session and "
                                        + after[i]
                                        + " after a reset (difference "
                                        + Math.abs(before[i] - after[i])
                                        + ")",
                                false);
                    }
                }
            }
        } finally {
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }
    }

    /** One greedy generation, keeping every logits row. */
    private static List<float[]> generate(
            Model model, State state, TornadoVMMasterPlan plan, String prompt, int tokens)
            throws Exception {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> promptTokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        promptTokens.addAll(
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, prompt)));
        promptTokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<float[]> rows = new ArrayList<>();
        Sampler capturing =
                logits -> {
                    float[] row = new float[logits.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = logits.get(i);
                    }
                    rows.add(row);
                    return Sampler.TENSOR_ARGMAX.sampleToken(logits);
                };

        int skippedSeed =
                org.beehive.jitllm.inference.PromptIngestion.of(state, promptTokens, 0)
                        .firstIndex();
        int budget = promptTokens.size() + tokens - skippedSeed;
        model.generateTokensGPU(
                state, 0, promptTokens, Set.of(), budget, capturing, false, null, plan);
        return rows;
    }
}
