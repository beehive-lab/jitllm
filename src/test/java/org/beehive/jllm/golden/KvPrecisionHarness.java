package org.beehive.jllm.golden;

import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.PromptIngestion;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.policy.StorageOptions;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Compares a run with an FP16 key/value cache against the same run with an FP32 one.
 *
 * <p>The FP32 run is the reference: it samples greedily and records the logits at every sampled
 * position. The FP16 run is <b>teacher-forced</b> onto the reference's tokens, so both see exactly
 * the same inputs and every row is comparable — a single early divergence cannot turn the rest of
 * the comparison into noise, and matching generated text is not what is being measured.
 *
 * <p>A row is compared by cosine similarity, by relative L2 error, and by whether the two agree on
 * the most likely next token. The rows come from the real generation loop, so prefill in whatever
 * mode the properties select, then single-token decode reading what prefill wrote.
 */
public final class KvPrecisionHarness {

    private KvPrecisionHarness() {}

    /** Logits at every sampled position, and the tokens that were fed back. */
    public record Trace(List<float[]> rows, int[] tokens, String executionMode, String kvCache) {}

    /** How close two traces are, row by row; the worst row decides. */
    public record Comparison(
            int rows, double minCosine, double maxRelativeL2, double top1Agreement) {

        public void assertWithin(
                String what, double minCosine, double maxRelativeL2, double minTop1) {
            String report = what + ": " + this;
            assertTrue("too few rows to say anything, " + report, rows >= 8);
            assertTrue("cosine below " + minCosine + ", " + report, this.minCosine >= minCosine);
            assertTrue(
                    "relative L2 above " + maxRelativeL2 + ", " + report,
                    this.maxRelativeL2 <= maxRelativeL2);
            assertTrue(
                    "top-1 agreement below " + minTop1 + ", " + report,
                    this.top1Agreement >= minTop1);
            // Half-precision storage always perturbs the logits a little. A run identical to the
            // FP32 reference means the FP16 cache was never read — the path silently kept FP32.
            assertTrue(
                    "FP16 storage changed nothing, so it was not used, " + report,
                    this.maxRelativeL2 > 1e-7);
        }
    }

    /**
     * A prompt long enough to cross several prefill chunks, as the chat template renders it.
     *
     * @param approximateTokens roughly how many prompt tokens to produce
     */
    public static List<Integer> longPrompt(Model model, int approximateTokens) {
        StringBuilder text =
                new StringBuilder("Read the following notes and answer the question after them.\n");
        String[] notes = {
            "The lighthouse keeper climbed the spiral stairs every evening to light the lamp.",
            "Ships far out at sea relied on its steady beam to pass the rocks safely.",
            "In winter the storms were so strong that spray reached the top of the tower.",
            "The keeper logged every passing vessel, the weather, and the state of the lamp.",
            "A supply boat came once a fortnight with oil, food and letters from the mainland.",
        };
        for (int i = 0; text.length() < approximateTokens * 4; i++) {
            text.append(i + 1).append(". ").append(notes[i % notes.length]).append('\n');
        }
        text.append("Question: what did the keeper record, and why did ships need the light?");
        ChatFormat format = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(format.getBeginOfText());
        }
        tokens.addAll(
                format.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, text.toString())));
        tokens.addAll(format.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        return tokens;
    }

    /** Loads the model for {@code gpu} or the CPU at {@code contextLength}. */
    public static Model load(Path file, int contextLength, boolean gpu) throws Exception {
        return ModelLoader.loadModel(file, contextLength, true, gpu);
    }

    /**
     * Runs the prompt and {@code decodeSteps} generated tokens.
     *
     * @param forced the reference tokens to feed back, or {@code null} to sample greedily
     */
    public static Trace run(
            Model model,
            boolean gpu,
            DataType keyValue,
            int prefillBatchSize,
            List<Integer> prompt,
            int decodeSteps,
            int[] forced) {
        StorageOptions storage = new StorageOptions(keyValue, false);
        State state =
                State.withStorageOptions(
                        storage,
                        () ->
                                prefillBatchSize > 1
                                        ? State.withPrefillBatchSize(
                                                prefillBatchSize, model::createNewState)
                                        : model.createNewState());
        List<float[]> rows = new ArrayList<>();
        List<Integer> fed = new ArrayList<>();
        Sampler sampler =
                logits -> {
                    float[] row = new float[logits.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = logits.get(i);
                    }
                    int token =
                            forced != null && rows.size() < forced.length
                                    ? forced[rows.size()]
                                    : Sampler.TENSOR_ARGMAX.sampleToken(logits);
                    rows.add(row);
                    fed.add(token);
                    return token;
                };
        int skippedSeed = PromptIngestion.of(state, prompt, 0).firstIndex();
        int budget = prompt.size() + decodeSteps - skippedSeed;
        String executionMode = "cpu";
        String kvCache = state.usesFp16KeyValueCache() ? "FP16" : "FP32";
        if (gpu) {
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            executionMode = plan.executionInfo().mode();
            kvCache = plan.executionInfo().kvCache();
            try {
                model.generateTokensGPU(
                        state, 0, prompt, Set.of(), budget, sampler, false, null, plan);
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } else {
            model.generateTokens(state, 0, prompt, Set.of(), budget, sampler, false, null);
        }
        return new Trace(
                rows, fed.stream().mapToInt(Integer::intValue).toArray(), executionMode, kvCache);
    }

    /**
     * The FP32 reference, then the FP16 run forced onto its tokens, compared.
     *
     * @param expectedMode the execution mode both plans must report, or {@code null} for any: a row
     *     must not pass on a different path than the one it names
     */
    public static Comparison compareFp16AgainstFp32(
            Model model,
            boolean gpu,
            int prefillBatchSize,
            List<Integer> prompt,
            int steps,
            String expectedMode) {
        Trace reference = run(model, gpu, DataType.F32, prefillBatchSize, prompt, steps, null);
        Trace fp16 =
                run(model, gpu, DataType.F16, prefillBatchSize, prompt, steps, reference.tokens());
        assertTrue(
                "the reference must run an FP32 cache: " + reference.kvCache(),
                reference.kvCache().equals("FP32"));
        assertTrue(
                "the test must run an FP16 cache: " + fp16.kvCache(),
                fp16.kvCache().equals("FP16"));
        if (expectedMode != null) {
            assertTrue(
                    "expected the "
                            + expectedMode
                            + " plan, the reference ran "
                            + reference.executionMode(),
                    reference.executionMode().equals(expectedMode));
            assertTrue(
                    "expected the "
                            + expectedMode
                            + " plan, the FP16 run ran "
                            + fp16.executionMode(),
                    fp16.executionMode().equals(expectedMode));
        }
        return compare(reference, fp16);
    }

    public static Comparison compare(Trace reference, Trace test) {
        int rows = Math.min(reference.rows().size(), test.rows().size());
        double minCosine = 1;
        double maxRelative = 0;
        int agree = 0;
        for (int r = 0; r < rows; r++) {
            float[] a = reference.rows().get(r);
            float[] b = test.rows().get(r);
            double dot = 0, na = 0, nb = 0, diff = 0;
            int argA = 0, argB = 0;
            for (int i = 0; i < a.length; i++) {
                dot += (double) a[i] * b[i];
                na += (double) a[i] * a[i];
                nb += (double) b[i] * b[i];
                double d = a[i] - b[i];
                diff += d * d;
                if (a[i] > a[argA]) argA = i;
                if (b[i] > b[argB]) argB = i;
            }
            minCosine = Math.min(minCosine, dot / Math.sqrt(na * nb));
            maxRelative = Math.max(maxRelative, Math.sqrt(diff / na));
            if (argA == argB) agree++;
        }
        return new Comparison(rows, minCosine, maxRelative, rows == 0 ? 0 : (double) agree / rows);
    }
}
