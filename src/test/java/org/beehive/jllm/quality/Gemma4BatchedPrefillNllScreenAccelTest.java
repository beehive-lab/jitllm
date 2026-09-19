package org.beehive.jllm.quality;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.backend.tornado.TornadoBatchPrefillPass;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.golden.TupleInfo;
import org.beehive.jllm.inference.Logits;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

// @formatter:off
/**
 * A bounded teacher-forced quality screen for Gemma 4's <b>batched</b> prefill.
 *
 * <p>The batched path does not compute the same arithmetic as the single-token one and does not
 * claim to. Its projections are tensor-core GEMMs with FP16 operands and FP32 accumulation over
 * weights decoded once into an FP16 scratch; two of them additionally split the depth across blocks
 * and sum the slices at the end, which reassociates the sum over K. The checks that came with each
 * of those changes were comparisons against the build before it — useful for catching a mistake in
 * the change, and silent about the accumulated distance from the reference.
 *
 * <p>This is the check that is not relative: it scores held-out text against the <b>host</b>, which
 * decodes the file's own blocks in FP32 and shares no kernel with any of it. A prefix goes through
 * the batched graphs, the scored positions then go through the decode graphs, and the host is
 * teacher-forced along the same token ids so that every compared position has seen exactly the same
 * history.
 *
 * <p>The prefix is deliberately longer than one chunk. A screen that fits inside a single chunk
 * would never exercise the boundary, and the boundary is where a batched prefill's mistakes live:
 * the position a row believes it is at, the window it attends, and what the key/value cache holds
 * when the next chunk starts.
 *
 * <p>What this is not: a quality benchmark. It is two builds on fixed inputs, and the bound is a
 * regression limit around a measured difference rather than a claim about the model.
 */
// @formatter:on
public class Gemma4BatchedPrefillNllScreenAccelTest {

    /** Chunk width the batched graphs are built for. */
    private static final int BATCH = 256;

    /** Prefix pushed through the batched graphs — one full chunk and part of a second. */
    private static final int PREFIX = 320;

    /** Positions scored after the prefix. */
    private static final int SCORED = 48;

    /**
     * How much worse the batched path's pooled NLL may be than the host's, in nats per token.
     *
     * <p>0.02 nats is about a 2% change in perplexity, the same limit the packed-activation screen
     * uses. A regression limit around a measured difference, not a calibrated quality threshold —
     * the measured value prints on every run, so drift toward the limit is visible long before it
     * trips.
     */
    private static final double MAX_POOLED_NLL_INCREASE = 0.02;

    private record Passage(String name, String path, int byteOffset, int byteLength) {}

    private static final List<Passage> PASSAGES =
            List.of(
                    new Passage("prose", "README.md", 0, 6000),
                    new Passage("architecture", "docs/architecture/architecture.md", 0, 6000));

    @Test
    public void theBatchedPrefillDoesNotMakeHeldOutTextLessLikely() throws Exception {
        screen(Fixture.GEMMA_4_E2B_Q8_0);
    }

    static void screen(Fixture fixture) throws Exception {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: fixture", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] environment absent — no TornadoVM device");
            assumeTrue("environment absent: no accelerator", false);
        }

        String prevTornado = System.getProperty("use.tornadovm");
        String prevPrefill = System.getProperty("jllm.withPrefillDecode");
        String prevBatch = System.getProperty("jllm.prefillBatchSize");
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", Integer.toString(BATCH));

        double batchedTotal = 0;
        double hostTotal = 0;
        long counted = 0;
        try {
            Model gpuModel = ModelLoader.loadModel(modelPath, 2048, true, true);
            State gpuState = gpuModel.createNewState();
            TornadoVMMasterPlan plan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(gpuState, gpuModel);
            assertTrue(
                    "the batched plan is what this screen exists to score, and it was not selected",
                    plan instanceof TornadoVMMasterPlanBatchPrefillDecode);
            TornadoVMMasterPlanBatchPrefillDecode batched =
                    (TornadoVMMasterPlanBatchPrefillDecode) plan;
            Model cpuModel = ModelLoader.loadModel(modelPath, 2048, true, false);

            System.out.printf(
                    "[NLL] model=%s device=%s batch=%d prefix=%d scored=%d%n",
                    modelPath.getFileName(), TupleInfo.deviceName(), BATCH, PREFIX, SCORED);
            try {
                for (Passage passage : PASSAGES) {
                    int[] tokens = tokenize(gpuModel, passage);
                    State cpuState = cpuModel.createNewState();

                    // The prefix through the batched graphs, in chunks, exactly as a prompt goes.
                    batched.resetSequenceState();
                    for (int start = 0; start < PREFIX; start += BATCH) {
                        int size = Math.min(BATCH, PREFIX - start);
                        int[] chunk = new int[size];
                        System.arraycopy(tokens, start, chunk, 0, size);
                        TornadoBatchPrefillPass.batchPrefill(
                                gpuModel, gpuState, chunk, start, size, batched);
                    }
                    // The host sees the same prefix one token at a time.
                    for (int i = 0; i < PREFIX; i++) {
                        InferenceCore.forwardJavaGemma4(cpuModel, cpuState, tokens[i], i);
                    }

                    double batchedSum = 0;
                    double hostSum = 0;
                    for (int p = PREFIX; p < PREFIX + SCORED; p++) {
                        Logits deviceLogits =
                                TornadoBatchPrefillPass.decode(
                                        gpuModel, gpuState, tokens[p], p, batched);
                        batchedSum +=
                                NllScoring.negativeLogLikelihood(
                                        toRow(deviceLogits), tokens[p + 1]);

                        FloatTensor hostLogits =
                                InferenceCore.forwardJavaGemma4(cpuModel, cpuState, tokens[p], p);
                        hostSum +=
                                NllScoring.negativeLogLikelihood(toRow(hostLogits), tokens[p + 1]);
                    }
                    batchedTotal += batchedSum;
                    hostTotal += hostSum;
                    counted += SCORED;
                    System.out.printf(
                            "[NLL] %-14s scored=%d  host=%.5f  batched=%.5f  delta=%+.5f"
                                    + " nats/token%n",
                            passage.name(),
                            SCORED,
                            hostSum / SCORED,
                            batchedSum / SCORED,
                            (batchedSum - hostSum) / SCORED);
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            restore("use.tornadovm", prevTornado);
            restore("jllm.withPrefillDecode", prevPrefill);
            restore("jllm.prefillBatchSize", prevBatch);
        }

        double hostMean = hostTotal / counted;
        double batchedMean = batchedTotal / counted;
        double delta = batchedMean - hostMean;
        System.out.printf(
                "[NLL] pooled over %d tokens: host=%.5f  batched=%.5f  delta=%+.5f nats/token"
                        + " (perplexity ratio %.4f)%n",
                counted, hostMean, batchedMean, delta, Math.exp(delta));
        assertTrue(
                String.format(
                        "batched prefill raised pooled NLL by %+.5f nats/token, above the %.3f"
                                + " regression limit (host %.5f, batched %.5f over %d tokens)",
                        delta, MAX_POOLED_NLL_INCREASE, hostMean, batchedMean, counted),
                delta <= MAX_POOLED_NLL_INCREASE);
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static int[] tokenize(Model model, Passage passage) throws Exception {
        byte[] raw = Files.readAllBytes(Paths.get(passage.path()));
        assertTrue(
                passage.path() + " is shorter than the recorded range",
                raw.length >= passage.byteOffset() + passage.byteLength());
        String text =
                new String(raw, passage.byteOffset(), passage.byteLength(), StandardCharsets.UTF_8);
        List<Integer> encoded = model.tokenizer().encodeAsList(text);
        int needed = PREFIX + SCORED + 1;
        assertTrue(
                passage.name() + " tokenizes to " + encoded.size() + ", fewer than " + needed,
                encoded.size() >= needed);
        int[] tokens = new int[needed];
        for (int i = 0; i < needed; i++) {
            tokens[i] = encoded.get(i);
        }
        return tokens;
    }

    private static float[] toRow(Logits logits) {
        float[] row = new float[logits.size()];
        for (int i = 0; i < row.length; i++) {
            row[i] = logits.get(i);
        }
        return row;
    }

    private static float[] toRow(FloatTensor logits) {
        float[] row = new float[logits.size()];
        for (int i = 0; i < row.length; i++) {
            row[i] = logits.getFloat(i);
        }
        return row;
    }
}
