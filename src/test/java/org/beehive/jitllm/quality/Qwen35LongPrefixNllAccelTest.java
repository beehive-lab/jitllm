package org.beehive.jllm.quality;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.List;
import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
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
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.junit.Test;

// @formatter:off
/**
 * A teacher-forced comparison of two builds of the batched prefill <b>after a long prefix</b>: one
 * fixed passage, a prefix longer than the prefill width so the batched kernels carry the recurrent
 * state and the KV cache across at least one chunk boundary, then 128 decoded positions scored as
 * negative log-likelihood. It complements {@link Qwen35NllScreenAccelTest}, whose batched variant
 * prefills only the first 128 tokens of each passage -- one chunk at any width from 128 up, so that
 * screen never exercises a chunk boundary and says nothing about how a change to the batched scan
 * accumulates over a long prompt.
 *
 * <p>It is a comparison, not a bound: it writes a report and asserts only that the input was what
 * it says. Run it once per build in its own JVM with the same width and diff the two files. The
 * report records the text's sha256 and byte range, the sha256 of the token ids, the chunk schedule,
 * the kernels the plan dispatched for the batched scan and attention, every scored position's NLL,
 * and the sha256 of those NLLs, so two reports either agree bit for bit or show where they part.
 *
 * <p>Width from {@code jllm.nllScreen.batch} (default 256); the prefix is one and a half widths
 * plus 64 tokens, so the last chunk is partial as well. Output to {@code jllm.nllScreen.out}.
 * Tensor cores and the FP16 KV cache on, as the benchmarked plan runs.
 */
// @formatter:on
public class Qwen35LongPrefixNllAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
        // The benchmarked plan's cache precision: it selects the attention kernel, so it is set
        // here and read back into the report from the state rather than assumed.
        System.setProperty("jllm.kvcache.fp16", "true");
    }

    private static final String OUTPUT_PROPERTY = "jllm.nllScreen.out";
    private static final String BATCH_PROPERTY = "jllm.nllScreen.batch";

    /**
     * Context the plan is sized for: 2048, or twice the width where the prefix would not fit (width
     * 2048 needs 3136 + 128 positions).
     */
    private static final int CONTEXT = Math.max(2048, 2 * Integer.getInteger(BATCH_PROPERTY, 256));

    /** Decoded positions scored after the prefix. */
    private static final int SCORED = 128;

    /** One passage, long enough for the widest prefix: the repository's README from byte zero. */
    private static final String PASSAGE_PATH = "README.md";

    private static final int PASSAGE_BYTES = 16000;

    @Test
    public void scoresAfterALongPrefix() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println("[SKIP] environment absent");
            assumeTrue("environment absent", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] no TornadoVM device");
            assumeTrue("environment absent", false);
        }
        int batch = Integer.getInteger(BATCH_PROPERTY, 256);
        int prefix = batch + batch / 2 + 64;
        assertTrue("prefix must cross a chunk boundary", prefix > batch);
        assertTrue("prefix and scored rows must fit the context", prefix + SCORED <= CONTEXT);

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(batch));
        StringBuilder report = new StringBuilder();
        try {
            Model model = ModelLoader.loadModel(modelPath, CONTEXT, true, true);
            State state = State.withPrefillBatchSize(batch, model::createNewState);
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            assertTrue(
                    "the batched plan: " + plan.getClass().getSimpleName(),
                    plan instanceof TornadoVMMasterPlanBatchPrefillDecode);
            var batchedPlan = (TornadoVMMasterPlanBatchPrefillDecode) plan;
            var config = (Qwen35Configuration) model.configuration();

            byte[] raw = Files.readAllBytes(Paths.get(PASSAGE_PATH));
            assertTrue(
                    PASSAGE_PATH + " is shorter than the recorded range",
                    raw.length >= PASSAGE_BYTES);
            String text = new String(raw, 0, PASSAGE_BYTES, StandardCharsets.UTF_8);
            List<Integer> encoded = model.tokenizer().encodeAsList(text);
            int needed = prefix + SCORED + 1;
            assertTrue(
                    "passage tokenizes to " + encoded.size() + ", fewer than " + needed,
                    encoded.size() >= needed);
            int[] tokens = new int[needed];
            for (int i = 0; i < needed; i++) {
                tokens[i] = encoded.get(i);
            }

            report.append("model=").append(modelPath.getFileName()).append('\n');
            report.append("modelBytes=").append(Files.size(modelPath)).append('\n');
            report.append("plan=").append(plan.getClass().getSimpleName()).append('\n');
            report.append("prefillBatch=").append(batch).append('\n');
            report.append("context=").append(CONTEXT).append('\n');
            report.append("prefixTokens=").append(prefix).append('\n');
            report.append("scoredTokens=").append(SCORED).append('\n');
            report.append("passage=")
                    .append(PASSAGE_PATH)
                    .append(" bytes=0..")
                    .append(PASSAGE_BYTES)
                    .append(" sha256=")
                    .append(sha256(raw, 0, PASSAGE_BYTES))
                    .append('\n');
            report.append("tokenIdsSha256=").append(sha256(tokens)).append('\n');
            var grids = PlanDispatchEvidence.gridSchedulerIfAvailable(plan);
            report.append("batchedDeltaRuleKernel=")
                    .append(PlanDispatchEvidence.batchedTaskKernels(plan, "ssm_delta_rule"))
                    .append('\n');
            report.append("batchedAttentionKernel=")
                    .append(PlanDispatchEvidence.batchedTaskKernels(plan, "attention"))
                    .append('\n');
            report.append("mmaBatchedProjections=")
                    .append(PlanDispatchEvidence.qwen35MmaBatchedTasks(grids).size())
                    .append('\n');
            report.append("fp16KvCache=").append(state.usesFp16KeyValueCache()).append('\n');
            report.append("stateDim=").append(config.headValueDim()).append('\n');

            double[] nll = new double[SCORED];
            int argmaxHits = 0;
            try {
                plan.resetSequenceState();
                StringBuilder schedule = new StringBuilder();
                for (int off = 0; off < prefix; off += batch) {
                    int size = Math.min(batch, prefix - off);
                    int[] chunk = java.util.Arrays.copyOfRange(tokens, off, off + size);
                    TornadoBatchPrefillPass.batchPrefill(
                            model, state, chunk, off, size, batchedPlan);
                    schedule.append(off).append('+').append(size).append(' ');
                }
                report.append("chunks=").append(schedule.toString().trim()).append('\n');

                for (int i = 0; i < SCORED; i++) {
                    int position = prefix + i;
                    Logits logits =
                            TornadoBatchPrefillPass.decode(
                                    model, state, tokens[position], position, batchedPlan);
                    float[] row = new float[logits.size()];
                    int argmax = 0;
                    for (int j = 0; j < row.length; j++) {
                        row[j] = logits.get(j);
                        if (row[j] > row[argmax]) {
                            argmax = j;
                        }
                    }
                    int target = tokens[position + 1];
                    nll[i] = NllScoring.negativeLogLikelihood(row, target);
                    if (argmax == target) {
                        argmaxHits++;
                    }
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }

            double sum = 0;
            for (double v : nll) {
                sum += v;
            }
            double mean = sum / SCORED;
            report.append("argmaxMatchesTarget=").append(argmaxHits).append('\n');
            report.append(
                    String.format(
                            "pooled scoredTokens=%d nll=%.6f ppl=%.4f%n",
                            SCORED, mean, Math.exp(mean)));
            report.append("nllSha256=").append(sha256(nll)).append('\n');
            for (int i = 0; i < SCORED; i++) {
                report.append(
                        String.format(
                                "pos=%d target=%d nll=%.6f bits=%016x%n",
                                prefix + i,
                                tokens[prefix + i + 1],
                                nll[i],
                                Double.doubleToLongBits(nll[i])));
            }
            System.out.print(report.substring(0, report.indexOf("pos=")));

            String out = System.getProperty(OUTPUT_PROPERTY);
            if (out != null) {
                Files.writeString(Paths.get(out), report.toString());
                System.out.println("[NLL] wrote " + out);
            }
            assertEquals("every scored position produced a finite NLL", SCORED, finite(nll));
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static int finite(double[] values) {
        int count = 0;
        for (double v : values) {
            if (Double.isFinite(v)) {
                count++;
            }
        }
        return count;
    }

    private static String sha256(byte[] raw, int offset, int length) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(raw, offset, length);
        return hex(digest.digest());
    }

    private static String sha256(int[] values) throws Exception {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            buffer.putInt(v);
        }
        return sha256(buffer.array(), 0, buffer.capacity());
    }

    private static String sha256(double[] values) throws Exception {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : values) {
            buffer.putDouble(v);
        }
        return sha256(buffer.array(), 0, buffer.capacity());
    }

    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder();
        for (byte b : bytes) {
            out.append(String.format("%02x", b));
        }
        return out.toString();
    }
}
