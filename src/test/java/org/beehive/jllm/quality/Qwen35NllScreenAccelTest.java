package org.beehive.jllm.quality;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.List;
import org.beehive.jllm.backend.tornado.TornadoForwardPass;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.golden.TupleInfo;
import org.beehive.jllm.inference.Logits;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

// @formatter:off
/**
 * A <b>small quality screen</b>, not a representative quality benchmark.
 *
 * <p>Five fixed passages, teacher-forced, scored as negative log-likelihood. It exists to compare
 * one configuration of this engine against another — the packed-integer projections against the
 * floating-point ones — on text neither of them was tuned against. Five passages of a few hundred
 * tokens say something about whether a change is catastrophic and very little about whether it is
 * good; that distinction is the reason for the name.
 *
 * <p><b>Provenance.</b> The passages are the repository's own files, quoted from a recorded byte
 * offset: identifiable by path, licensed under the licence this repository carries, and fixed
 * before any measurement was taken. Their sha256 and byte range are printed with the result so a
 * later run can prove it scored the same text. They lean technical, which is a limitation worth
 * stating rather than hiding: prose, Java source, licence legalese, a changelog and architectural
 * documentation are varied in register but not in domain.
 *
 * <p><b>Alignment.</b> The row produced after consuming token {@code t} predicts token {@code t +
 * 1}, so a passage of {@code n} tokens scores {@code n - 1} and its first token is the unscored
 * prefix — the whole of it. No chat template, no begin-of-text, no padding: the passage's own
 * tokens, from position zero. {@link NllScoring} holds the arithmetic and is tested without a
 * model.
 *
 * <p>Run it once per configuration, in its own JVM, and compare the two files it writes. The packed
 * path is resolved when the layer builder class loads, so it cannot be switched inside one process.
 */
// @formatter:on
public class Qwen35NllScreenAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    /** Tokens per passage, after tokenizing from the recorded offset. */
    private static final int TOKENS = 256;

    /** Where the result lands, so the two configurations can be compared afterwards. */
    private static final String OUTPUT_PROPERTY = "jllm.nllScreen.out";

    // @formatter:off
    /**
     * Chunk width for the batched variant, or absent for the single-token one.
     *
     * <p>The two paths quantize different amounts. {@code STANDARD} packs every position, prompt
     * included; the batched plan ingests the prompt with its own prefill kernels and packs only the
     * rows it decodes. A screen of one says nothing about the other, and the parity envelopes are
     * separate for the same reason, so the batched variant ingests the first half of each passage
     * as a prompt and scores only the decoded half.
     */
    // @formatter:on
    private static final String BATCH_PROPERTY = "jllm.nllScreen.batch";

    /** One passage: a repository file, a byte offset, and the register it represents. */
    private record Passage(String name, String path, int byteOffset, int byteLength) {}

    private static final List<Passage> PASSAGES =
            List.of(
                    new Passage("prose", "README.md", 0, 4000),
                    new Passage(
                            "java-source",
                            "src/main/java/org/beehive/jllm/backend/tornado/kernels/"
                                    + "TransformerComputeKernelsQ4_0.java",
                            0,
                            4000),
                    new Passage("verification-prose", "docs/architecture/verification.md", 0, 4000),
                    new Passage("changelog", "CHANGELOG.md", 0, 4000),
                    new Passage("architecture", "docs/architecture/architecture.md", 0, 4000));

    @Test
    public void theScreenScoresEveryPassage() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println("[SKIP] environment absent");
            assumeTrue("environment absent", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] no TornadoVM device");
            assumeTrue("environment absent", false);
        }

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        StringBuilder report = new StringBuilder();
        try {
            int batch = Integer.getInteger(BATCH_PROPERTY, 1);
            if (!applies(batch)) {
                System.out.println("[SKIP] " + getClass().getSimpleName() + " at batch " + batch);
                assumeTrue("plan this screen describes not selected", false);
            }
            if (batch > 1) {
                // The plan is chosen from these, not from the state's width: sizing the state
                // alone leaves the single-token plan in place.
                System.setProperty("jllm.withPrefillDecode", "true");
                System.setProperty("jllm.prefillBatchSize", String.valueOf(batch));
            }
            Model model = ModelLoader.loadModel(modelPath, 1024, true, true);
            State state =
                    batch > 1
                            ? State.withPrefillBatchSize(batch, model::createNewState)
                            : model.createNewState();
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);

            boolean packed =
                    org.beehive.jllm.backend.tornado.device.TornadoDevices.current()
                                    .capabilities()
                                    .supports(
                                            org.beehive.jllm.runtime.backend.DeviceCapability
                                                    .PACKED_INTEGER_DOT)
                            && !"false"
                                    .equalsIgnoreCase(
                                            System.getProperty(
                                                    "jllm.qwen35.packedIntegerDot", "true"));
            report.append("model=").append(modelPath.getFileName()).append('\n');
            // Identity, not a digest of it: the fixture's sha256 is pinned in GoldenFixture and
            // checked there, and hashing 15 GiB to restate it here would be the only slow thing in
            // this test.
            report.append("modelBytes=").append(Files.size(modelPath)).append('\n');
            report.append("tokenizer=")
                    .append(model.tokenizer().getClass().getSimpleName())
                    .append('\n');
            report.append("packedIntegerDot=").append(packed).append('\n');
            report.append("prefillBatch=").append(batch).append('\n');
            // The execution path this screen actually scored, read off the plan and the layer
            // builder rather than assumed from the absence of a batch width. Which projections
            // read a quantized activation is the whole subject, so it is recorded, not inferred.
            report.append("plan=").append(plan.getClass().getSimpleName()).append('\n');
            // From this plan's own scheduler: the conversion tasks exist only on the MMA branch.
            var grids =
                    org.beehive.jllm.backend.tornado.PlanDispatchEvidence.gridSchedulerIfAvailable(
                            plan);
            report.append("mmaBatchedProjections=")
                    .append(
                            org.beehive.jllm.backend.tornado.PlanDispatchEvidence
                                    .qwen35MmaBatchedTasks(grids)
                                    .size())
                    .append('\n');
            // Which delta-rule kernels this plan dispatches, so a screen's report says what it
            // scored. The batched scan ingests the prompt; the decode scan produces the scored
            // rows. The grid alone cannot tell two batched scans of one geometry apart, so the
            // batched kernel is read off the task graph by its method name, and the decode task's
            // local work is labelled as the decode task's -- the batched one is never a stand-in.
            report.append("batchedDeltaRuleKernel=")
                    .append(batch > 1 ? batchedDeltaRuleKernel(plan) : "none")
                    .append('\n');
            report.append("batchedDeltaRuleLocalWork=")
                    .append(deltaRuleLocalWork(grids, "batchLayer_"))
                    .append('\n');
            report.append("decodeDeltaRuleLocalWork=")
                    .append(deltaRuleLocalWork(grids, "layer_"))
                    .append('\n');
            // The cache precision selects the attention kernel; this screen leaves the default in
            // place, so the report says which one it scored rather than implying the benchmarked
            // plan's.
            report.append("fp16KvCache=").append(state.usesFp16KeyValueCache()).append('\n');
            report.append("batchedAttentionKernel=")
                    .append(
                            batch > 1
                                    ? org.beehive.jllm.backend.tornado.PlanDispatchEvidence
                                            .batchedTaskKernels(plan, "attention")
                                    : java.util.Set.of("none"))
                    .append('\n');
            report.append("executionCombination=")
                    .append(org.beehive.jllm.auxiliary.RunMetrics.snapshot().executionCombination())
                    .append('\n');
            verifyDispatch(grids, batch, model.configuration().dim());
            if (batch > 1) {
                verifyBatchedScan(
                        batchedDeltaRuleKernel(plan),
                        ((org.beehive.jllm.model.qwen35.Qwen35Configuration) model.configuration())
                                .headValueDim());
            }

            double pooledNll = 0;
            long pooledTokens = 0;
            try {
                for (Passage passage : PASSAGES) {
                    byte[] raw = Files.readAllBytes(Paths.get(passage.path()));
                    assertTrue(
                            passage.path() + " is shorter than the recorded range",
                            raw.length >= passage.byteOffset() + passage.byteLength());
                    String text =
                            new String(
                                    raw,
                                    passage.byteOffset(),
                                    passage.byteLength(),
                                    StandardCharsets.UTF_8);
                    List<Integer> encoded = model.tokenizer().encodeAsList(text);
                    assertTrue(
                            passage.name()
                                    + " tokenizes to "
                                    + encoded.size()
                                    + ", fewer than "
                                    + TOKENS,
                            encoded.size() >= TOKENS);
                    int[] tokens = new int[TOKENS];
                    for (int i = 0; i < TOKENS; i++) {
                        tokens[i] = encoded.get(i);
                    }

                    // An independent sequence: the recurrent state carries no notion of position,
                    // so it is cleared on the device as well as on the host between passages.
                    plan.resetSequenceState();

                    int[][] scored = NllScoring.scoredPositions(tokens);
                    // The batched variant ingests the first half as a prompt, unscored, through
                    // the batched prefill kernels, and scores the decoded half.
                    int firstScored = batch > 1 ? TOKENS / 2 : 0;
                    if (batch > 1) {
                        var batchedPlan =
                                (org.beehive.jllm.backend.tornado
                                                .TornadoVMMasterPlanBatchPrefillDecode)
                                        plan;
                        for (int off = 0; off < firstScored; off += batch) {
                            int size = Math.min(batch, firstScored - off);
                            int[] chunk = java.util.Arrays.copyOfRange(tokens, off, off + size);
                            org.beehive.jllm.backend.tornado.TornadoBatchPrefillPass.batchPrefill(
                                    model, state, chunk, off, size, batchedPlan);
                        }
                    }
                    double sum = 0;
                    int counted = 0;
                    for (int[] pair : scored) {
                        if (pair[0] < firstScored) {
                            continue;
                        }
                        Logits logits =
                                batch > 1
                                        ? org.beehive.jllm.backend.tornado.TornadoBatchPrefillPass
                                                .decode(
                                                        model,
                                                        state,
                                                        tokens[pair[0]],
                                                        pair[0],
                                                        (org.beehive.jllm.backend.tornado
                                                                        .TornadoVMMasterPlanBatchPrefillDecode)
                                                                plan)
                                        : TornadoForwardPass.forward(
                                                model, state, tokens[pair[0]], pair[0], plan);
                        float[] row = new float[logits.size()];
                        for (int i = 0; i < row.length; i++) {
                            row[i] = logits.get(i);
                        }
                        sum += NllScoring.negativeLogLikelihood(row, pair[1]);
                        counted++;
                    }
                    scored = new int[counted][2];
                    double mean = sum / counted;
                    pooledNll += sum;
                    pooledTokens += scored.length;

                    report.append(
                            String.format(
                                    "passage=%s path=%s bytes=%d..%d sha256=%s scoredTokens=%d"
                                            + " nll=%.6f ppl=%.4f%n",
                                    passage.name(),
                                    passage.path(),
                                    passage.byteOffset(),
                                    passage.byteOffset() + passage.byteLength(),
                                    sha256(raw, passage.byteOffset(), passage.byteLength()),
                                    scored.length,
                                    mean,
                                    Math.exp(mean)));
                    System.out.print(report.substring(report.lastIndexOf("passage=")));
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }

            long expectedTokens =
                    (long) PASSAGES.size() * (batch > 1 ? TOKENS - 1 - TOKENS / 2 : TOKENS - 1);
            assertEquals(
                    "every passage scored the same number of positions",
                    expectedTokens,
                    pooledTokens);
            double pooledMean = pooledNll / pooledTokens;
            report.append(
                    String.format(
                            "pooled scoredTokens=%d nll=%.6f ppl=%.4f%n",
                            pooledTokens, pooledMean, Math.exp(pooledMean)));
            System.out.print(report.substring(report.lastIndexOf("pooled ")));

            String out = System.getProperty(OUTPUT_PROPERTY);
            if (out != null) {
                Files.writeString(Paths.get(out), report.toString());
                System.out.println("[NLL] wrote " + out);
            }
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    /**
     * What this screen's own plan must be built with for its numbers to describe the path claimed.
     * Nothing here; the tensor-core subclass overrides it.
     */
    /**
     * What batched delta-rule scan the plan must have compiled for this screen's numbers to
     * describe the path claimed. Nothing here; the tensor-core subclass overrides it.
     */
    protected void verifyBatchedScan(String kernel, int stateDim) {}

    /**
     * Local work size of the plan's {@code ssm_delta_rule} task in the graphs whose names start
     * with {@code graphPrefix} ({@code batchLayer_} for the batched prefill, {@code layer_} for
     * decode), or {@code -1} if there is none. Every layer's grid is the same, so the first is
     * reported.
     */
    private static long deltaRuleLocalWork(
            uk.ac.manchester.tornado.api.GridScheduler grids, String graphPrefix) {
        if (grids == null) {
            return -1;
        }
        for (String key : new java.util.TreeSet<>(grids.keySet())) {
            if (key.startsWith(graphPrefix) && key.endsWith(".ssm_delta_rule")) {
                return grids.get(key).getLocalWork()[0];
            }
        }
        return -1;
    }

    /**
     * The kernel method every batched layer's {@code ssm_delta_rule} task compiles, read off the
     * plan's task graphs; fails if the layers disagree, since one screen scores one kernel.
     */
    static String batchedDeltaRuleKernel(TornadoVMMasterPlan plan) {
        java.util.Set<String> kernels =
                org.beehive.jllm.backend.tornado.PlanDispatchEvidence.batchedTaskKernels(
                        plan, "ssm_delta_rule");
        assertEquals("one batched delta-rule kernel across the layers", 1, kernels.size());
        return kernels.iterator().next();
    }

    /**
     * Whether this screen has anything to say at the given prefill width. The base screen scores
     * any plan; a subclass that describes one particular plan declines the widths that select
     * another, so a suite run without the property skips it rather than failing it.
     */
    protected boolean applies(int batch) {
        return true;
    }

    protected void verifyDispatch(
            uk.ac.manchester.tornado.api.GridScheduler grids, int batch, int dim) {}

    private static String sha256(byte[] raw, int offset, int length) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(raw, offset, length);
            StringBuilder hex = new StringBuilder();
            for (byte b : digest.digest()) {
                hex.append(String.format("%02x", b));
            }
            return hex.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
    }
}
