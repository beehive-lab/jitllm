package org.beehive.jllm.quality;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.List;
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
import org.junit.Test;

// @formatter:off
/**
 * Held-out teacher-forced scoring for the int8 projection investigation: passages the development
 * screens never used, a prefix ingested through the batched prefill in several chunks with a
 * partial tail, then 128 scored decode steps whose full logits rows are written out so two builds
 * can be compared position by position.
 *
 * <p>Properties: {@code jllm.heldout.batch} (prefill width, required), {@code jllm.heldout.out}
 * (directory for the reports and the logits rows, required), {@code jllm.kvcache.fp16} is set here.
 * The prefix is {@value #PREFIX} tokens: at width 512 that is two full chunks and a 128-token tail,
 * at width 1024 one full chunk and the same tail. The first passage is scored twice with a reset
 * between, and the two runs must be raw-bit identical (replay determinism of the build under test,
 * not of one build against another).
 */
// @formatter:on
public class Qwen35HeldOutNllAccelTest {

    static {
        System.setProperty("jllm.kvcache.fp16", "true");
    }

    private static final int PREFIX = 1152;
    private static final int SCORED = 128;
    private static final int TOKENS = PREFIX + SCORED + 1; // the last row needs a target
    private static final int CONTEXT = 2048;

    private record Passage(String name, String path, int byteOffset, int byteLength) {}

    /** Frozen before any candidate result was inspected. */
    private static final List<Passage> PASSAGES =
            List.of(
                    new Passage("prose-development", "docs/architecture/development.md", 0, 7770),
                    new Passage("prose-api", "docs/architecture/api.md", 0, 6768),
                    new Passage(
                            "code-java",
                            "src/main/java/org/beehive/jllm/bench/JllmBench.java",
                            0,
                            9000),
                    new Passage("code-python", "scripts/perf_gate.py", 0, 9000),
                    new Passage("structured-yaml", ".github/workflows/build-and-run.yml", 0, 9000),
                    new Passage("structured-xml", "pom.xml", 0, 9000),
                    new Passage("numeric-jsonl", "docs/perf-history.jsonl", 0, 7000));

    /**
     * A second set, never used by any screen or by the set above, selected with {@code
     * -Djllm.heldout.set=fresh}; frozen with the revised acceptance proposal before either build
     * was scored on it.
     */
    private static final List<Passage> FRESH =
            List.of(
                    new Passage(
                            "prose-verification-tail",
                            "docs/architecture/verification.md",
                            12000,
                            15166),
                    new Passage("prose-handoff", "/home/orion/jllm/HANDOFF-JLLM-CODEX.md", 0, 8633),
                    new Passage(
                            "code-java-loop",
                            "src/main/java/org/beehive/jllm/inference/TokenGenerationLoop.java",
                            20000,
                            10000),
                    new Passage(
                            "code-java-app",
                            "src/main/java/org/beehive/jllm/JllmApp.java",
                            0,
                            7840),
                    new Passage("code-python-tests", "scripts/tests/test_perf_gate.py", 0, 9000),
                    new Passage(
                            "structured-yaml2",
                            ".github/workflows/standalone-inference.yml",
                            0,
                            9000),
                    new Passage("structured-html", "docs/index.html", 0, 8769),
                    new Passage("numeric-jsonl2", "docs/perf-history.jsonl", 1500000, 7000));

    private static List<Passage> passages() {
        return "fresh".equals(System.getProperty("jllm.heldout.set")) ? FRESH : PASSAGES;
    }

    @Test
    public void theHeldOutPassagesAreScored() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        assumeTrue("environment absent", modelPath != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        int batch = Integer.getInteger("jllm.heldout.batch", 0);
        String outDir = System.getProperty("jllm.heldout.out");
        assumeTrue(
                "jllm.heldout.batch and jllm.heldout.out select this run",
                batch > 1 && outDir != null);
        Files.createDirectories(Paths.get(outDir));

        // Diagnostic seams (test-only): keep named projections on FP16, or feed the FP16 pair
        // quantized-dequantized activations.
        String exclude = System.getProperty("jllm.heldout.int8exclude", "");
        if (!exclude.isEmpty()) {
            java.util.Set<String> ex = java.util.Set.of(exclude.split(","));
            org.beehive.jllm.backend.tornado.layers.Qwen35BatchPrefillLayers
                            .int8TaskFilterForTests =
                    task -> !ex.contains(task);
        }
        org.beehive.jllm.backend.tornado.layers.Qwen35BatchPrefillLayers.fakeQuantizeForTests =
                Boolean.getBoolean("jllm.heldout.fakequant");
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(batch));
        Model model = ModelLoader.loadModel(modelPath, CONTEXT, true, true);
        State state = State.withPrefillBatchSize(batch, model::createNewState);
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        var batched = (TornadoVMMasterPlanBatchPrefillDecode) plan;
        StringBuilder summary = new StringBuilder();
        summary.append("build=int8-on-tensor-cores")
                .append(" batch=")
                .append(batch)
                .append(" prefix=")
                .append(PREFIX)
                .append(" scored=")
                .append(SCORED)
                .append(" set=")
                .append(System.getProperty("jllm.heldout.set", "original"))
                .append('\n');
        summary.append("pairs=")
                .append(
                        org.beehive.jllm.backend.tornado.PlanDispatchEvidence
                                .batchedTaskKernelsIfAny(plan, "ffn_gate_proj"))
                .append(" gateUp=")
                .append(
                        org.beehive.jllm.backend.tornado.PlanDispatchEvidence
                                .batchedTaskKernelsIfAny(plan, "ffn_gate_up"))
                .append('\n');
        double pooled = 0;
        long pooledTokens = 0;
        try {
            for (int p = 0; p < passages().size(); p++) {
                Passage passage = passages().get(p);
                byte[] raw = Files.readAllBytes(Paths.get(passage.path()));
                assertTrue(
                        passage.path() + " shorter than the recorded range",
                        raw.length >= passage.byteOffset() + passage.byteLength());
                String text =
                        new String(
                                raw,
                                passage.byteOffset(),
                                passage.byteLength(),
                                StandardCharsets.UTF_8);
                List<Integer> encoded = model.tokenizer().encodeAsList(text);
                assertTrue(
                        passage.name() + " tokenizes to " + encoded.size() + " < " + TOKENS,
                        encoded.size() >= TOKENS);
                int[] tokens = new int[TOKENS];
                for (int i = 0; i < TOKENS; i++) {
                    tokens[i] = encoded.get(i);
                }
                float[][] rows = score(model, state, batched, tokens, batch);
                if (p == 0) {
                    float[][] again = score(model, state, batched, tokens, batch);
                    for (int r = 0; r < rows.length; r++) {
                        for (int i = 0; i < rows[r].length; i++) {
                            assertEquals(
                                    "replay of " + passage.name() + " row " + r + " logit " + i,
                                    Float.floatToRawIntBits(rows[r][i]),
                                    Float.floatToRawIntBits(again[r][i]));
                        }
                    }
                    summary.append("replay=" + passage.name() + " raw-bit identical\n");
                }
                double sum = 0;
                Path rowsFile = Paths.get(outDir, passage.name() + ".rows.bin");
                Path posFile = Paths.get(outDir, passage.name() + ".pos.txt");
                StringBuilder pos = new StringBuilder();
                try (DataOutputStream out =
                        new DataOutputStream(new FileOutputStream(rowsFile.toFile()))) {
                    ByteBuffer buf =
                            ByteBuffer.allocate(4 * rows[0].length).order(ByteOrder.LITTLE_ENDIAN);
                    for (int r = 0; r < rows.length; r++) {
                        int position = PREFIX + r;
                        int target = tokens[position + 1];
                        double nll = NllScoring.negativeLogLikelihood(rows[r], target);
                        sum += nll;
                        int argmax = 0;
                        for (int i = 1; i < rows[r].length; i++) {
                            if (rows[r][i] > rows[r][argmax]) {
                                argmax = i;
                            }
                        }
                        pos.append(
                                String.format(
                                        java.util.Locale.ROOT,
                                        "pos=%d target=%d nll=%.6f argmax=%d%n",
                                        position,
                                        target,
                                        nll,
                                        argmax));
                        buf.clear();
                        for (float v : rows[r]) {
                            buf.putFloat(v);
                        }
                        out.write(buf.array());
                    }
                }
                Files.writeString(posFile, pos.toString());
                double mean = sum / rows.length;
                pooled += sum;
                pooledTokens += rows.length;
                String line =
                        String.format(
                                java.util.Locale.ROOT,
                                "passage=%s path=%s bytes=%d..%d sha256=%s scored=%d nll=%.6f ppl=%.4f%n",
                                passage.name(),
                                passage.path(),
                                passage.byteOffset(),
                                passage.byteOffset() + passage.byteLength(),
                                sha256(raw, passage.byteOffset(), passage.byteLength()),
                                rows.length,
                                mean,
                                Math.exp(mean));
                summary.append(line);
                System.out.print(line);
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
        double mean = pooled / pooledTokens;
        String line =
                String.format(
                        java.util.Locale.ROOT,
                        "pooled scored=%d nll=%.6f ppl=%.4f%n",
                        pooledTokens,
                        mean,
                        Math.exp(mean));
        summary.append(line);
        System.out.print(line);
        Files.writeString(Paths.get(outDir, "summary.txt"), summary.toString());
    }

    /** Prefix through the batched prefill, then {@link #SCORED} teacher-forced decode rows. */
    private static float[][] score(
            Model model,
            State state,
            TornadoVMMasterPlanBatchPrefillDecode plan,
            int[] tokens,
            int batch) {
        plan.resetSequenceState();
        for (int off = 0; off < PREFIX; off += batch) {
            int size = Math.min(batch, PREFIX - off);
            int[] chunk = java.util.Arrays.copyOfRange(tokens, off, off + size);
            TornadoBatchPrefillPass.batchPrefill(model, state, chunk, off, size, plan);
        }
        float[][] rows = new float[SCORED][];
        for (int r = 0; r < SCORED; r++) {
            int position = PREFIX + r;
            Logits logits =
                    TornadoBatchPrefillPass.decode(model, state, tokens[position], position, plan);
            float[] row = new float[logits.size()];
            for (int i = 0; i < row.length; i++) {
                row[i] = logits.get(i);
            }
            rows[r] = row;
        }
        return rows;
    }

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
