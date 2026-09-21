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
 * Adoption-evaluation scoring for the int8 projection candidate, driven by a frozen manifest
 * ({@code jllm.eval.manifest}: path, byte range, sha256, prefix and scored token counts per
 * passage): a prefix ingested through the batched prefill in several chunks with a partial tail,
 * then 128 scored decode steps whose full logits rows are written out so two builds can be compared
 * position by position.
 *
 * <p>Properties: {@code jllm.eval.manifest} (required), {@code jllm.eval.out} (directory for the
 * per-position reports and the full logits rows, required), {@code jllm.eval.batch} (prefill width,
 * default 512); {@code jllm.kvcache.fp16} is set here. Each passage names its own prefix and scored
 * lengths; at width 512 every prefix in the frozen manifest is several full chunks plus a 128-token
 * partial tail. The first passage is scored twice with a reset between, and the two runs must be
 * raw-bit identical (replay determinism of the build under test).
 */
// @formatter:on
public class Qwen35AdoptionTextEvalAccelTest {

    static {
        System.setProperty("jllm.kvcache.fp16", "true");
    }

    private static final int CONTEXT = 4096;

    private record Passage(
            String name, String path, int byteOffset, int byteLength, int prefix, int scored) {}

    /** The frozen manifest (path, byte range, sha256, prefix and scored lengths per passage). */
    private static List<Passage> passages() throws IOException {
        String manifest = System.getProperty("jllm.eval.manifest");
        if (manifest == null) {
            return List.of();
        }
        java.util.ArrayList<Passage> out = new java.util.ArrayList<>();
        String text = Files.readString(Paths.get(manifest));
        java.util.regex.Matcher m =
                java.util.regex.Pattern.compile(
                                "\\{[^}]*\"id\": \"([^\"]+)\"[^}]*\"path\": \"([^\"]+)\"[^}]*\"byteOffset\": (\\d+)[^}]*\"byteLength\": (\\d+)[^}]*\"prefixTokens\": (\\d+)[^}]*\"scoredTokens\": (\\d+)[^}]*\"sha256\": \"([0-9a-f]+)\"")
                        .matcher(text);
        while (m.find()) {
            Passage p =
                    new Passage(
                            m.group(1),
                            m.group(2),
                            Integer.parseInt(m.group(3)),
                            Integer.parseInt(m.group(4)),
                            Integer.parseInt(m.group(5)),
                            Integer.parseInt(m.group(6)));
            byte[] raw = Files.readAllBytes(Paths.get(p.path()));
            String sha = sha256(raw, p.byteOffset(), p.byteLength());
            assertEquals(p.name() + " content hash", m.group(7), sha);
            out.add(p);
        }
        return out;
    }

    @Test
    public void theManifestPassagesAreScored() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        assumeTrue("environment absent", modelPath != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        int batch = Integer.getInteger("jllm.eval.batch", 512);
        String outDir = System.getProperty("jllm.eval.out");
        List<Passage> passages = passages();
        assumeTrue(
                "jllm.eval.manifest and jllm.eval.out select this run",
                outDir != null && !passages.isEmpty());
        Files.createDirectories(Paths.get(outDir));

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
                .append(" manifest=")
                .append(System.getProperty("jllm.eval.manifest"))
                .append(" context=")
                .append(CONTEXT)
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
            for (int p = 0; p < passages.size(); p++) {
                Passage passage = passages.get(p);
                final int PREFIX = passage.prefix();
                final int SCORED = passage.scored();
                final int TOKENS = PREFIX + SCORED + 1;
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
                float[][] rows = score(model, state, batched, tokens, batch, PREFIX, SCORED);
                if (p == 0) {
                    float[][] again = score(model, state, batched, tokens, batch, PREFIX, SCORED);
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
                                        "pos=%d target=%d nll=%.6f argmax=%d token=%d%n",
                                        position,
                                        target,
                                        nll,
                                        argmax,
                                        tokens[position]));
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

    /** Prefix through the batched prefill, then the scored teacher-forced decode rows. */
    private static float[][] score(
            Model model,
            State state,
            TornadoVMMasterPlanBatchPrefillDecode plan,
            int[] tokens,
            int batch,
            int PREFIX,
            int SCORED) {
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
