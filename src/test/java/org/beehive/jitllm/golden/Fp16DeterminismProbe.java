package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Diagnostic for the FP16 run-to-run determinism defect. Read-only: it changes no production code
 * and is never run by a gate.
 *
 * <p>The probe repeats an identical single forward pass over a fixed prompt and compares the
 * per-layer KV cache between runs. The KV cache is the useful observation point because, unlike the
 * activation scratch buffers, it is written once per layer per position and not overwritten by
 * later layers — so the lowest layer index whose K or V differs localises the <b>earliest</b>
 * diverging operation rather than just the final symptom.
 *
 * <p>Usage: {@code -Dprobe.model=<gguf> [-Dprobe.runs=5] [-Dprobe.gpu=true] [-Dprobe.tokens=0]}
 */
public final class Fp16DeterminismProbe {

    private static final String PROMPT =
            "Explain what a matrix multiplication is in one paragraph.";

    public static void main(String[] args) throws Exception {
        Path model = Paths.get(System.getProperty("probe.model"));
        int runs = Integer.getInteger("probe.runs", 5);
        boolean gpu = Boolean.parseBoolean(System.getProperty("probe.gpu", "true"));
        int extraTokens = Integer.getInteger("probe.tokens", 0);

        System.out.println(
                "probe model="
                        + model.getFileName()
                        + " runs="
                        + runs
                        + " gpu="
                        + gpu
                        + " extraTokens="
                        + extraTokens);

        Model m = ModelLoader.loadModel(model, 512, true, gpu);
        int layers = m.configuration().numberOfLayers();

        List<float[]> keys = new ArrayList<>();
        List<float[]> values = new ArrayList<>();
        List<float[]> logits = new ArrayList<>();
        List<List<Integer>> tokens = new ArrayList<>();
        List<float[]> wrapX = new ArrayList<>();
        List<float[]> tempLogits = new ArrayList<>();
        List<float[]> xbFp16 = new ArrayList<>();
        List<List<float[]>> rowsPerRun = new ArrayList<>();

        for (int r = 0; r < runs; r++) {
            State state = m.createNewState();
            ChatFormat cf = m.chatFormat();
            List<Integer> prompt = new ArrayList<>();
            if (m.shouldAddBeginOfText()) {
                prompt.add(cf.getBeginOfText());
            }
            prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, PROMPT)));
            prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

            List<Integer> got = new ArrayList<>();
            List<float[]> allRows = new ArrayList<>();
            float[][] lastLogits = new float[1][];
            Sampler capture =
                    t -> {
                        lastLogits[0] = toArray(t);
                        allRows.add(lastLogits[0]);
                        int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
                        got.add(tok);
                        return tok;
                    };

            TornadoVMMasterPlan plan = null;
            try {
                if (gpu) {
                    plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, m);
                    m.generateTokensGPU(
                            state,
                            0,
                            prompt,
                            Set.of(),
                            prompt.size() + 1 + extraTokens,
                            capture,
                            false,
                            null,
                            plan);
                } else {
                    m.generateTokens(
                            state,
                            0,
                            prompt,
                            Set.of(),
                            prompt.size() + 1 + extraTokens,
                            capture,
                            false,
                            null);
                }
                keys.add(snapshot(state.workspace.wrapKeyCache));
                values.add(snapshot(state.workspace.wrapValueCache));
                logits.add(lastLogits[0]);
                rowsPerRun.add(allRows);
                tokens.add(got);
                // Walk the tail of the graph: final hidden state -> RMS partial sums ->
                // normalized FP16 input -> logits. The first of these that differs is the
                // earliest divergent operation.
                wrapX.add(snapshot(state.workspace.wrapX));
                tempLogits.add(snapshot(state.workspace.tempLogits));
                xbFp16.add(snapshotHalf(state.workspace.wrapXbFP16));
            } finally {
                if (plan != null) {
                    plan.freeTornadoExecutionPlan();
                }
            }
            System.out.println("  run " + r + " done, first token=" + got.get(0));
        }

        int kNonZero = 0;
        for (float v : keys.get(0)) {
            if (v != 0f) {
                kNonZero++;
            }
        }
        int vNonZero = 0;
        for (float v : values.get(0)) {
            if (v != 0f) {
                vNonZero++;
            }
        }
        System.out.println(
                "KV read-back check: K nonZero="
                        + kNonZero
                        + " V nonZero="
                        + vNonZero
                        + (kNonZero == 0
                                ? "  <-- K NOT READ BACK, layer result inconclusive"
                                : ""));

        int kvPerLayer = keys.get(0).length / layers;
        System.out.println(
                "layers="
                        + layers
                        + " kvCacheLen="
                        + keys.get(0).length
                        + " perLayer="
                        + kvPerLayer);

        System.out.println(
                "\n=== token sequences identical across runs: "
                        + tokens.stream().distinct().count()
                        + " distinct ===");

        System.out.println("\n=== earliest divergent layer (K cache) ===");
        reportEarliestLayer("K", keys, layers, kvPerLayer);
        System.out.println("\n=== earliest divergent layer (V cache) ===");
        reportEarliestLayer("V", values, layers, kvPerLayer);

        System.out.println("\n=== tail-of-graph: which buffer diverges first ===");
        reportBuffer("wrapX      (final hidden state, pre-norm)", wrapX);
        reportBuffer("tempLogits (RMS reduction partials)      ", tempLogits);
        reportBuffer("wrapXbFP16 (normalized input to vocab)   ", xbFp16);

        // Compare EVERY captured row, not just the last. Comparing only the final row is what
        // made an intermittent defect look absent earlier.
        // Cross-process fingerprint: hash every captured row. Comparing this between separate
        // JVM invocations separates intra-process non-determinism from inter-process variation.
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            for (float[] row : rowsPerRun.get(0)) {
                java.nio.ByteBuffer bb =
                        java.nio.ByteBuffer.allocate(row.length * 4)
                                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                for (float v : row) {
                    bb.putInt(Float.floatToRawIntBits(v));
                }
                md.update(bb.array());
            }
            System.out.println(
                    "PROCESS_FINGERPRINT " + java.util.HexFormat.of().formatHex(md.digest()));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }

        System.out.println("\n=== all-row reproducibility vs run 0 ===");
        for (int r = 1; r < rowsPerRun.size(); r++) {
            int badRows = 0;
            int firstBad = -1;
            double worst = 0;
            for (int row = 0; row < rowsPerRun.get(0).size(); row++) {
                float[] a = rowsPerRun.get(0).get(row);
                float[] b = rowsPerRun.get(r).get(row);
                boolean differs = false;
                for (int i = 0; i < a.length; i++) {
                    if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                        differs = true;
                        worst = Math.max(worst, Math.abs(a[i] - b[i]));
                    }
                }
                if (differs) {
                    badRows++;
                    if (firstBad < 0) {
                        firstBad = row;
                    }
                }
            }
            System.out.printf(
                    "  run0 vs run%d: divergentRows=%d/%d firstDivergentRow=%d worstAbs=%.6g %s%n",
                    r,
                    badRows,
                    rowsPerRun.get(0).size(),
                    firstBad,
                    worst,
                    badRows == 0 ? "REPRODUCIBLE" : "*** NOT REPRODUCIBLE ***");
        }

        // Cross-contamination check: capture a SECOND model in the same JVM, after the first.
        // Surefire runs the F16 and Q8_0 golden tests in one JVM, F16 first, so if a preceding
        // non-deterministic run perturbs a later one this is where it shows.
        String second = System.getProperty("probe.model2");
        if (second != null) {
            GoldenCapture.Result r2 = GoldenCapture.capture(Paths.get(second), true);
            try {
                java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
                for (float[] row : r2.rows) {
                    java.nio.ByteBuffer bb =
                            java.nio.ByteBuffer.allocate(row.length * 4)
                                    .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                    for (float v : row) {
                        bb.putInt(Float.floatToRawIntBits(v));
                    }
                    md.update(bb.array());
                }
                System.out.println(
                        "SECOND_MODEL_FINGERPRINT "
                                + java.util.HexFormat.of().formatHex(md.digest()));
            } catch (java.security.NoSuchAlgorithmException e) {
                throw new IllegalStateException(e);
            }
        }

        System.out.println("\n=== logits drift across runs (pairwise vs run 0) ===");
        float[] base = logits.get(0);
        for (int r = 1; r < logits.size(); r++) {
            stats("logits run0 vs run" + r, base, logits.get(r));
        }
    }

    private static void reportEarliestLayer(
            String tag, List<float[]> snaps, int layers, int perLayer) {
        for (int layer = 0; layer < layers; layer++) {
            int from = layer * perLayer;
            int to = from + perLayer;
            int differing = 0;
            double maxAbs = 0;
            for (int r = 1; r < snaps.size(); r++) {
                float[] a = snaps.get(0);
                float[] b = snaps.get(r);
                for (int i = from; i < to; i++) {
                    if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                        differing++;
                        maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
                    }
                }
            }
            if (differing > 0) {
                System.out.printf(
                        "  %s: earliest divergent layer = %d  (differing=%d, maxAbs=%.6g)%n",
                        tag, layer, differing, maxAbs);
                return;
            }
        }
        System.out.println("  " + tag + ": no divergence in any layer");
    }

    private static void reportBuffer(String label, List<float[]> snaps) {
        if (snaps.isEmpty() || snaps.get(0) == null) {
            System.out.println("  " + label + ": unavailable");
            return;
        }
        int changed = 0;
        double maxAbs = 0;
        for (int r = 1; r < snaps.size(); r++) {
            float[] a = snaps.get(0);
            float[] b = snaps.get(r);
            for (int i = 0; i < a.length; i++) {
                if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                    changed++;
                    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
                }
            }
        }
        float[] a0 = snaps.get(0);
        int nonZero = 0;
        for (float v : a0) {
            if (v != 0f) {
                nonZero++;
            }
        }
        // A buffer that is never copied back from the device stays at its host-side zeros and
        // would look "identical" for the wrong reason. Report the fill so that cannot be misread.
        System.out.printf(
                "  %s len=%d nonZero=%d changed=%d maxAbs=%.6g %s%n",
                label,
                a0.length,
                nonZero,
                changed,
                maxAbs,
                nonZero == 0
                        ? "NOT READ BACK (inconclusive)"
                        : (changed == 0 ? "IDENTICAL" : "*** DIVERGES ***"));
    }

    private static float[] snapshotHalf(
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray a) {
        if (a == null) {
            return null;
        }
        float[] out = new float[a.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i).getFloat32();
        }
        return out;
    }

    private static void stats(String label, float[] a, float[] b) {
        double maxAbs = 0;
        double sum = 0;
        double maxRel = 0;
        int diff = 0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs(a[i] - b[i]);
            if (d != 0) {
                diff++;
            }
            maxAbs = Math.max(maxAbs, d);
            sum += d;
            double denom = Math.max(Math.abs(a[i]), 1e-9);
            maxRel = Math.max(maxRel, d / denom);
        }
        int argA = argmax(a);
        int argB = argmax(b);
        System.out.printf(
                "  %s: changed=%d/%d maxAbs=%.6g meanAbs=%.4g maxRel=%.4g argmax %d->%d %s topk5=%s topk10=%s%n",
                label,
                diff,
                a.length,
                maxAbs,
                sum / a.length,
                maxRel,
                argA,
                argB,
                argA == argB ? "(same)" : "(DIFFERENT)",
                topKSame(a, b, 5) ? "same" : "CHANGED",
                topKSame(a, b, 10) ? "same" : "CHANGED");
    }

    /** Top-k membership as a set: the envelope gate cares about which tokens are candidates. */
    private static boolean topKSame(float[] a, float[] b, int k) {
        return topK(a, k).equals(topK(b, k));
    }

    private static java.util.Set<Integer> topK(float[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (p, q) -> Float.compare(v[q], v[p]));
        java.util.Set<Integer> out = new java.util.HashSet<>();
        for (int i = 0; i < k; i++) {
            out.add(idx[i]);
        }
        return out;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float[] snapshot(FloatArray fa) {
        float[] out = new float[fa.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fa.get(i);
        }
        return out;
    }

    private static float[] toArray(Object t) {
        if (t instanceof FloatArray fa) {
            return snapshot(fa);
        }
        org.beehive.jitllm.tensor.standard.FloatTensor ft =
                (org.beehive.jitllm.tensor.standard.FloatTensor) t;
        float[] out = new float[ft.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ft.getFloat(i);
        }
        return out;
    }

    private Fp16DeterminismProbe() {}
}
