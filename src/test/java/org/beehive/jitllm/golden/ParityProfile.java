package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Reports, per configuration, the quantities the parity gate asserts on: elementwise violations of
 * {@code atol + rtol·|ref|}, relative L2, cosine similarity, the error percentiles, and the
 * decision-level agreement. Picking a threshold from a single worst-case number is how the previous
 * row-max tolerance ended up both too loose for small logits and unexplained.
 */
public final class ParityProfile {

    private static final double RTOL = 1e-2;
    private static final double ATOL = 1e-3;

    public static void main(String[] args) throws Exception {
        for (String key : new String[] {"parity.model", "parity.f16", "parity.q8"}) {
            String path = System.getProperty(key);
            if (path != null) {
                profile(key, Path.of(path));
            }
        }
    }

    private static void profile(String label, Path model) throws Exception {
        GoldenCapture.Result cpu = GoldenCapture.capture(model, false);
        GoldenCapture.Result gpu = GoldenCapture.capture(model, true, cpu.tokenIds);
        GoldenCapture.Result gpuFree = GoldenCapture.capture(model, true);
        System.out.printf(
                "%s (%s) free-running token ids match CPU: %s%n",
                label, model.getFileName(), cpu.tokenIds.equals(gpuFree.tokenIds));

        int rows = cpu.rows.size();
        int vocab = cpu.rows.get(0).length;
        long total = (long) rows * vocab;
        long violations = 0;
        double maxAbs = 0, maxRatio = 0, worstRelL2 = 0, minCos = 1;
        int worstRelL2Row = -1;
        double[] allAbs = new double[(int) total];
        int n = 0;
        int argmaxDisagreements = 0;
        int top5Overlap = 0, top10Overlap = 0;

        for (int r = 0; r < rows; r++) {
            float[] ref = cpu.rows.get(r);
            float[] got = gpu.rows.get(r);
            double sqDiff = 0, sqRef = 0, sqGot = 0, dot = 0;
            for (int i = 0; i < ref.length; i++) {
                double d = Math.abs((double) ref[i] - got[i]);
                double tol = ATOL + RTOL * Math.abs((double) ref[i]);
                if (d > tol) {
                    violations++;
                }
                maxRatio = Math.max(maxRatio, d / tol);
                maxAbs = Math.max(maxAbs, d);
                allAbs[n++] = d;
                sqDiff += d * d;
                sqRef += (double) ref[i] * ref[i];
                sqGot += (double) got[i] * got[i];
                dot += (double) ref[i] * got[i];
            }
            double relL2 = Math.sqrt(sqDiff / sqRef);
            if (Boolean.getBoolean("parity.perRow")) {
                System.out.printf("    row %2d relL2=%.5g%n", r, relL2);
            }
            if (relL2 > worstRelL2) {
                worstRelL2 = relL2;
                worstRelL2Row = r;
            }
            minCos = Math.min(minCos, dot / (Math.sqrt(sqRef) * Math.sqrt(sqGot)));
            if (Envelope.argmax(ref) != Envelope.argmax(got)) {
                argmaxDisagreements++;
            }
            top5Overlap += overlap(ref, got, 5);
            top10Overlap += overlap(ref, got, 10);
        }

        // The scale of the reference itself, so an absolute bound can be read as a fraction of
        // it. Families differ by orders of magnitude here -- Granite multiplies its logits by
        // its logit_scale -- and an absolute atol calibrated on one does not transfer.
        double sqAll = 0;
        long counted = 0;
        for (float[] ref : cpu.rows) {
            for (float v : ref) {
                sqAll += (double) v * v;
                counted++;
            }
        }
        double refRms = Math.sqrt(sqAll / counted);
        System.out.printf("  refRms=%.6g  maxAbs/refRms=%.3g%n", refRms, maxAbs / refRms);

        Arrays.sort(allAbs);
        System.out.printf("%s rows=%d vocab=%d%n", label, rows, vocab);
        System.out.printf(
                "  elementwise (atol=%.0e rtol=%.0e): violations=%d/%d (%.4g%%) maxRatio=%.4g%n",
                ATOL, RTOL, violations, total, 100.0 * violations / total, maxRatio);
        System.out.printf(
                "  absErr: p50=%.3g p99=%.3g p99.99=%.3g max=%.3g%n",
                allAbs[(int) (total * 0.50)],
                allAbs[(int) (total * 0.99)],
                allAbs[(int) (total * 0.9999)],
                maxAbs);
        System.out.printf(
                "  worst relL2=%.4g (row %d)  minCosine=%.8f%n", worstRelL2, worstRelL2Row, minCos);
        System.out.printf(
                "  argmax disagreements=%d/%d  top5 overlap=%.3f/5  top10 overlap=%.3f/10%n",
                argmaxDisagreements,
                rows,
                top5Overlap / (double) rows,
                top10Overlap / (double) rows);

        // Sweep the absolute floor: the relative term only bites on the few large logits, so the
        // floor is what decides whether the elementwise gate is meaningful or noise-dominated.
        for (double atol : new double[] {1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2}) {
            long bad = 0;
            for (int r = 0; r < rows; r++) {
                float[] ref = cpu.rows.get(r);
                float[] got = gpu.rows.get(r);
                for (int i = 0; i < ref.length; i++) {
                    if (Math.abs((double) ref[i] - got[i])
                            > atol + RTOL * Math.abs((double) ref[i])) {
                        bad++;
                    }
                }
            }
            System.out.printf(
                    "    atol=%.0e rtol=%.0e -> violations=%d (%.4g%%)%n",
                    atol, RTOL, bad, 100.0 * bad / total);
        }
    }

    private static int overlap(float[] a, float[] b, int k) {
        int[] ta = topK(a, k);
        int[] tb = topK(b, k);
        int c = 0;
        for (int x : ta) {
            for (int y : tb) {
                if (x == y) {
                    c++;
                    break;
                }
            }
        }
        return c;
    }

    private static int[] topK(float[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Float.compare(v[q], v[p]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    private ParityProfile() {}
}
