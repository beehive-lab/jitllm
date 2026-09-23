package org.beehive.jitllm.golden;

import java.nio.file.Path;

/** Diagnostic: is the CPU/GPU gap a GPU-path property or a weight-precision property? */
public final class ParityCross {
    public static void main(String[] args) throws Exception {
        Path f16 = Path.of(System.getProperty("parity.f16"));
        Path q80 = Path.of(System.getProperty("parity.q8"));
        GoldenCapture.Result cpuF = GoldenCapture.capture(f16, false);
        GoldenCapture.Result gpuF = GoldenCapture.capture(f16, true, cpuF.tokenIds);
        GoldenCapture.Result cpuQ = GoldenCapture.capture(q80, false);
        GoldenCapture.Result gpuQ = GoldenCapture.capture(q80, true, cpuF.tokenIds);
        // Free-running (not teacher-forced): do the two paths now decode the same text?
        GoldenCapture.Result gpuFree = GoldenCapture.capture(f16, true);
        GoldenCapture.Result gpuQFree = GoldenCapture.capture(q80, true);
        System.out.printf(
                "free-running token ids: F16 cpu==gpu? %s ; Q8_0 cpu==gpu? %s%n",
                cpuF.tokenIds.equals(gpuFree.tokenIds), cpuQ.tokenIds.equals(gpuQFree.tokenIds));

        report("CPU-F16 vs GPU-F16", cpuF, gpuF);
        report("CPU-Q8_0 vs GPU-Q8_0", cpuQ, gpuQ);
        report("CPU-F16 vs CPU-Q8_0", cpuF, cpuQ);
        report("GPU-F16 vs GPU-Q8_0", gpuF, gpuQ);
    }

    private static void report(String label, GoldenCapture.Result a, GoldenCapture.Result b) {
        int n = Math.min(a.rows.size(), b.rows.size());
        double worst = 0, meanOfMean = 0;
        for (int r = 0; r < n; r++) {
            float[] x = a.rows.get(r), y = b.rows.get(r);
            double sum = 0;
            for (int i = 0; i < x.length; i++) {
                double d = Math.abs((double) x[i] - y[i]);
                worst = Math.max(worst, d);
                sum += d;
            }
            meanOfMean += sum / x.length;
        }
        System.out.printf(
                "%-24s rows=%d worstAbs=%.4f meanAbs=%.5f%n", label, n, worst, meanOfMean / n);
    }

    private ParityCross() {}
}
