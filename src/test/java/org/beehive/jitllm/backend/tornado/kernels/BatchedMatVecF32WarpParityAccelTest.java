package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * {@code batchedMatVecF32Warp} against {@code batchedMatVecF32} (local size 128), raw-bit equal
 * over NaN-poisoned outputs.
 *
 * <p>The mapping is first proved on the host: a model of the control's 128-lane shared tree and a
 * model of the candidate's four partials, {@code (p0 + p2) + (p1 + p3)} and five shuffle-down steps
 * produce the same float for every input, evaluated in Java's strict FP32 arithmetic on the same
 * per-lane partials. Then on the device: distinct input rows, signed values and cancellation-heavy
 * rows (large opposed pairs), the production shape {@code n = 5120, d = 48}, widths 32, 256, 512,
 * 1024 and 2048 with partial batches (active rows below the width), and {@code n} values that leave
 * the four partials unequal tails. A candidate that drops one partial and one that combines them in
 * a different order both differ.
 */
// @formatter:on
public class BatchedMatVecF32WarpParityAccelTest {

    private static final int LOCAL = 128;

    /** The control's reduction over 128 partials, as the kernel performs it. */
    static float controlTree(float[] p) {
        float[] s = p.clone();
        for (int stride = 64; stride > 0; stride >>= 1) {
            for (int i = 0; i < stride; i++) {
                s[i] += s[i + stride];
            }
        }
        return s[0];
    }

    /** The candidate's combination of the same 128 partials. */
    static float candidateTree(float[] p) {
        float[] c = new float[32];
        for (int l = 0; l < 32; l++) {
            c[l] = (p[l] + p[l + 64]) + (p[l + 32] + p[l + 96]);
        }
        for (int off = 16; off > 0; off >>= 1) {
            for (int l = 0; l < off; l++) {
                c[l] += c[l + off];
            }
        }
        return c[0];
    }

    @Test
    public void theCombinationReproducesTheControlsTreeOnTheHost() {
        Random rng = new Random(5);
        for (int trial = 0; trial < 20000; trial++) {
            float[] p = new float[128];
            float scale = (float) Math.pow(10, rng.nextInt(9) - 4);
            for (int i = 0; i < 128; i++) {
                p[i] = (rng.nextFloat() * 2 - 1) * scale;
            }
            if ((trial & 3) == 0) {
                // Cancellation: opposed pairs across the halves the first steps combine.
                for (int i = 0; i < 64; i++) {
                    p[i + 64] = -p[i] + (rng.nextFloat() - 0.5f) * 1e-6f * scale;
                }
            }
            assertEquals(
                    "trial " + trial,
                    Float.floatToRawIntBits(controlTree(p)),
                    Float.floatToRawIntBits(candidateTree(p)));
        }
    }

    private static FloatArray inputs(int rows, int n, long seed, boolean cancelling) {
        Random rng = new Random(seed);
        FloatArray x = new FloatArray(rows * n);
        for (int i = 0; i < rows * n; i++) {
            x.set(i, (rng.nextFloat() * 2 - 1) * (cancelling ? 64.0f : 1.0f));
        }
        return x;
    }

    private static FloatArray weights(int d, int n, long seed, boolean cancelling) {
        Random rng = new Random(seed);
        FloatArray w = new FloatArray(d * n);
        for (int r = 0; r < d; r++) {
            for (int j = 0; j < n; j++) {
                float v = rng.nextFloat() * 2 - 1;
                if (cancelling && (j & 1) == 1) {
                    // Opposed to its neighbour: products cancel to a small residual.
                    v = -w.get(r * n + j - 1) + (rng.nextFloat() - 0.5f) * 1e-3f;
                }
                w.set(r * n + j, v);
            }
        }
        return w;
    }

    private static WorkerGrid controlGrid(int rows, int d) {
        WorkerGrid g = new WorkerGrid1D(rows * d * LOCAL);
        g.setLocalWork(LOCAL, 1, 1);
        return g;
    }

    private static WorkerGrid warpGrid(int rows, int d) {
        WorkerGrid g = new WorkerGrid1D(rows * d * 32);
        g.setLocalWork(LOCAL, 1, 1);
        return g;
    }

    enum Variant {
        CANDIDATE,
        DROPPED_PARTIAL,
        MISCOMBINED
    }

    /** Runs control and one candidate variant; returns the number of active outputs that differ. */
    private static int mismatches(
            String what,
            int rows,
            int active,
            int n,
            int d,
            long seed,
            boolean cancelling,
            Variant v)
            throws Exception {
        FloatArray x = inputs(rows, n, seed, cancelling);
        FloatArray w = weights(d, n, seed + 1, cancelling);
        FloatArray control = new FloatArray(rows * d);
        FloatArray candidate = new FloatArray(rows * d);
        control.init(Float.NaN);
        candidate.init(Float.NaN);
        TaskGraph graph =
                new TaskGraph("mv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, control, candidate)
                        .task(
                                "control",
                                TransformerBatchPrefillKernels::batchedMatVecF32,
                                new KernelContext(),
                                x,
                                control,
                                w,
                                n,
                                d,
                                active,
                                LOCAL)
                        .task(
                                "candidate",
                                v == Variant.CANDIDATE
                                        ? TransformerBatchPrefillKernels::batchedMatVecF32Warp
                                        : v == Variant.DROPPED_PARTIAL
                                                ? BatchedMatVecF32WarpParityAccelTest
                                                        ::warpDroppingAPartial
                                                : BatchedMatVecF32WarpParityAccelTest
                                                        ::warpMiscombined,
                                new KernelContext(),
                                x,
                                candidate,
                                w,
                                n,
                                d,
                                active)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("mv.control", controlGrid(rows, d));
        s.addWorkerGrid("mv.candidate", warpGrid(rows, d));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int count = 0;
        String first = null;
        for (int i = 0; i < rows * d; i++) {
            float c = control.get(i);
            float k = candidate.get(i);
            if (i / d < active) {
                assertTrue(what + ": control not finite at " + i, Float.isFinite(c));
                if (v == Variant.CANDIDATE) {
                    assertTrue(what + ": candidate not finite at " + i, Float.isFinite(k));
                }
            } else {
                // Inactive rows: neither kernel writes them; the poison must survive in both.
                assertTrue(what + ": control wrote an inactive row at " + i, Float.isNaN(c));
                assertTrue(what + ": candidate wrote an inactive row at " + i, Float.isNaN(k));
                continue;
            }
            if (Float.floatToRawIntBits(c) != Float.floatToRawIntBits(k)) {
                if (first == null) {
                    first = "row " + (i / d) + " out " + (i % d) + ": " + c + " vs " + k;
                }
                count++;
            }
        }
        if (v == Variant.CANDIDATE) {
            assertEquals(what + ": " + count + " outputs differ, first at " + first, 0, count);
        }
        return count;
    }

    @Test
    public void theProductionShapeAgreesBitForBitAtEveryWidth() throws Exception {
        int[][] cases = {{32, 32}, {256, 256}, {512, 500}, {1024, 1024}, {2048, 2000}, {256, 5}};
        for (int[] c : cases) {
            mismatches(
                    "random w" + c[0] + " active " + c[1],
                    c[0],
                    c[1],
                    5120,
                    48,
                    11L + c[0],
                    false,
                    Variant.CANDIDATE);
            mismatches(
                    "cancelling w" + c[0] + " active " + c[1],
                    c[0],
                    c[1],
                    5120,
                    48,
                    21L + c[0],
                    true,
                    Variant.CANDIDATE);
        }
    }

    /** n values whose four partial sequences have different lengths, and n below 128. */
    @Test
    public void unequalTailsAgreeBitForBit() throws Exception {
        for (int n : new int[] {96, 130, 160, 200, 250, 5120 + 40, 5120 + 96}) {
            mismatches("tail n=" + n, 64, 64, n, 7, 100L + n, false, Variant.CANDIDATE);
            mismatches("tail cancelling n=" + n, 64, 64, n, 7, 200L + n, true, Variant.CANDIDATE);
        }
    }

    @Test
    public void theBrokenCandidatesDiffer() throws Exception {
        assertTrue(
                "dropped partial agreed",
                mismatches("dropped", 64, 64, 5120, 48, 3L, false, Variant.DROPPED_PARTIAL) > 0);
        assertTrue(
                "miscombined agreed",
                mismatches("miscombined", 64, 64, 5120, 48, 4L, true, Variant.MISCOMBINED) > 0);
    }

    /** Control vs candidate at the production shape, widths 32..2048, alternating after warm-up. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        int n = 5120;
        int d = 48;
        FloatArray w = weights(d, n, 7L, false);
        for (int rows : new int[] {32, 256, 512, 1024, 2048}) {
            FloatArray x = inputs(rows, n, 9L + rows, false);
            FloatArray outA = new FloatArray(rows * d);
            FloatArray outB = new FloatArray(rows * d);
            TaskGraph a =
                    new TaskGraph("ctl")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w)
                            .task(
                                    "p",
                                    TransformerBatchPrefillKernels::batchedMatVecF32,
                                    new KernelContext(),
                                    x,
                                    outA,
                                    w,
                                    n,
                                    d,
                                    rows,
                                    LOCAL)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
            TaskGraph b =
                    new TaskGraph("warp")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w)
                            .task(
                                    "p",
                                    TransformerBatchPrefillKernels::batchedMatVecF32Warp,
                                    new KernelContext(),
                                    x,
                                    outB,
                                    w,
                                    n,
                                    d,
                                    rows)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("ctl.p", controlGrid(rows, d));
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("warp.p", warpGrid(rows, d));
            try (TornadoExecutionPlan planA = new TornadoExecutionPlan(a.snapshot());
                    TornadoExecutionPlan planB = new TornadoExecutionPlan(b.snapshot())) {
                planA.withGridScheduler(sa).withProfiler(ProfilerMode.SILENT);
                planB.withGridScheduler(sb).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 5; i++) {
                    planA.execute();
                    planB.execute();
                }
                int samples = 15;
                long[] tA = new long[samples];
                long[] tB = new long[samples];
                for (int i = 0; i < samples; i++) {
                    if ((i & 1) == 0) {
                        tA[i] = kernelNs(planA.execute());
                        tB[i] = kernelNs(planB.execute());
                    } else {
                        tB[i] = kernelNs(planB.execute());
                        tA[i] = kernelNs(planA.execute());
                    }
                }
                report("control rows=" + rows + " blocks=" + rows * d, tA);
                report("warp    rows=" + rows + " blocks=" + rows * d / 4, tB);
            }
        }
    }

    private static long kernelNs(TornadoExecutionResult result) {
        return result.getProfilerResult().getDeviceKernelTime();
    }

    private static void report(String label, long[] ns) {
        long[] sorted = ns.clone();
        Arrays.sort(sorted);
        StringBuilder samples = new StringBuilder();
        for (long v : ns) {
            samples.append(String.format(Locale.ROOT, "%.1f;", v / 1e3));
        }
        System.out.printf(
                Locale.ROOT,
                "[screen] %-34s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }

    // ---- negative controls ---------------------------------------------------------------------

    /** The candidate without its fourth partial. */
    public static void warpDroppingAPartial(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            FloatArray w,
            int n,
            int d,
            int activeRows) {
        int lane = context.localIdx & 31;
        int output = (context.groupIdx << 2) + (context.localIdx >> 5);
        int batchIdx = output / d;
        int rowIdx = output - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        for (int j = lane; j < n; j += 128) {
            partial0 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 32; j < n; j += 128) {
            partial1 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 64; j < n; j += 128) {
            partial2 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        float combined = (partial0 + partial2) + partial1; // NEGATIVE CONTROL: partial3 missing
        combined += context.simdShuffleDown(combined, 16);
        combined += context.simdShuffleDown(combined, 8);
        combined += context.simdShuffleDown(combined, 4);
        combined += context.simdShuffleDown(combined, 2);
        combined += context.simdShuffleDown(combined, 1);
        if (lane == 0) {
            outputBatch.set(batchIdx * d + rowIdx, combined);
        }
    }

    /** The candidate combining its partials in the wrong order. */
    public static void warpMiscombined(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            FloatArray w,
            int n,
            int d,
            int activeRows) {
        int lane = context.localIdx & 31;
        int output = (context.groupIdx << 2) + (context.localIdx >> 5);
        int batchIdx = output / d;
        int rowIdx = output - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        for (int j = lane; j < n; j += 128) {
            partial0 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 32; j < n; j += 128) {
            partial1 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 64; j < n; j += 128) {
            partial2 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 96; j < n; j += 128) {
            partial3 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        float combined =
                ((partial0 + partial1) + partial2) + partial3; // NEGATIVE CONTROL: sequential
        combined += context.simdShuffleDown(combined, 16);
        combined += context.simdShuffleDown(combined, 8);
        combined += context.simdShuffleDown(combined, 4);
        combined += context.simdShuffleDown(combined, 2);
        combined += context.simdShuffleDown(combined, 1);
        if (lane == 0) {
            outputBatch.set(batchIdx * d + rowIdx, combined);
        }
    }
}
