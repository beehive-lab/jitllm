package org.beehive.jitllm.backend.tornado.kernels;

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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The shared-state delta-rule scan against the reference scan, on the device, over identical inputs
 * and identical non-zero initial states: every output of every active row and every element of the
 * final persistent state, raw-bit equal, with outputs NaN-poisoned.
 *
 * <p>The fixture is this model's recurrent geometry — 48 value heads over 16 key heads, a 128-wide
 * state — with the state at a non-zero offset (a second layer's slice), queries and keys
 * L2-normalized per head as the model feeds them (the query further scaled by {@code 1/sqrt(128)}),
 * decay in {@code (0, 1]} including exactly one for some heads, beta in {@code (0, 1)}. Cases: a
 * full chunk; a partial chunk; two chunks in sequence on the same state, so the second starts from
 * what the first left; and a long chunk.
 */
// @formatter:on
public class Qwen35DeltaRuleSharedParityAccelTest {

    private static final int VALUE_HEADS = 48;
    private static final int KEY_HEADS = 16;
    private static final int STATE_DIM = 128;
    private static final int LAYERS = 2;
    private static final int LAYER = 1;
    private static final int KEY_DIM = KEY_HEADS * STATE_DIM;
    private static final int VALUE_DIM = VALUE_HEADS * STATE_DIM;
    private static final int STATE_PER_LAYER = VALUE_HEADS * STATE_DIM * STATE_DIM;
    private static final int STATE_OFFSET = LAYER * STATE_PER_LAYER;

    private static FloatArray normalizedHeads(int rows, int heads, long seed, float scale) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(rows * heads * STATE_DIM);
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < heads; h++) {
                float[] x = new float[STATE_DIM];
                double norm = 0;
                for (int i = 0; i < STATE_DIM; i++) {
                    x[i] = (float) rng.nextGaussian();
                    norm += x[i] * x[i];
                }
                float inv = (float) (1.0 / Math.sqrt(norm + 1e-6));
                for (int i = 0; i < STATE_DIM; i++) {
                    a.set((r * heads + h) * STATE_DIM + i, x[i] * inv * scale);
                }
            }
        }
        return a;
    }

    private static FloatArray uniform(int count, long seed, float lo, float hi) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(count);
        for (int i = 0; i < count; i++) {
            a.set(i, lo + rng.nextFloat() * (hi - lo));
        }
        return a;
    }

    /** Decay per (row, head) in (0, 1], with heads 0 and 17 held at exactly one. */
    private static FloatArray decays(int rows, long seed) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(rows * VALUE_HEADS);
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < VALUE_HEADS; h++) {
                float g = 0.9f + 0.1f * rng.nextFloat();
                if (h == 0 || h == 17) {
                    g = 1.0f;
                }
                a.set(r * VALUE_HEADS + h, g);
            }
        }
        return a;
    }

    private static FloatArray initialState(long seed) {
        Random rng = new Random(seed);
        FloatArray s = new FloatArray(LAYERS * STATE_PER_LAYER);
        for (int i = 0; i < s.getSize(); i++) {
            s.set(i, (float) (rng.nextGaussian() * 0.05));
        }
        return s;
    }

    private static IntArray batchInfo(int activeRows) {
        IntArray info = new IntArray(3);
        info.set(0, 0);
        info.set(1, activeRows);
        info.set(2, 0);
        return info;
    }

    private static WorkerGrid referenceGrid() {
        WorkerGrid g = new WorkerGrid1D(VALUE_HEADS * STATE_DIM);
        g.setLocalWork(128, 1, 1);
        return g;
    }

    private static WorkerGrid sharedGrid() {
        WorkerGrid g =
                new WorkerGrid1D(
                        VALUE_HEADS
                                * (STATE_DIM / Qwen35BatchKernels.DELTA_SHARED_COLUMNS)
                                * Qwen35BatchKernels.DELTA_SHARED_COLUMNS);
        g.setLocalWork(Qwen35BatchKernels.DELTA_SHARED_COLUMNS, 1, 1);
        return g;
    }

    /** Runs both scans over the same inputs and state on a chain of chunks; compares after each. */
    private static void assertChain(String what, int rows, int[] activeRowsPerChunk, long seed)
            throws Exception {
        FloatArray stateA = initialState(seed);
        FloatArray stateB = initialState(seed);
        for (int chunk = 0; chunk < activeRowsPerChunk.length; chunk++) {
            int active = activeRowsPerChunk[chunk];
            long s = seed * 100 + chunk;
            FloatArray q =
                    normalizedHeads(rows, KEY_HEADS, s + 1, (float) (1.0 / Math.sqrt(STATE_DIM)));
            FloatArray k = normalizedHeads(rows, KEY_HEADS, s + 2, 1.0f);
            FloatArray v = uniform(rows * VALUE_DIM, s + 3, -1.0f, 1.0f);
            FloatArray decay = decays(rows, s + 4);
            FloatArray beta = uniform(rows * VALUE_HEADS, s + 5, 0.05f, 0.95f);
            IntArray info = batchInfo(active);
            FloatArray outA = new FloatArray(rows * VALUE_DIM);
            FloatArray outB = new FloatArray(rows * VALUE_DIM);
            outA.init(Float.NaN);
            outB.init(Float.NaN);

            TaskGraph graph =
                    new TaskGraph("delta")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    info,
                                    stateA,
                                    stateB,
                                    outA,
                                    outB)
                            .task(
                                    "reference",
                                    Qwen35BatchKernels::deltaRuleScan,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateA,
                                    outA,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .task(
                                    "shared",
                                    Qwen35BatchKernels::deltaRuleScanShared,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateB,
                                    outB,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION, outA, outB, stateA, stateB);
            GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("delta.reference", referenceGrid());
            scheduler.addWorkerGrid("delta.shared", sharedGrid());
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            String tag = what + " chunk " + chunk + " (" + active + " active)";
            int outDiff = 0;
            String first = null;
            for (int r = 0; r < active; r++) {
                for (int e = 0; e < VALUE_DIM; e++) {
                    float a = outA.get(r * VALUE_DIM + e);
                    float b = outB.get(r * VALUE_DIM + e);
                    assertTrue(
                            tag + ": reference output not finite at row " + r + " e " + e,
                            Float.isFinite(a));
                    assertTrue(
                            tag + ": shared output not finite at row " + r + " e " + e,
                            Float.isFinite(b));
                    if (Float.floatToRawIntBits(a) != Float.floatToRawIntBits(b)) {
                        if (first == null) {
                            first =
                                    "row "
                                            + r
                                            + " head "
                                            + (e / STATE_DIM)
                                            + " col "
                                            + (e % STATE_DIM)
                                            + ": "
                                            + a
                                            + " vs "
                                            + b;
                        }
                        outDiff++;
                    }
                }
            }
            assertEquals(tag + ": " + outDiff + " outputs differ, first at " + first, 0, outDiff);
            int stateDiff = 0;
            String firstState = null;
            for (int i = 0; i < stateA.getSize(); i++) {
                float a = stateA.get(i);
                float b = stateB.get(i);
                assertTrue(tag + ": reference state not finite at " + i, Float.isFinite(a));
                if (Float.floatToRawIntBits(a) != Float.floatToRawIntBits(b)) {
                    if (firstState == null) {
                        int inLayer = i - STATE_OFFSET;
                        firstState =
                                "element "
                                        + i
                                        + (inLayer >= 0
                                                ? " (head "
                                                        + inLayer / (STATE_DIM * STATE_DIM)
                                                        + ")"
                                                : " (other layer)")
                                        + ": "
                                        + a
                                        + " vs "
                                        + b;
                    }
                    stateDiff++;
                }
            }
            assertEquals(
                    tag + ": " + stateDiff + " state elements differ, first at " + firstState,
                    0,
                    stateDiff);
        }
    }

    /** One full 32-row chunk from a non-zero state. */
    @Test
    public void aFullChunkAgreesBitForBit() throws Exception {
        assertChain("full", 32, new int[] {32}, 1L);
    }

    /** A partial chunk: five active rows of thirty-two. */
    @Test
    public void aPartialChunkAgreesBitForBit() throws Exception {
        assertChain("partial", 32, new int[] {5}, 2L);
    }

    /** Three chunks in sequence, the state carried from one to the next, the last partial. */
    @Test
    public void chainedChunksAgreeBitForBit() throws Exception {
        assertChain("chain", 32, new int[] {32, 32, 7}, 3L);
    }

    /** A 256-row chunk. */
    @Test
    public void aLongChunkAgreesBitForBit() throws Exception {
        assertChain("long", 256, new int[] {256}, 4L);
    }

    /**
     * Kernel time of both scans at chunk lengths 32, 256, 512 and 1024 from the TornadoVM profiler,
     * alternating after a warm-up; every sample includes the state load and write-back. Opt-in;
     * prints distributions only.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        for (int rows : new int[] {32, 256, 512, 1024}) {
            long s = 900L + rows;
            FloatArray q =
                    normalizedHeads(rows, KEY_HEADS, s + 1, (float) (1.0 / Math.sqrt(STATE_DIM)));
            FloatArray k = normalizedHeads(rows, KEY_HEADS, s + 2, 1.0f);
            FloatArray v = uniform(rows * VALUE_DIM, s + 3, -1.0f, 1.0f);
            FloatArray decay = decays(rows, s + 4);
            FloatArray beta = uniform(rows * VALUE_HEADS, s + 5, 0.05f, 0.95f);
            IntArray info = batchInfo(rows);
            FloatArray stateA = initialState(s);
            FloatArray stateB = initialState(s);
            FloatArray outA = new FloatArray(rows * VALUE_DIM);
            FloatArray outB = new FloatArray(rows * VALUE_DIM);
            TaskGraph a =
                    new TaskGraph("ref")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    info,
                                    stateA)
                            .task(
                                    "p",
                                    Qwen35BatchKernels::deltaRuleScan,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateA,
                                    outA,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
            TaskGraph b =
                    new TaskGraph("shd")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    info,
                                    stateB)
                            .task(
                                    "p",
                                    Qwen35BatchKernels::deltaRuleScanShared,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateB,
                                    outB,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("ref.p", referenceGrid());
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("shd.p", sharedGrid());
            try (TornadoExecutionPlan planA = new TornadoExecutionPlan(a.snapshot());
                    TornadoExecutionPlan planB = new TornadoExecutionPlan(b.snapshot())) {
                planA.withGridScheduler(sa).withProfiler(ProfilerMode.SILENT);
                planB.withGridScheduler(sb).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 3; i++) {
                    planA.execute();
                    planB.execute();
                }
                int samples = 11;
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
                report("reference rows=" + rows, tA);
                report("shared    rows=" + rows, tB);
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
                "[screen] %-22s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
