package org.beehive.jitllm.backend.tornado.kernels;

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
 * The warp-per-column delta-rule scan — a reassociating kernel — against an FP64 recurrence over
 * exactly the same FP32 inputs, beside the shared-state scan it would replace.
 *
 * <p>Both device scans run the same chunks from the same non-zero initial state; after every chunk
 * the outputs of that chunk and the whole persistent state are compared with the FP64 recurrence
 * (control-vs-FP64 and candidate-vs-FP64) and with each other (candidate-vs-control), so a
 * candidate defect shows as error beyond the control's, not as a difference from a control that has
 * its own rounding. The sequence is chained to 2,048 tokens so growth across checkpoints is
 * visible, not only the final error.
 *
 * <p>The fixture is this model's recurrent geometry — 48 value heads over 16 key heads, a 128-wide
 * state at a non-zero offset — with queries and keys L2-normalized per head (the query scaled by
 * {@code 1/sqrt(128)}), decay in {@code (0, 1]} with two heads held at exactly one, beta in {@code
 * (0, 1)}.
 *
 * <p>The bound asserted on the candidate is the control's own worst error against FP64 times a
 * small factor, fixed here before the candidate was measured: a reassociated sum of 128 products
 * differs from a sequential one by a few units in the last place, not by orders of magnitude.
 */
// @formatter:on
public class Qwen35DeltaRuleWarpNumericsAccelTest {

    private static final int VALUE_HEADS = 48;
    private static final int KEY_HEADS = 16;
    private static final int STATE_DIM = 128;
    private static final int LAYERS = 2;
    private static final int LAYER = 1;
    private static final int KEY_DIM = KEY_HEADS * STATE_DIM;
    private static final int VALUE_DIM = VALUE_HEADS * STATE_DIM;
    private static final int STATE_PER_LAYER = VALUE_HEADS * STATE_DIM * STATE_DIM;
    private static final int STATE_OFFSET = LAYER * STATE_PER_LAYER;

    /** The candidate may not exceed this multiple of the control's own error against FP64. */
    private static final double CANDIDATE_OVER_CONTROL = 4.0;

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

    private static WorkerGrid sharedGrid() {
        WorkerGrid g = new WorkerGrid1D(VALUE_HEADS * STATE_DIM);
        g.setLocalWork(Qwen35BatchKernels.DELTA_SHARED_COLUMNS, 1, 1);
        return g;
    }

    private static WorkerGrid warpGrid() {
        WorkerGrid g = new WorkerGrid1D(VALUE_HEADS * STATE_DIM * 32);
        g.setLocalWork(Qwen35BatchKernels.DELTA_WARP_LOCAL, 1, 1);
        return g;
    }

    /** The recurrence in FP64 over the FP32 inputs; state in double, updated in place. */
    private static void referenceChunk(
            double[] state,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            int active,
            double[] out) {
        for (int row = 0; row < active; row++) {
            for (int head = 0; head < VALUE_HEADS; head++) {
                double g = decay.get(row * VALUE_HEADS + head);
                double b = beta.get(row * VALUE_HEADS + head);
                int keyRow = row * KEY_DIM + (head % KEY_HEADS) * STATE_DIM;
                int valueRow = row * VALUE_DIM + head * STATE_DIM;
                int base = STATE_OFFSET + head * STATE_DIM * STATE_DIM;
                for (int col = 0; col < STATE_DIM; col++) {
                    double prediction = 0;
                    for (int i = 0; i < STATE_DIM; i++) {
                        double decayed = state[base + i * STATE_DIM + col] * g;
                        state[base + i * STATE_DIM + col] = decayed;
                        prediction += decayed * k.get(keyRow + i);
                    }
                    double correction = (v.get(valueRow + col) - prediction) * b;
                    double readout = 0;
                    for (int i = 0; i < STATE_DIM; i++) {
                        double updated =
                                state[base + i * STATE_DIM + col] + k.get(keyRow + i) * correction;
                        state[base + i * STATE_DIM + col] = updated;
                        readout += updated * q.get(keyRow + i);
                    }
                    out[valueRow + col] = readout;
                }
            }
        }
    }

    private record Errors(double maxAbs, double relL2, double scale) {
        @Override
        public String toString() {
            return String.format(
                    Locale.ROOT, "maxAbs %.3e relL2 %.3e (scale %.3e)", maxAbs, relL2, scale);
        }
    }

    private static Errors errors(FloatArray got, double[] ref, int from, int to) {
        double maxAbs = 0;
        double num = 0;
        double den = 0;
        double scale = 0;
        for (int i = from; i < to; i++) {
            double d = got.get(i) - ref[i];
            maxAbs = Math.max(maxAbs, Math.abs(d));
            num += d * d;
            den += ref[i] * ref[i];
            scale = Math.max(scale, Math.abs(ref[i]));
        }
        return new Errors(maxAbs, Math.sqrt(num / Math.max(den, 1e-300)), scale);
    }

    private static Errors errorsBetween(FloatArray a, FloatArray b, int from, int to) {
        double maxAbs = 0;
        double num = 0;
        double den = 0;
        double scale = 0;
        for (int i = from; i < to; i++) {
            double d = a.get(i) - b.get(i);
            maxAbs = Math.max(maxAbs, Math.abs(d));
            num += d * d;
            den += (double) b.get(i) * b.get(i);
            scale = Math.max(scale, Math.abs(b.get(i)));
        }
        return new Errors(maxAbs, Math.sqrt(num / Math.max(den, 1e-300)), scale);
    }

    /**
     * Runs the chain on both scans and FP64; asserts the candidate within the control's error band.
     */
    private static void assertChain(String what, int rows, int[] activeRowsPerChunk, long seed)
            throws Exception {
        FloatArray stateControl = initialState(seed);
        FloatArray stateCandidate = initialState(seed);
        double[] stateRef = new double[LAYERS * STATE_PER_LAYER];
        for (int i = 0; i < stateRef.length; i++) {
            stateRef[i] = stateControl.get(i);
        }
        int position = 0;
        double worstControlOut = 0;
        double worstCandidateOut = 0;
        double worstControlState = 0;
        double worstCandidateState = 0;
        for (int chunk = 0; chunk < activeRowsPerChunk.length; chunk++) {
            int active = activeRowsPerChunk[chunk];
            long s = seed * 1000 + chunk;
            FloatArray q =
                    normalizedHeads(rows, KEY_HEADS, s + 1, (float) (1.0 / Math.sqrt(STATE_DIM)));
            FloatArray k = normalizedHeads(rows, KEY_HEADS, s + 2, 1.0f);
            FloatArray v = uniform(rows * VALUE_DIM, s + 3, -1.0f, 1.0f);
            FloatArray decay = decays(rows, s + 4);
            FloatArray beta = uniform(rows * VALUE_HEADS, s + 5, 0.05f, 0.95f);
            IntArray info = batchInfo(active);
            FloatArray outControl = new FloatArray(rows * VALUE_DIM);
            FloatArray outCandidate = new FloatArray(rows * VALUE_DIM);
            outControl.init(Float.NaN);
            outCandidate.init(Float.NaN);
            double[] outRef = new double[rows * VALUE_DIM];

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
                                    stateControl,
                                    stateCandidate,
                                    outControl,
                                    outCandidate)
                            .task(
                                    "control",
                                    Qwen35BatchKernels::deltaRuleScanShared,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateControl,
                                    outControl,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .task(
                                    "warp",
                                    Qwen35BatchKernels::deltaRuleScanWarp,
                                    new KernelContext(),
                                    q,
                                    k,
                                    v,
                                    decay,
                                    beta,
                                    stateCandidate,
                                    outCandidate,
                                    VALUE_HEADS,
                                    KEY_HEADS,
                                    STATE_DIM,
                                    STATE_OFFSET,
                                    info)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION,
                                    outControl,
                                    outCandidate,
                                    stateControl,
                                    stateCandidate);
            GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("delta.control", sharedGrid());
            scheduler.addWorkerGrid("delta.warp", warpGrid());
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }
            referenceChunk(stateRef, q, k, v, decay, beta, active, outRef);
            position += active;

            for (int i = 0; i < active * VALUE_DIM; i++) {
                assertTrue(what + ": control output not finite", Float.isFinite(outControl.get(i)));
                assertTrue(
                        what + ": candidate output not finite",
                        Float.isFinite(outCandidate.get(i)));
            }
            Errors cOut = errors(outControl, outRef, 0, active * VALUE_DIM);
            Errors xOut = errors(outCandidate, outRef, 0, active * VALUE_DIM);
            Errors dOut = errorsBetween(outCandidate, outControl, 0, active * VALUE_DIM);
            Errors cState =
                    errors(stateControl, stateRef, STATE_OFFSET, STATE_OFFSET + STATE_PER_LAYER);
            Errors xState =
                    errors(stateCandidate, stateRef, STATE_OFFSET, STATE_OFFSET + STATE_PER_LAYER);
            Errors dState =
                    errorsBetween(
                            stateCandidate,
                            stateControl,
                            STATE_OFFSET,
                            STATE_OFFSET + STATE_PER_LAYER);
            System.out.printf(
                    Locale.ROOT,
                    "[DELTA64] %s chunk %d (tokens %d..%d)%n   out:   control-vs-fp64 %s | candidate-vs-fp64 %s | candidate-vs-control %s%n   state: control-vs-fp64 %s | candidate-vs-fp64 %s | candidate-vs-control %s%n",
                    what,
                    chunk,
                    position - active,
                    position - 1,
                    cOut,
                    xOut,
                    dOut,
                    cState,
                    xState,
                    dState);
            worstControlOut = Math.max(worstControlOut, cOut.relL2());
            worstCandidateOut = Math.max(worstCandidateOut, xOut.relL2());
            worstControlState = Math.max(worstControlState, cState.relL2());
            worstCandidateState = Math.max(worstCandidateState, xState.relL2());
            // The other layer's slice must be untouched by both.
            for (int i = 0; i < STATE_OFFSET; i++) {
                assertTrue(
                        what + ": the other layer's state moved",
                        Float.floatToRawIntBits(stateCandidate.get(i))
                                == Float.floatToRawIntBits(stateControl.get(i)));
            }
            assertTrue(
                    what
                            + " chunk "
                            + chunk
                            + ": candidate output error "
                            + xOut
                            + " exceeds "
                            + CANDIDATE_OVER_CONTROL
                            + " x control "
                            + cOut,
                    xOut.relL2() <= CANDIDATE_OVER_CONTROL * Math.max(cOut.relL2(), 1e-9)
                            && xOut.maxAbs()
                                    <= CANDIDATE_OVER_CONTROL * Math.max(cOut.maxAbs(), 1e-9));
            assertTrue(
                    what
                            + " chunk "
                            + chunk
                            + ": candidate state error "
                            + xState
                            + " exceeds "
                            + CANDIDATE_OVER_CONTROL
                            + " x control "
                            + cState,
                    xState.relL2() <= CANDIDATE_OVER_CONTROL * Math.max(cState.relL2(), 1e-9)
                            && xState.maxAbs()
                                    <= CANDIDATE_OVER_CONTROL * Math.max(cState.maxAbs(), 1e-9));
        }
        System.out.printf(
                Locale.ROOT,
                "[DELTA64] %s worst relL2 over checkpoints: out control %.3e candidate %.3e; state control %.3e candidate %.3e%n",
                what,
                worstControlOut,
                worstCandidateOut,
                worstControlState,
                worstCandidateState);
    }

    /** One full chunk from a non-zero state. */
    @Test
    public void aFullChunk() throws Exception {
        assertChain("full", 32, new int[] {32}, 1L);
    }

    /** A partial chunk, then a full one, then a partial: chained with the state carried. */
    @Test
    public void chainedPartialChunks() throws Exception {
        assertChain("chain", 32, new int[] {5, 32, 7}, 2L);
    }

    /** Eight 256-token chunks: 2,048 tokens, a checkpoint after each. */
    @Test
    public void twoThousandTokensInChunksOf256() throws Exception {
        assertChain("long", 256, new int[] {256, 256, 256, 256, 256, 256, 256, 256}, 3L);
    }

    /** Two 1,024-token chunks: the production width. */
    @Test
    public void twoChunksOf1024() throws Exception {
        assertChain("wide", 1024, new int[] {1024, 1024}, 4L);
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
                    new TaskGraph("ctl")
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
                                    Qwen35BatchKernels::deltaRuleScanShared,
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
                    new TaskGraph("wrp")
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
                                    Qwen35BatchKernels::deltaRuleScanWarp,
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
            sa.addWorkerGrid("ctl.p", sharedGrid());
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("wrp.p", warpGrid());
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
                report("shared rows=" + rows, tA);
                report("warp   rows=" + rows, tB);
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
                "[screen] %-18s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
