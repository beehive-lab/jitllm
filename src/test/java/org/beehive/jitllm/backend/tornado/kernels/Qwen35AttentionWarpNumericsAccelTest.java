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
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The warp-first-pass batched attention against an independent FP64 reference, alongside the wide
 * kernel it replaces the first pass of, on the device over identical inputs.
 *
 * <p>The candidate reassociates each 256-product dot product (32 partials of eight, then a shuffle
 * tree) so its scores are not bit-equal to the control's; this test measures both kernels' errors
 * against the same FP64 causal scores and final attention and requires the candidate's to be no
 * worse than a small multiple of the control's, on every case. Cases: production geometry with
 * shuffled physical pages, a second layer and a non-zero slot; ranges from position zero, mid-page
 * and partial-chunk starts, the 16- and 128-position tile boundaries, the last positions of a 2048
 * context; tied keys (equal scores at many positions), cancelling keys (large opposed components
 * summing near zero), and query scales that put the scaled logits across small, moderate and wide
 * ranges. Score and output buffers are NaN-poisoned and every causal score and output is checked
 * written and finite. Two broken candidates — the shuffle reduction removed, and the keys read from
 * a neighbouring lane's dimensions — are shown to fail the same gate.
 */
// @formatter:on
public class Qwen35AttentionWarpNumericsAccelTest {

    private static final int HEADS = 24;
    private static final int HEAD_SIZE = 256;
    private static final int KV_HEADS = 4;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int BLOCK_SIZE = 16;
    private static final int CAPACITY = 2048;
    private static final int BLOCKS_PER_SLOT = CAPACITY / BLOCK_SIZE;
    private static final int SLOTS = 2;
    private static final int SLOT = 1;
    private static final int KV_LAYERS = 2;
    private static final int LAYER = 1;
    private static final int LOCAL = 128;
    private static final int BLOCK_CFG = BLOCK_SIZE | (BLOCKS_PER_SLOT << 16);
    private static final int BLOCK_STRIDE = KV_LAYERS * BLOCK_SIZE * KV_DIM;

    /** The candidate's error may not exceed this multiple of the control's, per metric and case. */
    private static final double RATIO_BOUND = 4.0;

    /** Below this absolute error both kernels are at the reference and the ratio is meaningless. */
    private static final double SCORE_FLOOR = 1e-5;

    private static final double OUTPUT_FLOOR = 1e-6;

    enum Keys {
        RANDOM,
        TIED,
        /** Opposed pairs at dimensions (d, d + 1): adjacent in the control's running sum. */
        CANCELLING_ADJACENT,
        /** Opposed pairs at dimensions (d, d + 32): on the same lane of the candidate. */
        CANCELLING_STRIDE32,
        /**
         * A few dimensions two orders of magnitude above the rest, as in models with massive
         * activations.
         */
        HEAVY_TAILED
    }

    /** One key/value store with a shuffled page map. */
    private static final class Store {
        final HalfFloatArray keys;
        final HalfFloatArray values;
        final IntArray blockTable;
        final int[] logicalToPhysical;

        Store(long seed, Keys mode) {
            Random rng = new Random(seed);
            int physicalBlocks = SLOTS * BLOCKS_PER_SLOT + 3;
            int elements = physicalBlocks * BLOCK_STRIDE;
            keys = new HalfFloatArray(elements);
            values = new HalfFloatArray(elements);
            for (int i = 0; i < elements; i++) {
                keys.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
                values.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
            }
            Integer[] pages = new Integer[physicalBlocks];
            for (int i = 0; i < physicalBlocks; i++) {
                pages[i] = i;
            }
            java.util.Collections.shuffle(Arrays.asList(pages), rng);
            blockTable = new IntArray(SLOTS * BLOCKS_PER_SLOT);
            logicalToPhysical = new int[SLOTS * BLOCKS_PER_SLOT];
            for (int i = 0; i < SLOTS * BLOCKS_PER_SLOT; i++) {
                blockTable.set(i, pages[i]);
                logicalToPhysical[i] = pages[i];
            }
            if (mode == Keys.TIED) {
                // Every fourth position of the slot, every layer and kv head, carries the same
                // key, so scores tie at many positions and the maximum is attained repeatedly.
                for (int p = 0; p < CAPACITY; p += 4) {
                    for (int kvHead = 0; kvHead < KV_HEADS; kvHead++) {
                        int from = keyBase(0, kvHead);
                        int to = keyBase(p, kvHead);
                        for (int d = 0; d < HEAD_SIZE; d++) {
                            keys.set(to + d, keys.get(from + d));
                        }
                    }
                }
            } else if (mode == Keys.HEAVY_TAILED) {
                for (int p = 0; p < CAPACITY; p++) {
                    for (int kvHead = 0; kvHead < KV_HEADS; kvHead++) {
                        int base = keyBase(p, kvHead);
                        for (int d : HEAVY_DIMS) {
                            keys.set(
                                    base + d,
                                    new HalfFloat(keys.get(base + d).getFloat32() * 100.0f));
                        }
                    }
                }
            } else if (mode == Keys.CANCELLING_ADJACENT || mode == Keys.CANCELLING_STRIDE32) {
                // Keys whose dimensions come in opposed pairs of large magnitude, so the dot
                // product is a small residual of large terms. Which summation order cancels a
                // pair before its rounding error accumulates depends on where the pair sits:
                // adjacent pairs cancel inside the control's running sum, pairs 32 apart inside
                // one candidate lane's eight products.
                int stride = mode == Keys.CANCELLING_ADJACENT ? 1 : 32;
                for (int p = 0; p < CAPACITY; p++) {
                    for (int kvHead = 0; kvHead < KV_HEADS; kvHead++) {
                        int base = keyBase(p, kvHead);
                        for (int d = 0; d < HEAD_SIZE; d++) {
                            if (pairFirst(d, stride)) {
                                float big = (rng.nextFloat() * 2.0f - 1.0f) * 64.0f;
                                float tiny = (rng.nextFloat() * 2.0f - 1.0f) * 0.01f;
                                keys.set(base + d, new HalfFloat(big));
                                keys.set(base + d + stride, new HalfFloat(-big + tiny));
                            }
                        }
                    }
                }
            }
        }

        /** The dimensions {@link Keys#HEAVY_TAILED} inflates, on keys and queries alike. */
        static final int[] HEAVY_DIMS = {3, 77, 130, 201};

        /** Whether dimension {@code d} is the first of an opposed pair {@code (d, d + stride)}. */
        static boolean pairFirst(int d, int stride) {
            return stride == 1 ? (d & 1) == 0 : (d & 32) == 0;
        }

        /** Element base of position {@code p}'s key/value for {@code kvHead}, layer LAYER. */
        int keyBase(int p, int kvHead) {
            int logical = SLOT * BLOCKS_PER_SLOT + p / BLOCK_SIZE;
            int physical = logicalToPhysical[logical];
            return physical * BLOCK_STRIDE
                    + LAYER * BLOCK_SIZE * KV_DIM
                    + (p % BLOCK_SIZE) * KV_DIM
                    + kvHead * HEAD_SIZE;
        }
    }

    private static FloatArray queries(int rows, long seed, float scale, int pairStride) {
        return queries(rows, seed, scale, pairStride, false);
    }

    private static FloatArray queries(
            int rows, long seed, float scale, int pairStride, boolean heavyTailed) {
        Random rng = new Random(seed);
        FloatArray q = new FloatArray(rows * HEADS * HEAD_SIZE);
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, (rng.nextFloat() * 2.0f - 1.0f) * scale);
        }
        if (heavyTailed) {
            for (int i = 0; i < q.getSize(); i++) {
                for (int d : Store.HEAVY_DIMS) {
                    if (i % HEAD_SIZE == d) {
                        q.set(i, q.get(i) * 100.0f);
                    }
                }
            }
        }
        if (pairStride > 0) {
            // Equal query weights on each opposed key pair, so the pair's products cancel.
            for (int i = 0; i < q.getSize(); i++) {
                if (Store.pairFirst(i % HEAD_SIZE, pairStride)) {
                    q.set(i + pairStride, q.get(i));
                }
            }
        }
        return q;
    }

    private static IntArray batchInfo(int startPos, int activeRows) {
        IntArray info = new IntArray(3);
        info.set(0, startPos);
        info.set(1, activeRows);
        info.set(2, SLOT);
        return info;
    }

    private static WorkerGrid grid(int rows) {
        WorkerGrid worker = new WorkerGrid1D(rows * HEADS * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        return worker;
    }

    /** The kernel's own addressing, checked against the store's host-side map. */
    private static void assertAddressing(Store store) {
        int layerOff = KvBlockAddress.layerOffset(LAYER, KV_DIM, BLOCK_CFG);
        for (int p : new int[] {0, 15, 16, 17, 1000, 2047}) {
            int kernel =
                    KvBlockAddress.offset(
                                    store.blockTable,
                                    SLOT,
                                    p,
                                    layerOff,
                                    KV_DIM,
                                    BLOCK_CFG,
                                    BLOCK_STRIDE)
                            + 2 * HEAD_SIZE;
            assertEquals("addressing of position " + p, store.keyBase(p, 2), kernel);
        }
    }

    /** Unit roundoff of FP32. */
    private static final double EPS = Math.scalb(1.0, -24);

    /**
     * Score error allowance on the cancellation cases, in units of {@code eps * sum |q_d k_d|}: the
     * forward bound of any FP32 summation order is {@code n * eps * sum |terms|}; a summation that
     * cancels its large terms early stays well inside it, one that cancels them late does not, and
     * neither is wrong. Eight units is under {@code n = 256} by a wide margin and above every
     * order's observed error by one.
     */
    private static final double TERM_UNITS = 8.0;

    /**
     * Errors of one kernel's stored scores and outputs against the FP64 reference. {@code
     * scoreMaxTermUnits} is the largest score error in units of {@code eps * sum |q_d k_d|} for
     * that score.
     */
    private record Errors(
            double scoreMaxAbs,
            double scoreMaxRel,
            double scoreMaxTermUnits,
            double outMaxAbs,
            double outRelL2,
            double scoreMaxAbsVsSequential) {
        /** The gate for the random, tied and scaled cases: no worse than a multiple of control. */
        boolean within(Errors control) {
            return scoreMaxAbs <= Math.max(RATIO_BOUND * control.scoreMaxAbs, SCORE_FLOOR)
                    && outputsWithin(control);
        }

        /**
         * The gate for the cancellation cases: every score inside the term-magnitude bound, and the
         * outputs no worse than a multiple of control's. The score ratio against control is
         * reported, not asserted, because it measures whose summation order the input's
         * cancellation structure happens to match (see the two cancellation cases).
         */
        boolean withinTermBound(Errors control) {
            return scoreMaxTermUnits <= TERM_UNITS && outputsWithin(control);
        }

        private boolean outputsWithin(Errors control) {
            return outMaxAbs <= Math.max(RATIO_BOUND * control.outMaxAbs, OUTPUT_FLOOR)
                    && outRelL2 <= Math.max(RATIO_BOUND * control.outRelL2, OUTPUT_FLOOR);
        }

        @Override
        public String toString() {
            return String.format(
                    Locale.ROOT,
                    "scores maxAbs %.3e maxRel %.3e termUnits %.2f (vs FP32 sequential %.3e) |"
                            + " outputs maxAbs %.3e relL2 %.3e",
                    scoreMaxAbs,
                    scoreMaxRel,
                    scoreMaxTermUnits,
                    scoreMaxAbsVsSequential,
                    outMaxAbs,
                    outRelL2);
        }
    }

    /** Runs the control and three candidates over one input and returns each one's errors. */
    private static Errors[] run(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        assertTrue(what + ": range exceeds the context", startPos + activeRows <= CAPACITY);
        assertAddressing(store);
        IntArray info = batchInfo(startPos, activeRows);
        int n = 4;
        FloatArray[] out = new FloatArray[n];
        FloatArray[] scores = new FloatArray[n];
        for (int i = 0; i < n; i++) {
            out[i] = new FloatArray(rows * HEADS * HEAD_SIZE);
            scores[i] = new FloatArray(rows * HEADS * CAPACITY);
            out[i].init(Float.NaN);
            scores[i].init(Float.NaN);
        }
        TaskGraph graph =
                new TaskGraph("attn")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                info,
                                q,
                                store.keys,
                                store.values,
                                store.blockTable,
                                out[0],
                                out[1],
                                out[2],
                                out[3],
                                scores[0],
                                scores[1],
                                scores[2],
                                scores[3])
                        .task(
                                "control",
                                Qwen35BatchKernels::attentionBatchFP16PagedScoredStagedWide,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                out[0],
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scores[0],
                                CAPACITY)
                        .task(
                                "warp",
                                Qwen35BatchKernels::attentionBatchFP16PagedScoredWarp,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                out[1],
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scores[1],
                                CAPACITY)
                        .task(
                                "noshuffle",
                                Qwen35AttentionWarpNumericsAccelTest::warpWithoutTheShuffle,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                out[2],
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scores[2],
                                CAPACITY)
                        .task(
                                "misindexed",
                                Qwen35AttentionWarpNumericsAccelTest::warpWithMisindexedKeys,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                out[3],
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scores[3],
                                CAPACITY)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION,
                                out[0],
                                out[1],
                                out[2],
                                out[3],
                                scores[0],
                                scores[1],
                                scores[2],
                                scores[3]);
        GridScheduler scheduler = new GridScheduler();
        for (String t : new String[] {"control", "warp", "noshuffle", "misindexed"}) {
            scheduler.addWorkerGrid("attn." + t, grid(rows));
        }
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        // The FP64 reference: causal scores, softmax, weighted values, per active (row, head).
        Errors[] errors = new Errors[n];
        double[] scoreMaxAbs = new double[n];
        double[] scoreMaxRel = new double[n];
        double[] outMaxAbs = new double[n];
        double[] outErr2 = new double[n];
        double outRef2 = 0;
        double[] refScores = new double[CAPACITY];
        double[] termSums = new double[CAPACITY];
        // The CPU reference path's own order: a running FP32 sum over d = 0..255. The control
        // shares it (FMA-contracted), the candidate does not; the distance of each kernel from it
        // says how much of a model-level parity shift against that path is order, not accuracy.
        float[] seqScores = new float[CAPACITY];
        double[] scoreMaxAbsVsSeq = new double[n];
        double[] refOut = new double[HEAD_SIZE];
        double[] scoreMaxTermUnits = new double[n];
        for (int row = 0; row < activeRows; row++) {
            int position = startPos + row;
            for (int head = 0; head < HEADS; head++) {
                int kvHead = head / KV_MUL;
                int qBase = row * HEADS * HEAD_SIZE + head * HEAD_SIZE;
                double scoreScale = 0;
                for (int p = 0; p <= position; p++) {
                    int kb = store.keyBase(p, kvHead);
                    double s = 0;
                    double terms = 0;
                    float seq = 0.0f;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        float qd = q.get(qBase + d);
                        float kd = store.keys.get(kb + d).getFloat32();
                        double t = (double) qd * (double) kd;
                        s += t;
                        terms += Math.abs(t);
                        seq = Math.fma(qd, kd, seq);
                    }
                    refScores[p] = s;
                    termSums[p] = terms;
                    seqScores[p] = seq;
                    scoreScale = Math.max(scoreScale, Math.abs(s));
                }
                double invSqrt = 1.0 / Math.sqrt(HEAD_SIZE);
                double max = Double.NEGATIVE_INFINITY;
                for (int p = 0; p <= position; p++) {
                    max = Math.max(max, refScores[p] * invSqrt);
                }
                double denominator = 0;
                Arrays.fill(refOut, 0);
                for (int p = 0; p <= position; p++) {
                    double w = Math.exp(refScores[p] * invSqrt - max);
                    denominator += w;
                    int vb = store.keyBase(p, kvHead);
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        refOut[d] += w * (double) store.values.get(vb + d).getFloat32();
                    }
                }
                int sBase = (row * HEADS + head) * CAPACITY;
                int oBase = row * HEADS * HEAD_SIZE + head * HEAD_SIZE;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    refOut[d] /= denominator;
                    outRef2 += refOut[d] * refOut[d];
                }
                for (int k = 0; k < n; k++) {
                    for (int p = 0; p <= position; p++) {
                        float got = scores[k].get(sBase + p);
                        assertTrue(
                                what
                                        + ": kernel "
                                        + k
                                        + " score not finite at row "
                                        + row
                                        + " head "
                                        + head
                                        + " p "
                                        + p,
                                Float.isFinite(got));
                        double err = Math.abs(got - refScores[p]);
                        scoreMaxAbs[k] = Math.max(scoreMaxAbs[k], err);
                        scoreMaxRel[k] =
                                Math.max(scoreMaxRel[k], err / Math.max(scoreScale, 1e-30));
                        scoreMaxTermUnits[k] =
                                Math.max(
                                        scoreMaxTermUnits[k],
                                        err / (EPS * Math.max(termSums[p], 1e-30)));
                        scoreMaxAbsVsSeq[k] =
                                Math.max(scoreMaxAbsVsSeq[k], Math.abs(got - seqScores[p]));
                    }
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        float got = out[k].get(oBase + d);
                        assertTrue(
                                what
                                        + ": kernel "
                                        + k
                                        + " output not finite at row "
                                        + row
                                        + " head "
                                        + head
                                        + " d "
                                        + d,
                                Float.isFinite(got));
                        double err = got - refOut[d];
                        outMaxAbs[k] = Math.max(outMaxAbs[k], Math.abs(err));
                        outErr2[k] += err * err;
                    }
                }
            }
        }
        for (int k = 0; k < n; k++) {
            errors[k] =
                    new Errors(
                            scoreMaxAbs[k],
                            scoreMaxRel[k],
                            scoreMaxTermUnits[k],
                            outMaxAbs[k],
                            Math.sqrt(outErr2[k] / Math.max(outRef2, 1e-300)),
                            scoreMaxAbsVsSeq[k]);
        }
        System.out.printf(Locale.ROOT, "[numerics] %-28s control    %s%n", what, errors[0]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s candidate  %s%n", what, errors[1]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s no-shuffle %s%n", what, errors[2]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s misindexed %s%n", what, errors[3]);
        return errors;
    }

    private static void assertGate(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        Errors[] e = run(what, store, rows, startPos, activeRows, q);
        assertTrue(
                what + ": candidate " + e[1] + " exceeds " + RATIO_BOUND + "x control " + e[0],
                e[1].within(e[0]));
        assertTrue(
                what + ": the shuffle-less candidate passed the gate: " + e[2], !e[2].within(e[0]));
        assertTrue(
                what + ": the misindexed candidate passed the gate: " + e[3], !e[3].within(e[0]));
    }

    private static void assertCancellationGate(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        Errors[] e = run(what, store, rows, startPos, activeRows, q);
        System.out.printf(
                Locale.ROOT,
                "[numerics] %-28s score error ratio candidate/control %.2f%n",
                what,
                e[1].scoreMaxAbs() / e[0].scoreMaxAbs());
        assertTrue(
                what + ": control outside the term bound: " + e[0],
                e[0].scoreMaxTermUnits() <= TERM_UNITS);
        assertTrue(
                what
                        + ": candidate "
                        + e[1]
                        + " outside the term bound or "
                        + RATIO_BOUND
                        + "x control outputs "
                        + e[0],
                e[1].withinTermBound(e[0]));
        assertTrue(
                what + ": the shuffle-less candidate passed the gate: " + e[2],
                !e[2].withinTermBound(e[0]));
        assertTrue(
                what + ": the misindexed candidate passed the gate: " + e[3],
                !e[3].withinTermBound(e[0]));
    }

    private static FloatArray q(int rows, long seed) {
        return queries(rows, seed, 1.0f, 0);
    }

    /** Position zero onward: the shortest ranges, one to thirty-two keys. */
    @Test
    public void theShortestRanges() throws Exception {
        assertGate("start 0", new Store(1L, Keys.RANDOM), 32, 0, 32, q(32, 11L));
    }

    /** Ranges ending around the 16- and 128-position boundaries and a mid-page start. */
    @Test
    public void theTileAndPageBoundaries() throws Exception {
        assertGate("start 100", new Store(7L, Keys.RANDOM), 32, 100, 32, q(32, 12L));
        assertGate("start 112", new Store(9L, Keys.RANDOM), 32, 112, 32, q(32, 13L));
        assertGate("start 15", new Store(2L, Keys.RANDOM), 32, 15, 32, q(32, 14L));
    }

    /** The last positions of a 2048 context, and a partial chunk. */
    @Test
    public void theLongestRangesAndAPartialChunk() throws Exception {
        assertGate("start 2016", new Store(3L, Keys.RANDOM), 32, 2016, 32, q(32, 15L));
        assertGate("partial 5 of 32", new Store(4L, Keys.RANDOM), 32, 1000, 5, q(32, 16L));
    }

    /** Tied keys: equal scores at every fourth position, the maximum attained many times. */
    @Test
    public void tiedKeys() throws Exception {
        assertGate("ties start 1500", new Store(5L, Keys.TIED), 32, 1500, 32, q(32, 17L));
    }

    /**
     * Cancelling keys and paired queries: dot products that are small residuals of large terms.
     * Adjacent pairs cancel inside the control's running sum and the candidate is the worse of the
     * two by the reference-relative measure; pairs 32 apart cancel inside one candidate lane and
     * the control is. Both stay inside the term-magnitude bound; the ratios are printed.
     */
    @Test
    public void cancellingKeys() throws Exception {
        assertCancellationGate(
                "cancel adjacent 700",
                new Store(6L, Keys.CANCELLING_ADJACENT),
                32,
                700,
                32,
                queries(32, 18L, 1.0f, 1));
        assertCancellationGate(
                "cancel stride32 700",
                new Store(6L, Keys.CANCELLING_STRIDE32),
                32,
                700,
                32,
                queries(32, 18L, 1.0f, 32));
    }

    /**
     * Heavy-tailed queries and keys (four dimensions a hundred times the rest, as models with
     * massive activations have): every FP32 order carries an error of the order of the large terms'
     * rounding — the running sum's is tens of {@code eps * sum |terms|}, well outside the
     * cancellation cases' allowance, so this case is gated on the ratio — and each kernel's
     * distance from the FP32 running sum the CPU reference path computes is printed: the control's
     * is zero, so a model-level parity metric against that path measures the candidate's order as
     * error and the control's not at all.
     */
    @Test
    public void heavyTailedDimensions() throws Exception {
        assertGate(
                "heavy start 900",
                new Store(10L, Keys.HEAVY_TAILED),
                32,
                900,
                32,
                queries(32, 22L, 1.0f, 0, true));
    }

    /** Query scales that put the scaled logits across narrow, moderate and wide finite ranges. */
    @Test
    public void variedLogitRanges() throws Exception {
        assertGate(
                "scale 0.05", new Store(8L, Keys.RANDOM), 32, 900, 32, queries(32, 19L, 0.05f, 0));
        assertGate("scale 4", new Store(8L, Keys.RANDOM), 32, 900, 32, queries(32, 20L, 4.0f, 0));
        assertGate("scale 24", new Store(8L, Keys.RANDOM), 32, 900, 32, queries(32, 21L, 24.0f, 0));
    }

    /**
     * Kernel time of both forms at several context depths and widths 32, 512 and 1024, from the
     * TornadoVM profiler, alternating order after a warm-up. Prints distributions only.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        Store store = new Store(9L, Keys.RANDOM);
        int[][] cases = {
            {32, 0},
            {32, 480},
            {32, 992},
            {32, 2016},
            {512, 0},
            {512, 512},
            {512, 1536},
            {1024, 0},
            {1024, 1024}
        };
        for (int[] c : cases) {
            int rows = c[0];
            int startPos = c[1];
            FloatArray q = queries(rows, 99L, 1.0f, 0);
            IntArray info = batchInfo(startPos, rows);
            FloatArray outA = new FloatArray(rows * HEADS * HEAD_SIZE);
            FloatArray outB = new FloatArray(rows * HEADS * HEAD_SIZE);
            FloatArray scoresA = new FloatArray(rows * HEADS * CAPACITY);
            FloatArray scoresB = new FloatArray(rows * HEADS * CAPACITY);
            TaskGraph a =
                    new TaskGraph("ctl")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    store.blockTable,
                                    scoresA)
                            .task(
                                    "p",
                                    Qwen35BatchKernels::attentionBatchFP16PagedScoredStagedWide,
                                    new KernelContext(),
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    outA,
                                    HEADS,
                                    HEAD_SIZE,
                                    KV_DIM,
                                    KV_MUL,
                                    LAYER,
                                    store.blockTable,
                                    BLOCK_CFG,
                                    BLOCK_STRIDE,
                                    LOCAL,
                                    scoresA,
                                    CAPACITY)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
            TaskGraph b =
                    new TaskGraph("warp")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    store.blockTable,
                                    scoresB)
                            .task(
                                    "p",
                                    Qwen35BatchKernels::attentionBatchFP16PagedScoredWarp,
                                    new KernelContext(),
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    outB,
                                    HEADS,
                                    HEAD_SIZE,
                                    KV_DIM,
                                    KV_MUL,
                                    LAYER,
                                    store.blockTable,
                                    BLOCK_CFG,
                                    BLOCK_STRIDE,
                                    LOCAL,
                                    scoresB,
                                    CAPACITY)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("ctl.p", grid(rows));
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("warp.p", grid(rows));
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
                report("wide rows=" + rows + " start=" + startPos, tA);
                report("warp rows=" + rows + " start=" + startPos, tB);
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
                "[screen] %-32s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }

    // ---- negative controls: the candidate with one thing broken ------------------------------

    private static float warpSum(KernelContext context, float value) {
        float sum = value;
        sum += context.simdShuffleDown(sum, 16);
        sum += context.simdShuffleDown(sum, 8);
        sum += context.simdShuffleDown(sum, 4);
        sum += context.simdShuffleDown(sum, 2);
        sum += context.simdShuffleDown(sum, 1);
        return context.simdBroadcastFirst(sum);
    }

    /** The candidate without its shuffle reduction: lane zero stores its own eight products. */
    public static void warpWithoutTheShuffle(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
        int tid = context.localIdx;
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 by warps: this lane's eight query elements, then this warp's positions.
        int warp = tid >> 5;
        int lane = tid & 31;
        int queryBase = row * heads * headSize + head * headSize;
        float q0 = queryBatch.get(queryBase + lane);
        float q1 = queryBatch.get(queryBase + lane + 32);
        float q2 = queryBatch.get(queryBase + lane + 64);
        float q3 = queryBatch.get(queryBase + lane + 96);
        float q4 = queryBatch.get(queryBase + lane + 128);
        float q5 = queryBatch.get(queryBase + lane + 160);
        float q6 = queryBatch.get(queryBase + lane + 192);
        float q7 = queryBatch.get(queryBase + lane + 224);
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = warp; p <= position; p += 4) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize
                            + lane;
            float partial = q0 * keyCache.get(base).getFloat32();
            partial += q1 * keyCache.get(base + 32).getFloat32();
            partial += q2 * keyCache.get(base + 64).getFloat32();
            partial += q3 * keyCache.get(base + 96).getFloat32();
            partial += q4 * keyCache.get(base + 128).getFloat32();
            partial += q5 * keyCache.get(base + 160).getFloat32();
            partial += q6 * keyCache.get(base + 192).getFloat32();
            partial += q7 * keyCache.get(base + 224).getFloat32();
            float score = partial; // NEGATIVE CONTROL: lane zero's partial only
            if (lane == 0) {
                scores.set(scoreBase + p, score);
                maxScore = TornadoMath.max(maxScore, score * invSqrt);
            }
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, 128 positions at a time, as in the wide kernel.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(128);
        float[] accumulated = new float[4];
        for (int t = 0; t < 4; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += 128) {
            int tileEnd = tileStart + 128 - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /** The candidate reading each lane's keys from the neighbouring lane's dimensions. */
    public static void warpWithMisindexedKeys(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
        int tid = context.localIdx;
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 by warps: this lane's eight query elements, then this warp's positions.
        int warp = tid >> 5;
        int lane = tid & 31;
        int queryBase = row * heads * headSize + head * headSize;
        float q0 = queryBatch.get(queryBase + lane);
        float q1 = queryBatch.get(queryBase + lane + 32);
        float q2 = queryBatch.get(queryBase + lane + 64);
        float q3 = queryBatch.get(queryBase + lane + 96);
        float q4 = queryBatch.get(queryBase + lane + 128);
        float q5 = queryBatch.get(queryBase + lane + 160);
        float q6 = queryBatch.get(queryBase + lane + 192);
        float q7 = queryBatch.get(queryBase + lane + 224);
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = warp; p <= position; p += 4) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize
                            + ((lane + 1) & 31); // NEGATIVE CONTROL: neighbour's dimensions
            float partial = q0 * keyCache.get(base).getFloat32();
            partial += q1 * keyCache.get(base + 32).getFloat32();
            partial += q2 * keyCache.get(base + 64).getFloat32();
            partial += q3 * keyCache.get(base + 96).getFloat32();
            partial += q4 * keyCache.get(base + 128).getFloat32();
            partial += q5 * keyCache.get(base + 160).getFloat32();
            partial += q6 * keyCache.get(base + 192).getFloat32();
            partial += q7 * keyCache.get(base + 224).getFloat32();
            float score = warpSum(context, partial);
            if (lane == 0) {
                scores.set(scoreBase + p, score);
                maxScore = TornadoMath.max(maxScore, score * invSqrt);
            }
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, 128 positions at a time, as in the wide kernel.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(128);
        float[] accumulated = new float[4];
        for (int t = 0; t < 4; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += 128) {
            int tileEnd = tileStart + 128 - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }
}
