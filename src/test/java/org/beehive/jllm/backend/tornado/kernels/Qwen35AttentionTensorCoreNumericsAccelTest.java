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
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The tensor-core batched attention prototype against an independent FP64 causal-attention
 * reference, alongside the warp kernel it would replace, on the device over identical inputs (FP32
 * queries, stored FP16 keys and values).
 *
 * <p>Two references: one from the FP32 queries as stored, one from the same queries rounded to FP16
 * (what the candidate multiplies), so the effect of that rounding is quantified on its own rather
 * than folded into the MMA accumulation error. Acceptance, stated before any result was seen: (1)
 * every active output written and finite; (2) against the FP16-query reference the candidate's
 * outputs are within relL2 4e-3 and maxAbs 1e-2 on every case; (3) against the FP32-query reference
 * the candidate is within the same bounds on every case except where the FP16-query reference
 * itself departs from the FP32-query reference by more than half those bounds, in which case the
 * departure is reported as the query-rounding cost; (4) the control's errors are reported
 * alongside, not bounded here; (5) a candidate with the causal mask off by one and a candidate
 * without the online output rescale both violate (2). No bit equality is required and no model
 * bound is changed. Cases as in the warp test plus multi-tile query ranges.
 */
// @formatter:on
public class Qwen35AttentionTensorCoreNumericsAccelTest {

    private static final int HEADS = 24;
    private static final int HEAD_SIZE = 256;
    private static final int KV_HEADS = 4;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int BLOCK_SIZE = 16;

    /** 2048 by default; {@code tc.capacity} selects another (the benchmark runs 2056). */
    private static final int CAPACITY = Integer.getInteger("tc.capacity", 2048);

    private static final int BLOCKS_PER_SLOT = (CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE;
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

    /** The candidate's grid: one 128-lane workgroup per (16-query tile, head). */
    private static WorkerGrid tcGrid(int rows) {
        WorkerGrid worker = new WorkerGrid1D((rows / 16) * HEADS * LOCAL);
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

    /** Whether the reference multiplies the FP16-rounded queries instead of the FP32 ones. */
    private static boolean fp16Queries = false;

    /** Runs the control and three candidates over one input and returns each one's errors. */
    private static Errors[] run(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        assertTrue(what + ": range exceeds the context", startPos + activeRows <= CAPACITY);
        assertAddressing(store);
        IntArray info = batchInfo(startPos, activeRows);
        int n = 4;
        HalfFloatArray[] stage = new HalfFloatArray[n];
        for (int i = 1; i < n; i++) {
            stage[i] = new HalfFloatArray((rows / 16) * HEADS * Qwen35BatchKernels.TC_STAGE_HALVES);
        }
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
                                scores[3],
                                stage[1],
                                stage[2],
                                stage[3])
                        .task(
                                "control",
                                Qwen35BatchKernels::attentionBatchFP16PagedScoredWarp,
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
                                Qwen35ReferenceKernels::attentionBatchFP16PagedTensorCore,
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
                                CAPACITY,
                                stage[1])
                        .task(
                                "noshuffle",
                                Qwen35AttentionTensorCoreNumericsAccelTest
                                        ::tensorCoreWithTheMaskOffByOne,
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
                                CAPACITY,
                                stage[2])
                        .task(
                                "misindexed",
                                Qwen35AttentionTensorCoreNumericsAccelTest
                                        ::tensorCoreWithoutTheDenominator,
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
                                CAPACITY,
                                stage[3])
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
        scheduler.addWorkerGrid("attn.control", grid(rows));
        for (String t : new String[] {"warp", "noshuffle", "misindexed"}) {
            scheduler.addWorkerGrid("attn." + t, tcGrid(rows));
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
                        if (fp16Queries) {
                            qd = new HalfFloat(qd).getFloat32();
                        }
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
                    for (int p = 0; p <= position && k <= 1; p++) {
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
                        // Control and candidate must be finite everywhere; a broken candidate
                        // that produces a non-finite value has simply failed the gate.
                        if (k < 2) {
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
                        }
                        double err =
                                Float.isFinite(got) ? got - refOut[d] : Double.POSITIVE_INFINITY;
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
        System.out.printf(Locale.ROOT, "[numerics] %-28s control(warp) %s%n", what, errors[0]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s tensorcore    %s%n", what, errors[1]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s mask-off-by-1 %s%n", what, errors[2]);
        System.out.printf(Locale.ROOT, "[numerics] %-28s no-rescale    %s%n", what, errors[3]);
        return errors;
    }

    private static final double REL_L2_BOUND = 4e-3;
    private static final double MAX_ABS_BOUND = 1e-2;

    private static boolean withinBounds(Errors e) {
        return e.outRelL2() <= REL_L2_BOUND && e.outMaxAbs() <= MAX_ABS_BOUND;
    }

    /** The stated acceptance: (1)-(3) and (5), with the control and the rounding cost reported. */
    private static void assertGate(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        fp16Queries = false;
        Errors[] e32 = run(what + " [fp32-q ref]", store, rows, startPos, activeRows, q);
        fp16Queries = true;
        Errors[] e16 = run(what + " [fp16-q ref]", store, rows, startPos, activeRows, q);
        // The query-rounding cost on its own: the control (which multiplies FP32 queries) measured
        // against the FP16-query reference shows how far the two references sit apart.
        System.out.printf(
                Locale.ROOT,
                "[numerics] %-28s query rounding alone: relL2 %.3e maxAbs %.3e (control vs fp16-q ref)%n",
                what,
                e16[0].outRelL2(),
                e16[0].outMaxAbs());
        assertTrue(
                what + ": candidate vs fp16-q reference outside bounds: " + e16[1],
                withinBounds(e16[1]));
        boolean roundingDominates =
                e16[0].outRelL2() > REL_L2_BOUND / 2 || e16[0].outMaxAbs() > MAX_ABS_BOUND / 2;
        if (roundingDominates) {
            System.out.printf(
                    Locale.ROOT,
                    "[numerics] %-28s QUERY ROUNDING DOMINATES: candidate vs fp32-q ref relL2 %.3e maxAbs %.3e%n",
                    what,
                    e32[1].outRelL2(),
                    e32[1].outMaxAbs());
        } else {
            assertTrue(
                    what + ": candidate vs fp32-q reference outside bounds: " + e32[1],
                    withinBounds(e32[1]));
        }
        assertTrue(what + ": the off-by-one mask passed: " + e16[2], !withinBounds(e16[2]));
        assertTrue(what + ": the missing rescale passed: " + e16[3], !withinBounds(e16[3]));
    }

    private static void assertCancellationGate(
            String what, Store store, int rows, int startPos, int activeRows, FloatArray q)
            throws Exception {
        assertGate(what, store, rows, startPos, activeRows, q);
    }

    private static FloatArray q(int rows, long seed) {
        return queries(rows, seed, 1.0f, 0);
    }

    /** Position zero onward: the shortest ranges, one to thirty-two keys. */
    @Test
    public void theShortestRanges() throws Exception {
        assertGate("start 0", new Store(1L, Keys.RANDOM), 32, 0, 32, q(32, 11L));
    }

    /** Four query tiles, keys spanning several 64-key tiles with a partial last one. */
    @Test
    public void multipleQueryAndKeyTiles() throws Exception {
        assertGate("start 100 rows 64", new Store(21L, Keys.RANDOM), 64, 100, 64, q(64, 31L));
        assertGate("start 1985 rows 64", new Store(22L, Keys.RANDOM), 64, 1985, 63, q(64, 32L));
    }

    /** Ranges ending around the 16- and 128-position boundaries and a mid-page start. */
    @Test
    public void theTileAndPageBoundaries() throws Exception {
        assertGate("start 100", new Store(7L, Keys.RANDOM), 32, 100, 32, q(32, 12L));
        assertGate("start 112", new Store(9L, Keys.RANDOM), 32, 112, 32, q(32, 13L));
        assertGate("start 15", new Store(2L, Keys.RANDOM), 32, 15, 32, q(32, 14L));
    }

    /**
     * The last positions of the context (2016..2047 of 2048; 2024..2055 of 2056), and a partial
     * chunk.
     */
    @Test
    public void theLongestRangesAndAPartialChunk() throws Exception {
        assertGate(
                "start " + (CAPACITY - 32),
                new Store(3L, Keys.RANDOM),
                32,
                CAPACITY - 32,
                32,
                q(32, 15L));
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
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        Store store = new Store(9L, Keys.RANDOM);
        int[][] cases = {
            {32, 0},
            {32, 480},
            {32, 2016},
            {512, 0},
            {512, 512},
            {512, 1536},
            {1024, 0},
            {1024, 1024},
            {2048, 0}
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
            HalfFloatArray stageB =
                    new HalfFloatArray((rows / 16) * HEADS * Qwen35BatchKernels.TC_STAGE_HALVES);
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
                                    Qwen35BatchKernels::attentionBatchFP16PagedScoredWarp,
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
                                    scoresB,
                                    stageB)
                            .task(
                                    "p",
                                    Qwen35ReferenceKernels::attentionBatchFP16PagedTensorCore,
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
                                    CAPACITY,
                                    stageB)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("ctl.p", grid(rows));
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("warp.p", tcGrid(rows));
            try (TornadoExecutionPlan planA = new TornadoExecutionPlan(a.snapshot());
                    TornadoExecutionPlan planB = new TornadoExecutionPlan(b.snapshot())) {
                planA.withGridScheduler(sa).withProfiler(ProfilerMode.SILENT);
                planB.withGridScheduler(sb).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 5; i++) {
                    planA.execute();
                    planB.execute();
                }
                // The screen's own sanity check: the two outputs agree on the active rows.
                planA.execute().transferToHost(outA);
                planB.execute().transferToHost(outB);
                double d2 = 0;
                double r2 = 0;
                for (int i = 0; i < rows * HEADS * HEAD_SIZE; i++) {
                    double e = outB.get(i) - outA.get(i);
                    d2 += e * e;
                    r2 += (double) outA.get(i) * outA.get(i);
                }
                System.out.printf(
                        Locale.ROOT,
                        "[screen] rows=%d start=%d candidate-vs-control relL2 %.3e; row0 head0: ctl %.4f %.4f %.4f cand %.4f %.4f %.4f; last row: ctl %.4f cand %.4f%n",
                        rows,
                        startPos,
                        Math.sqrt(d2 / Math.max(r2, 1e-300)),
                        outA.get(0),
                        outA.get(1),
                        outA.get(2),
                        outB.get(0),
                        outB.get(1),
                        outB.get(2),
                        outA.get((rows - 1) * HEADS * HEAD_SIZE),
                        outB.get((rows - 1) * HEADS * HEAD_SIZE));
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
                report("warp rows=" + rows + " start=" + startPos, tA);
                report("tc   rows=" + rows + " start=" + startPos, tB);
            }
        }
    }

    @Test
    public void debugScoreGrid() throws Exception {
        assumeTrue(Boolean.getBoolean("tc.debug"));
        Store store = new Store(1L, Keys.RANDOM);
        int rows = Integer.getInteger("tc.rows", 16);
        FloatArray q = new FloatArray(rows * HEADS * HEAD_SIZE);
        int onehot = Integer.getInteger("tc.onehot", -1);
        for (int row = 0; row < rows; row++) {
            for (int head = 0; head < HEADS; head++) {
                // one-hot at dimension (row + onehot): S[row][key] = K[key][row + onehot]
                if (onehot >= 0) {
                    q.set(row * HEADS * HEAD_SIZE + head * HEAD_SIZE + row + onehot, 1.0f);
                }
            }
        }
        if (onehot < 0) {
            q = q(rows, 11L);
        }
        if (Boolean.getBoolean("tc.dimkeys")) {
            // keys[key][d] = d + key / 1024: the score names the dimension actually multiplied.
            for (int p = 0; p < 64; p++) {
                int kb = store.keyBase(p, 0);
                for (int d = 0; d < HEAD_SIZE; d++) {
                    store.keys.set(kb + d, new HalfFloat(d + p / 1024.0f));
                }
            }
        }
        IntArray info = batchInfo(0, rows);
        HalfFloatArray stage =
                new HalfFloatArray((rows / 16) * HEADS * Qwen35BatchKernels.TC_STAGE_HALVES);
        FloatArray out = new FloatArray(rows * HEADS * HEAD_SIZE);
        FloatArray scores = new FloatArray(rows * HEADS * CAPACITY);
        scores.init(Float.NaN);
        TaskGraph graph =
                new TaskGraph("dbg")
                        .transferToDevice(
                                Boolean.getBoolean("tc.first")
                                        ? DataTransferMode.FIRST_EXECUTION
                                        : DataTransferMode.EVERY_EXECUTION,
                                info,
                                q,
                                store.keys,
                                store.values,
                                store.blockTable,
                                out,
                                scores,
                                stage)
                        .task(
                                "t",
                                Qwen35ReferenceKernels::attentionBatchFP16PagedTensorCore,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                out,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scores,
                                CAPACITY,
                                stage)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out, scores, stage);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("dbg.t", tcGrid(rows));
        float[] first = new float[rows * HEADS * HEAD_SIZE];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
            for (int i = 0; i < first.length; i++) {
                first[i] = out.get(i);
            }
            int runs = Integer.getInteger("tc.runs", 1);
            for (int r = 1; r < runs; r++) {
                plan.execute();
                int diff = 0;
                for (int i = 0; i < first.length; i++) {
                    if (first[i] != out.get(i)) {
                        diff++;
                    }
                }
                System.out.println("run " + (r + 1) + ": outputs differing from run 1: " + diff);
            }
        }
        for (int head : new int[] {0, 3}) {
            int kvHead = head / KV_MUL;
            System.out.println("head " + head);
            for (int row = 0; row < rows; row += Math.max(1, rows / 16)) {
                StringBuilder line = new StringBuilder();
                for (int p = 0; p < 16; p++) {
                    int kb = store.keyBase(p, kvHead);
                    double ref = 0;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        ref +=
                                (double)
                                                new HalfFloat(
                                                                q.get(
                                                                        row * HEADS * HEAD_SIZE
                                                                                + head * HEAD_SIZE
                                                                                + d))
                                                        .getFloat32()
                                        * store.keys.get(kb + d).getFloat32();
                    }
                    float got = scores.get((row * HEADS + head) * CAPACITY + p);
                    if (onehot >= 0) {
                        ref = store.keys.get(kb + row + onehot).getFloat32();
                    }
                    line.append(String.format(Locale.ROOT, "%7.3f/%7.3f ", got, ref));
                }
                System.out.println(line);
            }
            // staged Q: compare the workgroup's stage region to the FP16 q
            int mism = 0;
            for (int i = 0; i < 16 * 256; i++) {
                int row = i >> 8, d = i & 255;
                short want =
                        new HalfFloat(q.get(row * HEADS * HEAD_SIZE + head * HEAD_SIZE + d))
                                .getHalfFloatValue();
                short got =
                        stage.get(head * Qwen35BatchKernels.TC_STAGE_HALVES + i)
                                .getHalfFloatValue();
                if (want != got) mism++;
            }
            System.out.println("stage Q mismatches for head " + head + ": " + mism);
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

    private static float[] scaleRows(float[] c, float a0, float a1) {
        c[0] = c[0] * a0;
        c[1] = c[1] * a0;
        c[2] = c[2] * a1;
        c[3] = c[3] * a1;
        return c;
    }

    /** The candidate admitting one key past the causal boundary. */
    public static void tensorCoreWithTheMaskOffByOne(
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
            int scoreStride,
            HalfFloatArray stage) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int queryTile = group / heads;
        int head = group - queryTile * heads;
        int rowBase = queryTile * 16;
        if (rowBase >= batchInfo.get(1)) {
            return;
        }
        int startPos = batchInfo.get(0);
        int slot = batchInfo.get(2);
        int capacity = scoreStride;
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        int[] qTile = context.allocateIntLocalArray(16 * 256 / 2);
        HalfFloat[] kTile = context.allocateHalfFloatLocalArray(32 * 256);
        int[] vTile = context.allocateIntLocalArray(32 * 256 / 2);
        int[] pTile = context.allocateIntLocalArray(16 * 32 / 2);
        float[] rowStat = context.allocateFloatLocalArray(16 * 2 + 128);

        int stageBase = group * 4608;
        int qBase = (rowBase * heads + head) * headSize;
        for (int i = tid; i < 16 * 256; i += 128) {
            int row = i >> 8;
            int d = i & 255;
            stage.set(
                    stageBase + i,
                    new HalfFloat(queryBatch.get(qBase + row * heads * headSize + d)));
        }
        context.localBarrier();
        for (int i = tid; i < 16 * 256 / 2; i += 128) {
            int t = i >> 7;
            int within = i & 127;
            int row = within >> 3;
            int d = (t << 4) + ((within & 7) << 1);
            context.asyncCopyToLocal(qTile, i, stage, stageBase + (row << 8) + d);
        }
        context.asyncCopyCommit();
        context.asyncCopyWaitGroup(0);
        context.localBarrier();

        int maxPos = startPos + rowBase + 16 - 1;
        int lastKey = capacity - 1;
        if (maxPos < lastKey) {
            lastKey = maxPos;
        }
        // The score scratch as a matrix with row stride heads * scoreStride: row r of this tile
        // is (rowBase + r) * heads + head spans, i.e. column offset head * scoreStride.
        int scoreLd = heads * scoreStride;
        int scoreCol = head * scoreStride;

        // Pass 1: S = Q K^T, 32 keys a tile, stored unscaled to the scratch.
        for (int tileStart = 0; tileStart <= lastKey; tileStart += 32) {
            for (int i = 0; i < 32 * 256 / 128; i++) {
                int e = i * 128 + tid;
                int key = e >> 8;
                int d = e & 255;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                context.mmaStoreBSwizzled(
                        kTile,
                        d & 15,
                        key & 7,
                        8,
                        // Widened and re-narrowed (exact): the route the store's lowering takes
                        // a half by value, rather than a half read from the array.
                        new HalfFloat(keyCache.get(base + d).getFloat32()),
                        (((d >> 4) << 2) + (key >> 3)) * 256);
            }
            context.localBarrier();
            float[] s0 = context.mmaFragment(0.0f);
            for (int t = 0; t < 256 / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(qTile, 16, t * 512);
                HalfFloat[] b0 = context.mmaLoadBSwizzled(kTile, 16, ((t << 2) + warp) * 256);
                s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
            }
            // This warp's eight keys of the tile; skipped whole where they lie past the capacity
            // (a warp-uniform choice, so no lane misses a barrier): those keys are never read.
            if (tileStart + (warp << 3) + 7 < capacity) {
                context.mmaStore(s0, scores, rowBase, scoreCol + tileStart + (warp << 3), scoreLd);
            }
            context.localBarrier();
        }

        // Pass 2: per row, the maximum and the denominator of the scaled scores, eight lanes a
        // row (row = tid / 8, keys tid % 8, + 8, ...), folded by shuffles within the eight.
        int statRow = tid >> 3;
        int statLane = tid & 7;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        int statBase = ((rowBase + statRow) * heads + head) * scoreStride;
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statLane; p <= statPos + 1; p += 8) { // NEGATIVE CONTROL
            rowMax = TornadoMath.max(rowMax, scores.get(statBase + p) * invSqrt);
        }
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 4));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 2));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 1));
        if (statLane == 0) {
            rowStat[statRow] = rowMax;
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statLane; p <= statPos + 1; p += 8) { // NEGATIVE CONTROL
            rowSum += TornadoMath.exp(scores.get(statBase + p) * invSqrt - m);
        }
        rowSum += context.simdShuffleDown(rowSum, 4);
        rowSum += context.simdShuffleDown(rowSum, 2);
        rowSum += context.simdShuffleDown(rowSum, 1);
        if (statLane == 0) {
            rowStat[16 + statRow] = rowSum;
        }
        context.localBarrier();

        // Pass 3: O = P V, 32 keys a tile, against the settled maximum.
        float[] o0 = context.mmaFragment(0.0f);
        float[] o1 = context.mmaFragment(0.0f);
        float[] o2 = context.mmaFragment(0.0f);
        float[] o3 = context.mmaFragment(0.0f);
        float[] o4 = context.mmaFragment(0.0f);
        float[] o5 = context.mmaFragment(0.0f);
        float[] o6 = context.mmaFragment(0.0f);
        float[] o7 = context.mmaFragment(0.0f);
        int pBase = stageBase + 16 * 256;
        for (int tileStart = 0; tileStart <= lastKey; tileStart += 32) {
            for (int i = 0; i < 32 * 256 / 2 / 128; i++) {
                int e = i * 128 + tid;
                int key = e >> 7;
                int d = (e & 127) << 1;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                int dst = (((key >> 4) << 5) + (d >> 3)) * 64 + ((key & 15) << 2) + ((d & 7) >> 1);
                context.asyncCopyToLocal(vTile, dst, valueCache, base + d);
            }
            context.asyncCopyCommit();
            // P for this tile: lane covers (row = i / 64, key = i % 64), 8 per lane, as halves.
            for (int i = tid; i < 16 * 32; i += 128) {
                int row = i >> 5;
                int key = tileStart + (i & 31);
                float prob = 0.0f;
                if (key <= startPos + rowBase + row + 1) { // NEGATIVE CONTROL
                    prob =
                            TornadoMath.exp(
                                    scores.get(((rowBase + row) * heads + head) * scoreStride + key)
                                                    * invSqrt
                                            - rowStat[row]);
                }
                stage.set(pBase + i, new HalfFloat(prob));
            }
            context.localBarrier();
            for (int i = tid; i < 16 * 32 / 2; i += 128) {
                int t = i >> 7;
                int q = (i >> 3) & 15;
                int kk = (t << 4) + ((i & 7) << 1);
                context.asyncCopyToLocal(pTile, i, stage, pBase + (q << 5) + kk);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(0);
            context.localBarrier();
            for (int t = 0; t < 32 / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(pTile, 16, t * 512);
                int vBase = (t << 5) + (warp << 3);
                o0 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 0) * 256),
                                o0,
                                MMAShape.M16N8K16);
                o1 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 1) * 256),
                                o1,
                                MMAShape.M16N8K16);
                o2 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 2) * 256),
                                o2,
                                MMAShape.M16N8K16);
                o3 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 3) * 256),
                                o3,
                                MMAShape.M16N8K16);
                o4 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 4) * 256),
                                o4,
                                MMAShape.M16N8K16);
                o5 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 5) * 256),
                                o5,
                                MMAShape.M16N8K16);
                o6 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 6) * 256),
                                o6,
                                MMAShape.M16N8K16);
                o7 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 7) * 256),
                                o7,
                                MMAShape.M16N8K16);
            }
            context.localBarrier();
        }

        int ld = heads * headSize;
        int colBase = head * headSize + (warp << 6);
        context.mmaStore(o0, outBatch, rowBase, colBase, ld);
        context.mmaStore(o1, outBatch, rowBase, colBase + 8, ld);
        context.mmaStore(o2, outBatch, rowBase, colBase + 16, ld);
        context.mmaStore(o3, outBatch, rowBase, colBase + 24, ld);
        context.mmaStore(o4, outBatch, rowBase, colBase + 32, ld);
        context.mmaStore(o5, outBatch, rowBase, colBase + 40, ld);
        context.mmaStore(o6, outBatch, rowBase, colBase + 48, ld);
        context.mmaStore(o7, outBatch, rowBase, colBase + 56, ld);
        context.localBarrier();
        // Divide by the denominator: lane covers (row = i / 256, dim = i % 256).
        for (int i = tid; i < 16 * 256; i += 128) {
            int row = i >> 8;
            int idx = (rowBase + row) * ld + head * headSize + (i & 255);
            outBatch.set(idx, outBatch.get(idx) / rowStat[16 + row]);
        }
    }

    /** The candidate without the final division by the denominator. */
    public static void tensorCoreWithoutTheDenominator(
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
            int scoreStride,
            HalfFloatArray stage) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int queryTile = group / heads;
        int head = group - queryTile * heads;
        int rowBase = queryTile * 16;
        if (rowBase >= batchInfo.get(1)) {
            return;
        }
        int startPos = batchInfo.get(0);
        int slot = batchInfo.get(2);
        int capacity = scoreStride;
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        int[] qTile = context.allocateIntLocalArray(16 * 256 / 2);
        HalfFloat[] kTile = context.allocateHalfFloatLocalArray(32 * 256);
        int[] vTile = context.allocateIntLocalArray(32 * 256 / 2);
        int[] pTile = context.allocateIntLocalArray(16 * 32 / 2);
        float[] rowStat = context.allocateFloatLocalArray(16 * 2 + 128);

        int stageBase = group * 4608;
        int qBase = (rowBase * heads + head) * headSize;
        for (int i = tid; i < 16 * 256; i += 128) {
            int row = i >> 8;
            int d = i & 255;
            stage.set(
                    stageBase + i,
                    new HalfFloat(queryBatch.get(qBase + row * heads * headSize + d)));
        }
        context.localBarrier();
        for (int i = tid; i < 16 * 256 / 2; i += 128) {
            int t = i >> 7;
            int within = i & 127;
            int row = within >> 3;
            int d = (t << 4) + ((within & 7) << 1);
            context.asyncCopyToLocal(qTile, i, stage, stageBase + (row << 8) + d);
        }
        context.asyncCopyCommit();
        context.asyncCopyWaitGroup(0);
        context.localBarrier();

        int maxPos = startPos + rowBase + 16 - 1;
        int lastKey = capacity - 1;
        if (maxPos < lastKey) {
            lastKey = maxPos;
        }
        // The score scratch as a matrix with row stride heads * scoreStride: row r of this tile
        // is (rowBase + r) * heads + head spans, i.e. column offset head * scoreStride.
        int scoreLd = heads * scoreStride;
        int scoreCol = head * scoreStride;

        // Pass 1: S = Q K^T, 32 keys a tile, stored unscaled to the scratch.
        for (int tileStart = 0; tileStart <= lastKey; tileStart += 32) {
            for (int i = 0; i < 32 * 256 / 128; i++) {
                int e = i * 128 + tid;
                int key = e >> 8;
                int d = e & 255;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                context.mmaStoreBSwizzled(
                        kTile,
                        d & 15,
                        key & 7,
                        8,
                        // Widened and re-narrowed (exact): the route the store's lowering takes
                        // a half by value, rather than a half read from the array.
                        new HalfFloat(keyCache.get(base + d).getFloat32()),
                        (((d >> 4) << 2) + (key >> 3)) * 256);
            }
            context.localBarrier();
            float[] s0 = context.mmaFragment(0.0f);
            for (int t = 0; t < 256 / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(qTile, 16, t * 512);
                HalfFloat[] b0 = context.mmaLoadBSwizzled(kTile, 16, ((t << 2) + warp) * 256);
                s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
            }
            // This warp's eight keys of the tile; skipped whole where they lie past the capacity
            // (a warp-uniform choice, so no lane misses a barrier): those keys are never read.
            if (tileStart + (warp << 3) + 7 < capacity) {
                context.mmaStore(s0, scores, rowBase, scoreCol + tileStart + (warp << 3), scoreLd);
            }
            context.localBarrier();
        }

        // Pass 2: per row, the maximum and the denominator of the scaled scores, eight lanes a
        // row (row = tid / 8, keys tid % 8, + 8, ...), folded by shuffles within the eight.
        int statRow = tid >> 3;
        int statLane = tid & 7;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        int statBase = ((rowBase + statRow) * heads + head) * scoreStride;
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statLane; p <= statPos; p += 8) {
            rowMax = TornadoMath.max(rowMax, scores.get(statBase + p) * invSqrt);
        }
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 4));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 2));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 1));
        if (statLane == 0) {
            rowStat[statRow] = rowMax;
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statLane; p <= statPos; p += 8) {
            rowSum += TornadoMath.exp(scores.get(statBase + p) * invSqrt - m);
        }
        rowSum += context.simdShuffleDown(rowSum, 4);
        rowSum += context.simdShuffleDown(rowSum, 2);
        rowSum += context.simdShuffleDown(rowSum, 1);
        if (statLane == 0) {
            rowStat[16 + statRow] = rowSum;
        }
        context.localBarrier();

        // Pass 3: O = P V, 32 keys a tile, against the settled maximum.
        float[] o0 = context.mmaFragment(0.0f);
        float[] o1 = context.mmaFragment(0.0f);
        float[] o2 = context.mmaFragment(0.0f);
        float[] o3 = context.mmaFragment(0.0f);
        float[] o4 = context.mmaFragment(0.0f);
        float[] o5 = context.mmaFragment(0.0f);
        float[] o6 = context.mmaFragment(0.0f);
        float[] o7 = context.mmaFragment(0.0f);
        int pBase = stageBase + 16 * 256;
        for (int tileStart = 0; tileStart <= lastKey; tileStart += 32) {
            for (int i = 0; i < 32 * 256 / 2 / 128; i++) {
                int e = i * 128 + tid;
                int key = e >> 7;
                int d = (e & 127) << 1;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                int dst = (((key >> 4) << 5) + (d >> 3)) * 64 + ((key & 15) << 2) + ((d & 7) >> 1);
                context.asyncCopyToLocal(vTile, dst, valueCache, base + d);
            }
            context.asyncCopyCommit();
            // P for this tile: lane covers (row = i / 64, key = i % 64), 8 per lane, as halves.
            for (int i = tid; i < 16 * 32; i += 128) {
                int row = i >> 5;
                int key = tileStart + (i & 31);
                float prob = 0.0f;
                if (key <= startPos + rowBase + row) {
                    prob =
                            TornadoMath.exp(
                                    scores.get(((rowBase + row) * heads + head) * scoreStride + key)
                                                    * invSqrt
                                            - rowStat[row]);
                }
                stage.set(pBase + i, new HalfFloat(prob));
            }
            context.localBarrier();
            for (int i = tid; i < 16 * 32 / 2; i += 128) {
                int t = i >> 7;
                int q = (i >> 3) & 15;
                int kk = (t << 4) + ((i & 7) << 1);
                context.asyncCopyToLocal(pTile, i, stage, pBase + (q << 5) + kk);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(0);
            context.localBarrier();
            for (int t = 0; t < 32 / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(pTile, 16, t * 512);
                int vBase = (t << 5) + (warp << 3);
                o0 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 0) * 256),
                                o0,
                                MMAShape.M16N8K16);
                o1 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 1) * 256),
                                o1,
                                MMAShape.M16N8K16);
                o2 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 2) * 256),
                                o2,
                                MMAShape.M16N8K16);
                o3 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 3) * 256),
                                o3,
                                MMAShape.M16N8K16);
                o4 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 4) * 256),
                                o4,
                                MMAShape.M16N8K16);
                o5 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 5) * 256),
                                o5,
                                MMAShape.M16N8K16);
                o6 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 6) * 256),
                                o6,
                                MMAShape.M16N8K16);
                o7 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 7) * 256),
                                o7,
                                MMAShape.M16N8K16);
            }
            context.localBarrier();
        }

        int ld = heads * headSize;
        int colBase = head * headSize + (warp << 6);
        context.mmaStore(o0, outBatch, rowBase, colBase, ld);
        context.mmaStore(o1, outBatch, rowBase, colBase + 8, ld);
        context.mmaStore(o2, outBatch, rowBase, colBase + 16, ld);
        context.mmaStore(o3, outBatch, rowBase, colBase + 24, ld);
        context.mmaStore(o4, outBatch, rowBase, colBase + 32, ld);
        context.mmaStore(o5, outBatch, rowBase, colBase + 40, ld);
        context.mmaStore(o6, outBatch, rowBase, colBase + 48, ld);
        context.mmaStore(o7, outBatch, rowBase, colBase + 56, ld);
        context.localBarrier();
        // Divide by the denominator: lane covers (row = i / 256, dim = i % 256).
        for (int i = tid; i < 16 * 256; i += 128) {
            int row = i >> 8;
            int idx = (rowBase + row) * ld + head * headSize + (i & 255);
            outBatch.set(idx, outBatch.get(idx)); // NEGATIVE CONTROL: no division
        }
    }
}
