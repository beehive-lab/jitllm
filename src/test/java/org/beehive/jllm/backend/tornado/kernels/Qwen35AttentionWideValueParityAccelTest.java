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
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The wide-value-tile batched attention against the staged kernel it was copied from, on the
 * device, over identical inputs: the stored causal scores (unchanged by construction, and checked)
 * and the final outputs, raw-bit equal, over NaN-poisoned scratch and outputs.
 *
 * <p>Only the value pass changes — 128 positions per tile instead of 16 — while each output element
 * still accumulates its positions in increasing order, so the bits must not move. Cases cover
 * ranges around the 16- and 128-position tile boundaries (ends at 15, 16, 17, 127, 128, 129 and
 * beyond), short ranges, non-zero starts mid-page, shuffled physical pages, a second layer and a
 * non-zero slot, partial chunks, width 64, and ranges through position 2047.
 */
// @formatter:on
public class Qwen35AttentionWideValueParityAccelTest {

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

    /** One key/value store with a shuffled page map, shared by every case. */
    private static final class Store {
        final HalfFloatArray keys;
        final HalfFloatArray values;
        final IntArray blockTable;

        Store(long seed) {
            Random rng = new Random(seed);
            int physicalBlocks = SLOTS * BLOCKS_PER_SLOT + 3;
            int elements = physicalBlocks * BLOCK_STRIDE;
            keys = new HalfFloatArray(elements);
            values = new HalfFloatArray(elements);
            for (int i = 0; i < elements; i++) {
                keys.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
                values.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
            }
            // Every slot's logical pages map to distinct physical pages, in shuffled order, so
            // consecutive positions cross into non-adjacent storage at every page boundary.
            Integer[] pages = new Integer[physicalBlocks];
            for (int i = 0; i < physicalBlocks; i++) {
                pages[i] = i;
            }
            java.util.Collections.shuffle(Arrays.asList(pages), rng);
            blockTable = new IntArray(SLOTS * BLOCKS_PER_SLOT);
            for (int i = 0; i < SLOTS * BLOCKS_PER_SLOT; i++) {
                blockTable.set(i, pages[i]);
            }
        }
    }

    private static FloatArray queries(int rows, long seed) {
        Random rng = new Random(seed);
        FloatArray q = new FloatArray(rows * HEADS * HEAD_SIZE);
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, rng.nextFloat() * 2.0f - 1.0f);
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

    private static void assertBitParity(
            String what, Store store, int rows, int startPos, int activeRows) throws Exception {
        assertTrue(what + ": range exceeds the context", startPos + activeRows <= CAPACITY);
        FloatArray q = queries(rows, 7L + startPos + rows);
        IntArray info = batchInfo(startPos, activeRows);
        FloatArray original = new FloatArray(rows * HEADS * HEAD_SIZE);
        FloatArray candidate = new FloatArray(rows * HEADS * HEAD_SIZE);
        FloatArray scoresOriginal = new FloatArray(rows * HEADS * CAPACITY);
        FloatArray scores = new FloatArray(rows * HEADS * CAPACITY);
        original.init(Float.NaN);
        candidate.init(Float.NaN);
        scoresOriginal.init(Float.NaN);
        scores.init(Float.NaN);

        TaskGraph graph =
                new TaskGraph("attn")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                info,
                                q,
                                store.keys,
                                store.values,
                                store.blockTable,
                                original,
                                candidate,
                                scoresOriginal,
                                scores)
                        .task(
                                "original",
                                Qwen35ReferenceKernels::attentionBatchFP16PagedScoredStaged,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                original,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                store.blockTable,
                                BLOCK_CFG,
                                BLOCK_STRIDE,
                                LOCAL,
                                scoresOriginal,
                                CAPACITY)
                        .task(
                                "wide",
                                Qwen35BatchKernels::attentionBatchFP16PagedScoredStagedWide,
                                new KernelContext(),
                                info,
                                q,
                                store.keys,
                                store.values,
                                candidate,
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
                                CAPACITY)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION,
                                original,
                                candidate,
                                scoresOriginal,
                                scores);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("attn.original", grid(rows));
        scheduler.addWorkerGrid("attn.wide", grid(rows));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        // The stored causal scores: every (row, head, position <= row's position) of the active
        // rows, raw-bit equal and finite.
        int scoreMismatches = 0;
        String firstScore = null;
        for (int row = 0; row < activeRows; row++) {
            int position = startPos + row;
            for (int head = 0; head < HEADS; head++) {
                int base = (row * HEADS + head) * CAPACITY;
                for (int p = 0; p <= position; p++) {
                    float o = scoresOriginal.get(base + p);
                    float c = scores.get(base + p);
                    assertTrue(
                            what
                                    + ": original score not finite at row "
                                    + row
                                    + " head "
                                    + head
                                    + " p "
                                    + p,
                            Float.isFinite(o));
                    assertTrue(
                            what
                                    + ": candidate score not finite at row "
                                    + row
                                    + " head "
                                    + head
                                    + " p "
                                    + p,
                            Float.isFinite(c));
                    if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(c)) {
                        if (firstScore == null) {
                            firstScore =
                                    "row " + row + " head " + head + " p " + p + ": " + o + " vs "
                                            + c;
                        }
                        scoreMismatches++;
                    }
                }
            }
        }
        assertEquals(
                what + ": " + scoreMismatches + " scores differ, first at " + firstScore,
                0,
                scoreMismatches);

        int mismatches = 0;
        String first = null;
        int stride = HEADS * HEAD_SIZE;
        for (int i = 0; i < rows * stride; i++) {
            int row = i / stride;
            float o = original.get(i);
            float c = candidate.get(i);
            if (row < activeRows) {
                assertTrue(what + ": original not finite at " + where(i), Float.isFinite(o));
                assertTrue(what + ": candidate not finite at " + where(i), Float.isFinite(c));
            }
            if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(c)) {
                if (first == null) {
                    first = where(i) + ": original " + o + " candidate " + c;
                }
                mismatches++;
            }
        }
        assertEquals(
                what + ": " + mismatches + " outputs differ, first at " + first, 0, mismatches);
    }

    private static String where(int i) {
        int stride = HEADS * HEAD_SIZE;
        return "row "
                + (i / stride)
                + " head "
                + (i % stride / HEAD_SIZE)
                + " d "
                + (i % HEAD_SIZE);
    }

    /** Position zero onward: the shortest causal ranges, one to thirty-two keys. */
    @Test
    public void theShortestRangesAgreeBitForBit() throws Exception {
        assertBitParity("start 0", new Store(1L), 32, 0, 32);
    }

    /** Ranges ending at 0..31 (the 16-boundary), 100..131 (the 128-boundary) and 255..286. */
    @Test
    public void theTileBoundariesAgreeBitForBit() throws Exception {
        assertBitParity("start 100", new Store(7L), 32, 100, 32);
        assertBitParity("start 255", new Store(8L), 32, 255, 32);
        assertBitParity("start 112", new Store(9L), 32, 112, 32);
    }

    /** A chunk that starts mid-page and crosses page boundaries within its own rows. */
    @Test
    public void aChunkStartingMidPageAgreesBitForBit() throws Exception {
        assertBitParity("start 15", new Store(2L), 32, 15, 32);
    }

    /** The longest ranges of a 2048 context, reaching its last position. */
    @Test
    public void theLongestRangesAgreeBitForBit() throws Exception {
        assertBitParity("start 2016", new Store(3L), 32, 2016, 32);
    }

    /** A partial chunk: only five of thirty-two rows are real. */
    @Test
    public void aPartialChunkAgreesBitForBit() throws Exception {
        assertBitParity("partial", new Store(4L), 32, 1000, 5);
    }

    /** Width 64, at a long range and mid-page. */
    @Test
    public void width64AgreesBitForBit() throws Exception {
        assertBitParity("w64 start 1975", new Store(5L), 64, 1975, 64);
        assertBitParity("w64 start 0", new Store(6L), 64, 0, 64);
    }

    /**
     * Kernel time of both forms at several context depths and both widths, from the TornadoVM
     * profiler, alternating order after a warm-up. Prints distributions only.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        Store store = new Store(9L);
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
            FloatArray q = queries(rows, 99L);
            IntArray info = batchInfo(startPos, rows);
            FloatArray outA = new FloatArray(rows * HEADS * HEAD_SIZE);
            FloatArray outB = new FloatArray(rows * HEADS * HEAD_SIZE);
            FloatArray scores = new FloatArray(rows * HEADS * CAPACITY);
            FloatArray scoresA = new FloatArray(rows * HEADS * CAPACITY);
            TaskGraph a =
                    new TaskGraph("orig")
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
                                    Qwen35ReferenceKernels::attentionBatchFP16PagedScoredStaged,
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
                    new TaskGraph("scored")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    store.blockTable,
                                    scores)
                            .task(
                                    "p",
                                    Qwen35BatchKernels::attentionBatchFP16PagedScoredStagedWide,
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
                                    scores,
                                    CAPACITY)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("orig.p", grid(rows));
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("scored.p", grid(rows));
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
                report("staged rows=" + rows + " start=" + startPos, tA);
                report("wide   rows=" + rows + " start=" + startPos, tB);
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
}
