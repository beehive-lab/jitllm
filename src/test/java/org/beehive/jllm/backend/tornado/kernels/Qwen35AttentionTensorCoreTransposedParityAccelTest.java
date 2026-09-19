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
 * The transposed-score tensor-core attention kernels — {@code attentionBatchFP16PagedTensorCoreT}
 * (16 queries, four warps) and {@code attentionBatchFP16PagedTensorCoreT32} (32 queries, eight
 * warps) — against the retained {@code attentionBatchFP16PagedTensorCore}: raw-bit equal outputs
 * over every row of the chunk (padding rows included, which all three write) on the production
 * geometry (24 heads over 4 key/value heads, 256-wide) with shuffled pages, at capacities 2048 and
 * 2056, chunk starts 0, 512 and 2016, whole and partial chunks, NaN-poisoned outputs; the score
 * scratch sized to whole key tiles as the state sizes it. An opt-in screen times the three at 2048
 * rows.
 */
// @formatter:on
public class Qwen35AttentionTensorCoreTransposedParityAccelTest {

    private static final int HEADS = 24;
    private static final int KV_HEADS = 4;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int HEAD_SIZE = 256;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int BLOCK_SIZE = 16;
    private static final int KV_LAYERS = 2;
    private static final int LAYER = 1;
    private static final int SLOT = 1;
    private static final int SLOTS = 2;

    private static final class Store {
        final int capacity;
        final int blocksPerSlot;
        final int blockCfg;
        final int blockStride;
        final HalfFloatArray keys;
        final HalfFloatArray values;
        final IntArray blockTable;

        Store(int capacity, long seed) {
            this.capacity = capacity;
            blocksPerSlot = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blockCfg = BLOCK_SIZE | (blocksPerSlot << 16);
            blockStride = KV_LAYERS * BLOCK_SIZE * KV_DIM;
            int physical = SLOTS * blocksPerSlot + 3;
            Random rng = new Random(seed);
            keys = new HalfFloatArray(physical * blockStride);
            values = new HalfFloatArray(physical * blockStride);
            for (int i = 0; i < keys.getSize(); i++) {
                keys.set(i, new HalfFloat(rng.nextFloat() * 2 - 1));
                values.set(i, new HalfFloat(rng.nextFloat() * 2 - 1));
            }
            Integer[] pages = new Integer[physical];
            for (int i = 0; i < physical; i++) {
                pages[i] = i;
            }
            java.util.Collections.shuffle(Arrays.asList(pages), rng);
            blockTable = new IntArray(SLOTS * blocksPerSlot);
            for (int i = 0; i < blockTable.getSize(); i++) {
                blockTable.set(i, pages[i]);
            }
        }
    }

    enum Kernel {
        RETAINED,
        T16,
        T32
    }

    private static FloatArray run(
            Kernel kernel, Store store, int rows, int active, int startPos, long seed)
            throws Exception {
        FloatArray q = new FloatArray(rows * HEADS * HEAD_SIZE);
        Random rng = new Random(seed);
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, (rng.nextFloat() * 2 - 1) * 3.0f);
        }
        IntArray info = new IntArray(4);
        info.set(0, startPos);
        info.set(1, active);
        info.set(2, SLOT);
        FloatArray out = new FloatArray(rows * HEADS * HEAD_SIZE);
        out.init(Float.NaN);
        FloatArray scores =
                new FloatArray(rows * HEADS * Qwen35BatchKernels.tcScoreKeys(store.capacity));
        HalfFloatArray stage =
                new HalfFloatArray((rows / 16) * HEADS * Qwen35BatchKernels.TC_STAGE_HALVES);
        int local = kernel == Kernel.T32 ? 256 : 128;
        int groups = (rows / (kernel == Kernel.T32 ? 32 : 16)) * HEADS;
        TaskGraph g =
                new TaskGraph("a")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                info,
                                q,
                                store.keys,
                                store.values,
                                store.blockTable,
                                out,
                                scores,
                                stage);
        switch (kernel) {
            case RETAINED ->
                    g.task(
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
                            store.blockCfg,
                            store.blockStride,
                            local,
                            scores,
                            store.capacity,
                            stage);
            case T16 ->
                    g.task(
                            "t",
                            Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT,
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
                            store.blockCfg,
                            store.blockStride,
                            local,
                            scores,
                            store.capacity,
                            stage);
            default ->
                    g.task(
                            "t",
                            Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT32,
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
                            store.blockCfg,
                            store.blockStride,
                            local,
                            scores,
                            store.capacity,
                            stage);
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid grid = new WorkerGrid1D(groups * local);
        grid.setLocalWork(local, 1, 1);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("a.t", grid);
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        return out;
    }

    private static void assertCase(int capacity, int rows, int active, int startPos, long seed)
            throws Exception {
        Store store = new Store(capacity, seed);
        FloatArray control = run(Kernel.RETAINED, store, rows, active, startPos, seed + 1);
        String what =
                "capacity "
                        + capacity
                        + " rows "
                        + rows
                        + " active "
                        + active
                        + " start "
                        + startPos;
        int writtenRows = ((active + 15) / 16) * 16;
        for (Kernel kernel : new Kernel[] {Kernel.T16, Kernel.T32}) {
            if (kernel == Kernel.T32 && rows % 32 != 0) {
                continue;
            }
            FloatArray candidate = run(kernel, store, rows, active, startPos, seed + 1);
            int mismatches = 0;
            String first = null;
            for (int i = 0; i < control.getSize(); i++) {
                int row = i / (HEADS * HEAD_SIZE);
                float c = control.get(i);
                float d = candidate.get(i);
                if (row < writtenRows) {
                    assertTrue(what + ": control not finite at " + i, Float.isFinite(c));
                }
                if (Float.floatToRawIntBits(c) != Float.floatToRawIntBits(d)) {
                    // The 32-query kernel writes the padding rows of its own tile, the retained
                    // kernel those of its 16-query tile; both leave nothing else.
                    if (row >= writtenRows && (Float.isNaN(c) || Float.isNaN(d))) {
                        continue;
                    }
                    if (first == null) {
                        first = kernel + " row " + row + " index " + i + ": " + c + " vs " + d;
                    }
                    mismatches++;
                }
            }
            assertEquals(
                    what + ": " + mismatches + " outputs differ, first " + first, 0, mismatches);
        }
    }

    @Test
    public void wholeChunksAtCapacity2056AreRawBitEqual() throws Exception {
        assertCase(2056, 256, 256, 0, 1L);
        assertCase(2056, 512, 512, 512, 2L);
        assertCase(2056, 32, 32, 2016, 3L);
    }

    @Test
    public void partialChunksAreRawBitEqual() throws Exception {
        assertCase(2056, 256, 200, 0, 4L);
        assertCase(2056, 64, 40, 2000, 5L);
        assertCase(2048, 64, 17, 1024, 6L);
    }

    @Test
    public void capacity2048IsRawBitEqual() throws Exception {
        assertCase(2048, 512, 512, 0, 7L);
        assertCase(2048, 32, 32, 2016, 8L);
    }

    @Test
    public void theFullProductionChunkIsRawBitEqual() throws Exception {
        assertCase(2056, 2048, 2048, 0, 9L);
    }

    /** Retained, T and T32 at 2048 rows, start 0, capacity 2056; opt in with JLLM_KERNEL_SCREEN. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        Store store = new Store(2056, 11L);
        int rows = 2048;
        FloatArray q = new FloatArray(rows * HEADS * HEAD_SIZE);
        Random rng = new Random(12L);
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, rng.nextFloat() * 2 - 1);
        }
        IntArray info = new IntArray(4);
        info.set(1, rows);
        info.set(2, SLOT);
        for (Kernel kernel : Kernel.values()) {
            FloatArray out = new FloatArray(rows * HEADS * HEAD_SIZE);
            FloatArray scores = new FloatArray(rows * HEADS * Qwen35BatchKernels.tcScoreKeys(2056));
            HalfFloatArray stage =
                    new HalfFloatArray((rows / 16) * HEADS * Qwen35BatchKernels.TC_STAGE_HALVES);
            int local = kernel == Kernel.T32 ? 256 : 128;
            int groups = (rows / (kernel == Kernel.T32 ? 32 : 16)) * HEADS;
            TaskGraph g =
                    new TaskGraph("s")
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION,
                                    info,
                                    q,
                                    store.keys,
                                    store.values,
                                    store.blockTable,
                                    out,
                                    scores,
                                    stage);
            switch (kernel) {
                case RETAINED ->
                        g.task(
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
                                store.blockCfg,
                                store.blockStride,
                                local,
                                scores,
                                2056,
                                stage);
                case T16 ->
                        g.task(
                                "t",
                                Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT,
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
                                store.blockCfg,
                                store.blockStride,
                                local,
                                scores,
                                2056,
                                stage);
                default ->
                        g.task(
                                "t",
                                Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT32,
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
                                store.blockCfg,
                                store.blockStride,
                                local,
                                scores,
                                2056,
                                stage);
            }
            g.transferToHost(DataTransferMode.UNDER_DEMAND, out);
            WorkerGrid grid = new WorkerGrid1D(groups * local);
            grid.setLocalWork(local, 1, 1);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("s.t", grid);
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 3; i++) {
                    p.execute();
                }
                long[] t = new long[9];
                for (int i = 0; i < t.length; i++) {
                    t[i] = p.execute().getProfilerResult().getDeviceKernelTime();
                }
                Arrays.sort(t);
                System.out.printf(
                        Locale.ROOT,
                        "[screen] %-8s min %.1f median %.1f max %.1f us%n",
                        kernel,
                        t[0] / 1e3,
                        t[4] / 1e3,
                        t[8] / 1e3);
            }
        }
    }
}
