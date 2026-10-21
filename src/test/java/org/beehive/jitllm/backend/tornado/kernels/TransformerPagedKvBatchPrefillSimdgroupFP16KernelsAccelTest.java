package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@link TransformerPagedKvBatchPrefillSimdgroupFP16Kernels} attention kernels (64-, 96- and
 * 128-wide heads, unscaled and scaled) against a double-precision host reference of causal
 * attention over a paged FP16 cache: shuffled physical blocks, a batch that starts mid-sequence, a
 * last query tile that is only partly active and a padding tile, and grouped-query heads. The
 * reference reads the cache values back as stored, so the FP16 rounding is common to both and the
 * bound is the FP32 kernels' one. Cache rows past the batch's last position hold FP16 NaN and query
 * rows past the active ones FP32 NaN, so reading either into the result fails the test.
 */
// @formatter:on
public class TransformerPagedKvBatchPrefillSimdgroupFP16KernelsAccelTest {

    private static final int BLOCK = 16;
    private static final float UNTOUCHED = -12345.5f;

    /** Granite-like attention multiplier for the scaled kernels. */
    private static final float SCALE = 0.0078125f;

    private static void check(int head, int start, int active, int gridRows, boolean scaled) {
        assumeTrue("SIMD-group matrices are Metal only", BatchPrefillGemmPolicy.tiled());
        int nHeads = 4;
        int kvMul = 2;
        int kvDim = (nHeads / kvMul) * head;
        int dim = nHeads * head;
        int layers = 2;
        int layerIndex = 1;
        int slot = 1;
        int blocksPerSlot = 8;
        int physBlocks = 2 * blocksPerSlot;
        int blockStride = layers * BLOCK * kvDim;
        int blockCfg = BLOCK | (blocksPerSlot << 16);
        int lastValid = start + active - 1;

        IntArray blockTable = new IntArray(2 * blocksPerSlot);
        for (int b = 0; b < blocksPerSlot; b++) {
            blockTable.set(slot * blocksPerSlot + b, (b * 5 + 3) % physBlocks);
            blockTable.set(b, (b * 5 + 4) % physBlocks);
        }
        Random rnd = new Random(31L * start + active + (scaled ? 7 : 0));
        HalfFloatArray keys = new HalfFloatArray(physBlocks * blockStride);
        HalfFloatArray values = new HalfFloatArray(physBlocks * blockStride);
        keys.init(new HalfFloat(Float.NaN));
        values.init(new HalfFloat(Float.NaN));
        float[][] k = new float[lastValid + 1][kvDim];
        float[][] v = new float[lastValid + 1][kvDim];
        for (int p = 0; p <= lastValid; p++) {
            int base =
                    blockTable.get(slot * blocksPerSlot + p / BLOCK) * blockStride
                            + layerIndex * BLOCK * kvDim
                            + (p % BLOCK) * kvDim;
            for (int i = 0; i < kvDim; i++) {
                HalfFloat kh = new HalfFloat(rnd.nextFloat() - 0.5f);
                HalfFloat vh = new HalfFloat(rnd.nextFloat() - 0.5f);
                keys.set(base + i, kh);
                values.set(base + i, vh);
                k[p][i] = kh.getFloat32();
                v[p][i] = vh.getFloat32();
            }
        }
        FloatArray q = new FloatArray(gridRows * dim);
        q.init(Float.NaN);
        for (int t = 0; t < active; t++) {
            for (int i = 0; i < dim; i++) {
                q.set(t * dim + i, (scaled ? 16.0f : 2.0f) * (rnd.nextFloat() - 0.5f));
            }
        }
        FloatArray out = new FloatArray(gridRows * dim);
        out.init(UNTOUCHED);
        IntArray holder = new IntArray(3);
        holder.set(0, start);
        holder.set(1, active);
        holder.set(2, slot);

        String name =
                "simdgroupAttentionFP16"
                        + (scaled ? "Scaled" : "")
                        + head
                        + "x"
                        + start
                        + "x"
                        + active;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                holder,
                                q,
                                keys,
                                values,
                                out,
                                blockTable);
        KernelContext context = new KernelContext();
        if (scaled) {
            graph.task(
                    "t",
                    switch (head) {
                        case 64 ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedScaledHead64;
                        case 96 ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedScaledHead96;
                        default ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedScaledHead128;
                    },
                    context,
                    holder,
                    q,
                    keys,
                    values,
                    out,
                    nHeads,
                    head,
                    kvDim,
                    kvMul,
                    layerIndex,
                    blockTable,
                    blockCfg,
                    blockStride,
                    dim,
                    SCALE);
        } else {
            graph.task(
                    "t",
                    switch (head) {
                        case 64 ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedHead64;
                        case 96 ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedHead96;
                        default ->
                                TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                        ::batchedFlashAttentionKVFP16PagedHead128;
                    },
                    context,
                    holder,
                    q,
                    keys,
                    values,
                    out,
                    nHeads,
                    head,
                    kvDim,
                    kvMul,
                    layerIndex,
                    blockTable,
                    blockCfg,
                    blockStride,
                    dim);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        int tiles = (gridRows + 31) / 32;
        int threads = TransformerPagedKvBatchPrefillSimdgroupFP16Kernels.THREADS;
        WorkerGrid1D worker = new WorkerGrid1D(tiles * nHeads * threads);
        worker.setLocalWork(threads, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".t", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

        double scale = scaled ? SCALE : 1.0 / Math.sqrt(head);
        for (int t = 0; t < gridRows; t++) {
            for (int h = 0; h < nHeads; h++) {
                if (t >= active) {
                    for (int d = 0; d < head; d++) {
                        assertEquals(
                                "row " + t + " is past the batch and must stay untouched",
                                UNTOUCHED,
                                out.get(t * dim + h * head + d),
                                0.0f);
                    }
                    continue;
                }
                int pos = start + t;
                int kvOff = (h / kvMul) * head;
                double[] s = new double[pos + 1];
                double max = Double.NEGATIVE_INFINITY;
                for (int p = 0; p <= pos; p++) {
                    double dot = 0.0;
                    for (int d = 0; d < head; d++) {
                        dot += (double) q.get(t * dim + h * head + d) * k[p][kvOff + d];
                    }
                    s[p] = dot * scale;
                    max = Math.max(max, s[p]);
                }
                double sum = 0.0;
                for (int p = 0; p <= pos; p++) {
                    s[p] = Math.exp(s[p] - max);
                    sum += s[p];
                }
                for (int d = 0; d < head; d++) {
                    double expected = 0.0;
                    for (int p = 0; p <= pos; p++) {
                        expected += s[p] * v[p][kvOff + d];
                    }
                    expected /= sum;
                    assertEquals(
                            name + " token " + t + " head " + h + " dim " + d,
                            expected,
                            out.get(t * dim + h * head + d),
                            2e-5);
                }
            }
        }
    }

    @Test
    public void fromTheStartOfTheSequenceWithWholeTiles() {
        for (int head : new int[] {64, 96, 128}) {
            check(head, 0, 64, 64, false);
        }
    }

    @Test
    public void midSequenceWithAPartialAndAPaddingTile() {
        for (int head : new int[] {64, 96, 128}) {
            check(head, 37, 45, 96, false);
        }
    }

    @Test
    public void midSequenceWithASingleToken() {
        for (int head : new int[] {64, 96, 128}) {
            check(head, 70, 1, 32, false);
        }
    }

    @Test
    public void scaledMidSequenceWithAPartialAndAPaddingTile() {
        for (int head : new int[] {64, 96, 128}) {
            check(head, 37, 45, 96, true);
        }
    }
}
