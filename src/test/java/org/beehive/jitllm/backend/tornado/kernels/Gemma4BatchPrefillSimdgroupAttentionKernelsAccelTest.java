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
 * The {@link Gemma4BatchPrefillSimdgroupAttentionKernels} (256- and 512-wide heads) against a
 * double-precision host reference of unscaled, causal, sliding-window attention: a chunk that
 * starts mid-sequence, a last query tile that is only partly active, grouped-query heads, a window
 * shorter than the sequence and one longer. Cache rows past the chunk's last position and query
 * rows past the active ones hold NaN, so reading either into the result fails the test.
 */
// @formatter:on
public class Gemma4BatchPrefillSimdgroupAttentionKernelsAccelTest {

    private static final float UNTOUCHED = -12344.0f; // exact in FP16

    private static void check(int head, int start, int active, int window) {
        assumeTrue("SIMD-group matrices are Metal only", BatchPrefillGemmPolicy.tiled());
        int nHeads = 4;
        int kvMul = 4;
        int kvDim = (nHeads / kvMul) * head;
        int qDim = nHeads * head;
        int stride = qDim + 2 * kvDim;
        int context = 256;
        int base = 3 * kvDim;
        int lastValid = start + active - 1;
        int tile = Gemma4BatchPrefillSimdgroupAttentionKernels.QUERY_TILE;
        int rows = ((active + tile - 1) / tile) * tile;

        Random rnd = new Random(31L * head + start + active + window);
        FloatArray keys = new FloatArray(base + context * kvDim + kvDim);
        FloatArray values = new FloatArray(base + context * kvDim + kvDim);
        keys.init(Float.NaN);
        values.init(Float.NaN);
        double[][] k = new double[lastValid + 1][kvDim];
        double[][] v = new double[lastValid + 1][kvDim];
        for (int p = 0; p <= lastValid; p++) {
            for (int i = 0; i < kvDim; i++) {
                float kk = 0.25f * (rnd.nextFloat() - 0.5f);
                float vv = rnd.nextFloat() - 0.5f;
                k[p][i] = kk;
                v[p][i] = vv;
                keys.set(base + p * kvDim + i, kk);
                values.set(base + p * kvDim + i, vv);
            }
        }
        FloatArray qkv = new FloatArray(rows * stride);
        qkv.init(Float.NaN);
        for (int t = 0; t < active; t++) {
            for (int i = 0; i < qDim; i++) {
                qkv.set(t * stride + i, rnd.nextFloat() - 0.5f);
            }
        }
        HalfFloatArray out = new HalfFloatArray(rows * qDim);
        out.init(new HalfFloat(UNTOUCHED));
        IntArray holder = new IntArray(3);
        holder.set(0, start);
        holder.set(1, active);

        String name = "gemma4Attention" + head + "x" + start + "x" + active + "x" + window;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, holder, qkv, keys, values, out);
        KernelContext ctx = new KernelContext();
        if (head == 256) {
            graph.task(
                    "t",
                    Gemma4BatchPrefillSimdgroupAttentionKernels
                            ::batchedSlidingWindowAttentionHead256,
                    ctx,
                    holder,
                    qkv,
                    keys,
                    values,
                    out,
                    nHeads,
                    head,
                    kvDim,
                    kvMul,
                    stride,
                    base,
                    window);
        } else {
            graph.task(
                    "t",
                    Gemma4BatchPrefillSimdgroupAttentionKernels
                            ::batchedSlidingWindowAttentionHead512,
                    ctx,
                    holder,
                    qkv,
                    keys,
                    values,
                    out,
                    nHeads,
                    head,
                    kvDim,
                    kvMul,
                    stride,
                    base,
                    window);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        int threads = Gemma4BatchPrefillSimdgroupAttentionKernels.THREADS;
        WorkerGrid1D worker = new WorkerGrid1D((rows / tile) * nHeads * threads);
        worker.setLocalWork(threads, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".t", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

        for (int t = 0; t < rows; t++) {
            for (int h = 0; h < nHeads; h++) {
                if (t >= active) {
                    for (int d = 0; d < head; d++) {
                        assertEquals(
                                "row " + t + " is past the chunk and must stay untouched",
                                UNTOUCHED,
                                out.get(t * qDim + h * head + d).getFloat32(),
                                0.0f);
                    }
                    continue;
                }
                int pos = start + t;
                int lo = Math.max(0, pos - window + 1);
                int kvOff = (h / kvMul) * head;
                double[] s = new double[pos + 1];
                double max = Double.NEGATIVE_INFINITY;
                for (int p = lo; p <= pos; p++) {
                    double dot = 0.0;
                    for (int d = 0; d < head; d++) {
                        dot += qkv.get(t * stride + h * head + d) * k[p][kvOff + d];
                    }
                    s[p] = dot;
                    max = Math.max(max, dot);
                }
                double sum = 0.0;
                for (int p = lo; p <= pos; p++) {
                    s[p] = Math.exp(s[p] - max);
                    sum += s[p];
                }
                for (int d = 0; d < head; d++) {
                    double expected = 0.0;
                    for (int p = lo; p <= pos; p++) {
                        expected += s[p] * v[p][kvOff + d];
                    }
                    expected /= sum;
                    assertEquals(
                            name + " token " + t + " head " + h + " dim " + d,
                            expected,
                            out.get(t * qDim + h * head + d).getFloat32(),
                            2e-3 * Math.max(1.0, Math.abs(expected)));
                }
            }
        }
    }

    @Test
    public void causalFromTheStart() {
        for (int head : new int[] {256, 512}) {
            check(head, 0, 40, 1 << 20);
        }
    }

    @Test
    public void slidingWindowMidSequenceWithAPartialTile() {
        for (int head : new int[] {256, 512}) {
            check(head, 37, 45, 24);
        }
    }

    @Test
    public void singleTokenMidSequence() {
        for (int head : new int[] {256, 512}) {
            check(head, 70, 1, 64);
        }
    }
}
