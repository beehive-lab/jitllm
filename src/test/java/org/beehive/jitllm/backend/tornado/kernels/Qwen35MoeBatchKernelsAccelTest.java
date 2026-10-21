package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The batched {@code qwen35moe} expert path, stage by stage, each against a host evaluation of the
 * device's own inputs to that stage: routing, grouping, the grouped gate/up GEMM and the grouped
 * down GEMM, then the combine.
 */
public class Qwen35MoeBatchKernelsAccelTest {

    private static final int BLOCK_BYTES = 34;

    private static byte[] q8Weights(int rows, int cols, long seed) {
        byte[] raw = new byte[rows * (cols / 32) * BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int b = 0; b < rows * (cols / 32); b++) {
            int bits = 0x2000 | rng.nextInt(0x400) | (rng.nextBoolean() ? 0x8000 : 0);
            raw[b * BLOCK_BYTES] = (byte) (bits & 0xFF);
            raw[b * BLOCK_BYTES + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static ByteArray device(byte[] raw) {
        ByteArray a = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            a.set(i, raw[i]);
        }
        return a;
    }

    private static FloatArray gaussian(int n, long seed, float scale) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, (float) rng.nextGaussian() * scale);
        }
        return a;
    }

    /** {@code W[row] . x} over Q8_0 row {@code row} of width {@code cols}, x as int8 + scales. */
    private static double dot(byte[] w, int row, int cols, ByteArray q, FloatArray s, int xRow) {
        int blocks = cols / 32;
        double sum = 0;
        for (int b = 0; b < blocks; b++) {
            int off = (row * blocks + b) * BLOCK_BYTES;
            float ws =
                    new HalfFloat((short) ((w[off] & 0xFF) | ((w[off + 1] & 0xFF) << 8)))
                            .getFloat32();
            long d = 0;
            for (int i = 0; i < 32; i++) {
                d += (long) w[off + 2 + i] * q.get(xRow * cols + b * 32 + i);
            }
            sum += d * (double) ws * s.get(xRow * blocks + b);
        }
        return sum;
    }

    private static WorkerGrid lanes(int n, int local) {
        WorkerGrid g = new WorkerGrid1D((n + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    @Test
    public void theExpertPathMatchesTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int batch = 128, dim = 256, experts = 64, used = 4, hidden = 128;
        int active = 100;
        int assignments = batch * used;
        FloatArray x = gaussian(batch * dim, 1, 1f);
        FloatArray router = gaussian(experts * dim, 2, 0.1f);
        FloatArray sharedGateInput = gaussian(dim, 3, 0.1f);
        byte[] gateRaw = q8Weights(experts * hidden, dim, 4);
        byte[] upRaw = q8Weights(experts * hidden, dim, 5);
        byte[] downRaw = q8Weights(experts * dim, hidden, 6);
        ByteArray gate = device(gateRaw), up = device(upRaw), down = device(downRaw);

        IntArray batchInfo = new IntArray(3);
        batchInfo.set(1, active);
        ByteArray xq = new ByteArray(batch * dim);
        FloatArray xs = new FloatArray(batch * dim / 32);
        FloatArray logits = new FloatArray(batch * experts);
        FloatArray sharedGate = new FloatArray(batch);
        IntArray ids = new IntArray(assignments);
        ids.init(-7);
        FloatArray weights = new FloatArray(assignments);
        IntArray sortedToken = new IntArray(assignments);
        IntArray position = new IntArray(assignments);
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, experts);
        IntArray tiles = new IntArray(1 + 3 * maxTiles);
        FloatArray hiddenOut = new FloatArray(assignments * hidden);
        ByteArray hq = new ByteArray(assignments * hidden);
        FloatArray hs = new FloatArray(assignments * hidden / 32);
        FloatArray routed = new FloatArray(assignments * dim);
        FloatArray shared = gaussian(batch * dim, 7, 1f);
        FloatArray residual = gaussian(batch * dim, 8, 1f);
        float[] residualBefore = residual.toHeapArray();

        TaskGraph g =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                x,
                                router,
                                sharedGateInput,
                                gate,
                                up,
                                down,
                                batchInfo,
                                xq,
                                xs,
                                logits,
                                sharedGate,
                                ids,
                                weights,
                                sortedToken,
                                position,
                                tiles,
                                hiddenOut,
                                hq,
                                hs,
                                routed,
                                shared,
                                residual)
                        .task(
                                "q",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                x,
                                xq,
                                xs,
                                dim)
                        .task(
                                "r",
                                Qwen35MoeBatchKernels::routerTiled,
                                new KernelContext(),
                                x,
                                router,
                                logits,
                                batchInfo,
                                dim,
                                experts)
                        .task(
                                "sg",
                                Qwen35MoeBatchKernels::sharedGateBatch,
                                new KernelContext(),
                                x,
                                sharedGateInput,
                                sharedGate,
                                batchInfo,
                                dim)
                        .task(
                                "t",
                                Qwen35MoeBatchKernels::routerTopKBatch,
                                new KernelContext(),
                                logits,
                                ids,
                                weights,
                                batchInfo,
                                experts,
                                used)
                        .task(
                                "g",
                                Qwen35MoeBatchKernels::groupByExpert,
                                new KernelContext(),
                                ids,
                                batchInfo,
                                sortedToken,
                                position,
                                tiles,
                                experts,
                                used,
                                batch)
                        .task(
                                "gu",
                                Qwen35MoeBatchKernels::groupedGateUpQ8_0,
                                new KernelContext(),
                                xq,
                                xs,
                                sortedToken,
                                tiles,
                                gate,
                                up,
                                hiddenOut,
                                dim,
                                hidden)
                        .task(
                                "hq",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                hiddenOut,
                                hq,
                                hs,
                                hidden)
                        .task(
                                "d",
                                Qwen35MoeBatchKernels::groupedDownQ8_0,
                                new KernelContext(),
                                hq,
                                hs,
                                tiles,
                                down,
                                routed,
                                dim,
                                hidden)
                        .task(
                                "c",
                                Qwen35MoeBatchKernels::combine,
                                new KernelContext(),
                                routed,
                                position,
                                weights,
                                shared,
                                sharedGate,
                                residual,
                                batchInfo,
                                dim,
                                used)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION,
                                xq,
                                xs,
                                logits,
                                sharedGate,
                                ids,
                                weights,
                                sortedToken,
                                position,
                                tiles,
                                hiddenOut,
                                hq,
                                hs,
                                routed,
                                residual);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("moe.q", lanes(batch * dim, 128));
        WorkerGrid rg = new WorkerGrid2D(batch / 64 * 256, experts / 64);
        rg.setLocalWork(256, 1, 1);
        s.addWorkerGrid("moe.r", rg);
        s.addWorkerGrid("moe.sg", lanes(batch * 32, 128));
        s.addWorkerGrid("moe.t", lanes(batch * 32, 32));
        s.addWorkerGrid(
                "moe.g",
                lanes(Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS));
        WorkerGrid gu = new WorkerGrid2D(hidden / 128 * 256, maxTiles);
        gu.setLocalWork(256, 1, 1);
        s.addWorkerGrid("moe.gu", gu);
        s.addWorkerGrid("moe.hq", lanes(assignments * hidden, 128));
        WorkerGrid dg = new WorkerGrid2D(dim / 128 * 256, maxTiles);
        dg.setLocalWork(256, 1, 1);
        s.addWorkerGrid("moe.d", dg);
        s.addWorkerGrid("moe.c", lanes(batch * dim, 256));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }

        // Routing: the top-k of the device's logits, softmaxed over the kept ones.
        for (int t = 0; t < active; t++) {
            double[] row = new double[experts];
            for (int e = 0; e < experts; e++) {
                double l = 0;
                for (int i = 0; i < dim; i++) {
                    l += router.get(e * dim + i) * x.get(t * dim + i);
                }
                row[e] = l;
                assertEquals("logit " + t + "," + e, l, logits.get(t * experts + e), 1e-3);
            }
            boolean[] taken = new boolean[experts];
            double top = 0, total = 0;
            double[] kept = new double[used];
            for (int k = 0; k < used; k++) {
                int best = -1;
                for (int e = 0; e < experts; e++) {
                    if (!taken[e]
                            && (best < 0
                                    || logits.get(t * experts + e)
                                            > logits.get(t * experts + best))) {
                        best = e;
                    }
                }
                taken[best] = true;
                assertEquals("expert " + t + "," + k, best, ids.get(t * used + k));
                kept[k] = logits.get(t * experts + best);
            }
            for (int k = 0; k < used; k++) {
                total += Math.exp(kept[k] - kept[0]);
            }
            for (int k = 0; k < used; k++) {
                assertEquals(
                        "weight " + t + "," + k,
                        Math.exp(kept[k] - kept[0]) / total,
                        weights.get(t * used + k),
                        1e-5);
            }
        }

        // Grouping: every active assignment appears once, at its position, grouped by expert.
        int tileCount = tiles.get(0);
        int covered = 0;
        for (int i = 0; i < tileCount; i++) {
            covered += tiles.get(3 + 3 * i);
        }
        assertEquals(active * used, covered);
        for (int a = 0; a < active * used; a++) {
            int row = position.get(a);
            assertEquals("token of row " + row, a / used, sortedToken.get(row));
        }

        // Gate/up and down, against the device's own quantized inputs.
        double worstHidden = 0, worstDown = 0;
        for (int i = 0; i < tileCount; i++) {
            int expert = tiles.get(1 + 3 * i),
                    start = tiles.get(2 + 3 * i),
                    rows = tiles.get(3 + 3 * i);
            for (int r = start; r < start + rows; r++) {
                int token = sortedToken.get(r);
                for (int c = 0; c < hidden; c++) {
                    double gg = dot(gateRaw, expert * hidden + c, dim, xq, xs, token);
                    double uu = dot(upRaw, expert * hidden + c, dim, xq, xs, token);
                    double h = gg / (1 + Math.exp(-gg)) * uu;
                    worstHidden =
                            Math.max(
                                    worstHidden,
                                    Math.abs(h - hiddenOut.get(r * hidden + c))
                                            / (1 + Math.abs(h)));
                }
                for (int c = 0; c < dim; c++) {
                    double o = dot(downRaw, expert * dim + c, hidden, hq, hs, r);
                    worstDown =
                            Math.max(
                                    worstDown,
                                    Math.abs(o - routed.get(r * dim + c)) / (1 + Math.abs(o)));
                }
            }
        }
        assertTrue("gate/up worst " + worstHidden, worstHidden < 1e-3);
        assertTrue("down worst " + worstDown, worstDown < 1e-3);

        // Combine.
        double worstCombine = 0;
        for (int t = 0; t < batch; t++) {
            for (int c = 0; c < dim; c++) {
                double expect = residualBefore[t * dim + c];
                if (t < active) {
                    for (int k = 0; k < used; k++) {
                        int a = t * used + k;
                        expect += weights.get(a) * routed.get(position.get(a) * dim + c);
                    }
                    expect += sharedGate.get(t) * shared.get(t * dim + c);
                }
                worstCombine =
                        Math.max(
                                worstCombine,
                                Math.abs(expect - residual.get(t * dim + c))
                                        / (1 + Math.abs(expect)));
            }
        }
        assertTrue("combine worst " + worstCombine, worstCombine < 1e-4);
    }
}
