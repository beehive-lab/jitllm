package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jllm.backend.tornado.scheduling.LaneAttentionPolicy;
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
 * The lane-cooperative 128-wide attention kernel against a host reference, before any model runs
 * it.
 *
 * <p>The reference is computed on the CPU in FP32 from the <b>same FP16 values the cache holds</b>,
 * so the only differences left are summation order and the transcendental, not quantisation. It is
 * a plain three-pass softmax attention, deliberately not an online one, so that a mistake in the
 * kernel's running-maximum bookkeeping cannot be mirrored by the thing it is checked against.
 *
 * <p>Both halves of the operator are checked: the split partials the kernel writes, for finiteness
 * and for the layout the combine expects, and the combined output. A non-finite value or a wrong
 * shape fails before any numerical metric is computed, because a NaN that reaches a relative error
 * is reported as a small number.
 *
 * <p>Sequence lengths are chosen for the boundaries, not for realism: one and two positions, which
 * leave most splits empty; lengths either side of a warp (32), of the 16-warp block's stride (512),
 * of a page (64) and of an even split; and 512 and 2048 because those are the depths the decode
 * measurements are taken at. The block table is a reversed permutation and the slot is nonzero, so
 * a kernel that assumed physically contiguous pages or slot zero fails here. Q, K and V are three
 * unrelated functions with magnitudes spanning two orders, so a kernel that happened to work on
 * uniform input does not pass.
 */
// @formatter:on
public class LaneAttentionHead128AccelTest {

    private static final int HEAD_SIZE = 128;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int SPLITS = 8;
    private static final int PAGE = 64;
    private static final int SLOT = 1;
    private static final int SLOTS = 2;

    /** Deliberately spread: the largest query element is ~30x the smallest non-zero one. */
    private static float q(int head, int d) {
        return (float) (Math.sin(0.017 * d + 0.31 * head) * (1.0 + 4.0 * ((d + head) % 7)));
    }

    private static float k(int pos, int kvHead, int d) {
        return (float)
                (Math.cos(0.011 * d + 0.007 * pos + 0.9 * kvHead) * (0.05 + 0.5 * ((pos + d) % 5)));
    }

    private static float v(int pos, int kvHead, int d) {
        return (float)
                ((Math.sin(0.019 * d - 0.013 * pos + 0.4 * kvHead) + 0.25) * (0.1 + (d % 11)));
    }

    @Test
    public void theLaneKernelAgreesWithAHostReferenceAtEveryBoundary() throws Exception {
        assumeTrue(
                "this device does not select the lane-cooperative attention kernel",
                LaneAttentionPolicy.laneCooperativeAttention(HEAD_SIZE));

        int[] lengths = {
            1, 2, 5, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 511, 512, 513, 2048
        };
        for (int seqLen : lengths) {
            check(seqLen);
        }
    }

    private void check(int seqLen) throws Exception {
        int blocksPerSlot = (seqLen + PAGE - 1) / PAGE + 1;
        int blockCfg = PAGE | (blocksPerSlot << 16);
        int blockStride = PAGE * KV_DIM;

        // A reversed mapping, so logical order is not physical order anywhere.
        IntArray blockTable = new IntArray(SLOTS * blocksPerSlot);
        for (int s = 0; s < SLOTS; s++) {
            for (int b = 0; b < blocksPerSlot; b++) {
                blockTable.set(s * blocksPerSlot + b, s * blocksPerSlot + (blocksPerSlot - 1 - b));
            }
        }

        int physBlocks = SLOTS * blocksPerSlot;
        HalfFloatArray keyCache = new HalfFloatArray(physBlocks * blockStride);
        HalfFloatArray valueCache = new HalfFloatArray(physBlocks * blockStride);
        keyCache.init(new HalfFloat(0.0f));
        valueCache.init(new HalfFloat(0.0f));

        // Only the slot under test is filled; the other holds zeros, so reading the wrong slot
        // produces a visibly wrong answer rather than a plausible one.
        float[][][] kHost = new float[seqLen][KV_HEADS][HEAD_SIZE];
        float[][][] vHost = new float[seqLen][KV_HEADS][HEAD_SIZE];
        for (int p = 0; p < seqLen; p++) {
            int phys = blockTable.get(SLOT * blocksPerSlot + p / PAGE);
            int base = phys * blockStride + (p % PAGE) * KV_DIM;
            for (int kv = 0; kv < KV_HEADS; kv++) {
                for (int d = 0; d < HEAD_SIZE; d++) {
                    HalfFloat kh = new HalfFloat(k(p, kv, d));
                    HalfFloat vh = new HalfFloat(v(p, kv, d));
                    keyCache.set(base + kv * HEAD_SIZE + d, kh);
                    valueCache.set(base + kv * HEAD_SIZE + d, vh);
                    // The reference reads back what was stored, so FP16 rounding is common to both.
                    kHost[p][kv][d] = kh.getFloat32();
                    vHost[p][kv][d] = vh.getFloat32();
                }
            }
        }

        FloatArray query = new FloatArray(HEADS * HEAD_SIZE);
        for (int h = 0; h < HEADS; h++) {
            for (int d = 0; d < HEAD_SIZE; d++) {
                query.set(h * HEAD_SIZE + d, q(h, d));
            }
        }
        IntArray position = new IntArray(2);
        position.set(0, seqLen - 1);
        position.set(1, SLOT);

        int attSize = HEADS * SPLITS * (HEAD_SIZE + 2);
        FloatArray att = new FloatArray(attSize);
        att.init(Float.NaN);
        FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
        xb.init(Float.NaN);

        run(
                query,
                keyCache,
                valueCache,
                att,
                xb,
                position,
                blockTable,
                blockCfg,
                blockStride,
                seqLen);

        // --- shape and finiteness, before any metric ---------------------------------------
        assertEquals("partial buffer size", attSize, att.getSize());
        assertEquals("output size", HEADS * HEAD_SIZE, xb.getSize());
        for (int h = 0; h < HEADS; h++) {
            int headBase = h * SPLITS * (HEAD_SIZE + 2);
            int mBase = headBase + SPLITS * HEAD_SIZE;
            int lBase = mBase + SPLITS;
            int chunk = (seqLen + SPLITS - 1) / SPLITS;
            for (int s = 0; s < SPLITS; s++) {
                float ms = att.get(mBase + s);
                float ls = att.get(lBase + s);
                boolean empty = s * chunk >= seqLen;
                assertTrue(
                        "seqLen "
                                + seqLen
                                + " head "
                                + h
                                + " split "
                                + s
                                + ": L must be finite, was "
                                + ls,
                        Float.isFinite(ls));
                if (empty) {
                    assertTrue(
                            "seqLen "
                                    + seqLen
                                    + " head "
                                    + h
                                    + " split "
                                    + s
                                    + ": an empty split must be -inf, was "
                                    + ms,
                            ms == Float.NEGATIVE_INFINITY);
                    assertEquals(
                            "seqLen "
                                    + seqLen
                                    + " head "
                                    + h
                                    + " split "
                                    + s
                                    + ": an empty split's denominator must be zero",
                            0.0f,
                            ls,
                            0.0f);
                } else {
                    assertTrue(
                            "seqLen "
                                    + seqLen
                                    + " head "
                                    + h
                                    + " split "
                                    + s
                                    + ": a non-empty split's M must be finite, was "
                                    + ms,
                            Float.isFinite(ms));
                    assertTrue(
                            "seqLen "
                                    + seqLen
                                    + " head "
                                    + h
                                    + " split "
                                    + s
                                    + ": a non-empty split's L must be positive, was "
                                    + ls,
                            ls > 0.0f);
                }
                for (int d = 0; d < HEAD_SIZE; d++) {
                    float n = att.get(headBase + s * HEAD_SIZE + d);
                    assertTrue(
                            "seqLen "
                                    + seqLen
                                    + " head "
                                    + h
                                    + " split "
                                    + s
                                    + " dim "
                                    + d
                                    + ": numerator not finite ("
                                    + n
                                    + ")",
                            Float.isFinite(n));
                    if (empty) {
                        assertEquals(
                                "seqLen "
                                        + seqLen
                                        + ": an empty split's numerator must be an"
                                        + " explicit zero, not left unwritten",
                                0.0f,
                                n,
                                0.0f);
                    }
                }
            }
        }
        for (int i = 0; i < xb.getSize(); i++) {
            assertTrue(
                    "seqLen " + seqLen + ": output not finite at " + i + " (" + xb.get(i) + ")",
                    Float.isFinite(xb.get(i)));
        }

        // --- against the host reference -----------------------------------------------------
        double worstRel = 0;
        int worstAt = -1;
        double largest = 0;
        double sumSqDiff = 0;
        double sumSqRef = 0;
        for (int h = 0; h < HEADS; h++) {
            int kv = h / KV_MUL;
            double[] scores = new double[seqLen];
            double max = Double.NEGATIVE_INFINITY;
            double inv = 1.0 / Math.sqrt(HEAD_SIZE);
            for (int p = 0; p < seqLen; p++) {
                double dot = 0;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    dot += (double) query.get(h * HEAD_SIZE + d) * kHost[p][kv][d];
                }
                scores[p] = dot * inv;
                max = Math.max(max, scores[p]);
            }
            double denom = 0;
            for (int p = 0; p < seqLen; p++) {
                scores[p] = Math.exp(scores[p] - max);
                denom += scores[p];
            }
            for (int d = 0; d < HEAD_SIZE; d++) {
                double acc = 0;
                for (int p = 0; p < seqLen; p++) {
                    acc += scores[p] * vHost[p][kv][d];
                }
                double ref = acc / denom;
                double got = xb.get(h * HEAD_SIZE + d);
                largest = Math.max(largest, Math.abs(ref));
                double rel = Math.abs(ref - got) / Math.max(1e-3, Math.abs(ref));
                sumSqDiff += (ref - got) * (ref - got);
                sumSqRef += ref * ref;
                if (rel > worstRel) {
                    worstRel = rel;
                    worstAt = h * HEAD_SIZE + d;
                }
            }
        }
        double relL2 = Math.sqrt(sumSqDiff / sumSqRef);
        System.out.printf(
                "[LANEATTN] seqLen %4d: relL2 %.3g, worst elementwise %.3g at %d, largest |ref|"
                        + " %.3g%n",
                seqLen, relL2, worstRel, worstAt, largest);
        assertTrue("seqLen " + seqLen + ": nothing was written", largest > 0);
        assertTrue(
                "seqLen "
                        + seqLen
                        + ": worst relative error "
                        + worstRel
                        + " at element "
                        + worstAt,
                worstRel <= 2e-3);
        assertTrue("seqLen " + seqLen + ": relative L2 " + relL2, relL2 <= 1e-4);
    }

    private static void run(
            FloatArray query,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray att,
            FloatArray xb,
            IntArray position,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int seqLen)
            throws Exception {
        String name = "lane_" + seqLen;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                query,
                                keyCache,
                                valueCache,
                                position,
                                blockTable,
                                att,
                                xb)
                        .task(
                                "split",
                                TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead128,
                                new KernelContext(),
                                query,
                                keyCache,
                                valueCache,
                                att,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                position,
                                0,
                                blockTable,
                                blockCfg,
                                blockStride,
                                SPLITS)
                        .task(
                                "combine",
                                TransformerComputeKernelsLayered::combineSplitKVAttention,
                                new KernelContext(),
                                att,
                                xb,
                                HEADS,
                                HEAD_SIZE,
                                SPLITS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, att, xb);

        int local = LaneAttentionPolicy.WARPS_PER_GROUP * 32;
        WorkerGrid1D split = new WorkerGrid1D(HEADS * SPLITS * local);
        split.setLocalWork(local, 1, 1);
        WorkerGrid1D combine = new WorkerGrid1D(HEADS * 64);
        combine.setLocalWork(64, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".split", split);
        scheduler.addWorkerGrid(name + ".combine", combine);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }
}
