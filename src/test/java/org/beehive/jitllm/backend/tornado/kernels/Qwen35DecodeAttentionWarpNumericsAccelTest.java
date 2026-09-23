package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

// @formatter:off
/**
 * The warp-per-position split-KV decode attention ({@code
 * processHeadsFlashAttentionSplitKVFP16PagedWarp}) against an FP64 reference computed from the
 * stored FP16 keys and values, alongside the lane-per-position kernel it replaces, both followed by
 * the unchanged combine.
 *
 * <p>Production geometry (24 heads over 4 key/value heads, 256 wide), shuffled pages, sequence
 * lengths from one position to 2049, random queries with a few large logits, and 8, 16 and 32
 * splits. The candidate's error must not exceed the control's by more than a small factor on any
 * case; a negative control (one product dropped from the score) must fail. An opt-in screen times
 * both kernels at depths 512 and 2048.
 */
// @formatter:on
public class Qwen35DecodeAttentionWarpNumericsAccelTest {

    private static final int HEADS = 24;
    private static final int KV_HEADS = 4;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int HEAD_SIZE = 256;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int BLOCK_SIZE = 16;
    private static final int LOCAL = 32;

    private static final class Store {
        final int blocksPerSlot;
        final int blockCfg;
        final int blockStride;
        final IntArray blockTable;
        final HalfFloatArray keys;
        final HalfFloatArray values;
        final int seqLen;

        Store(int seqLen, long seed) {
            this.seqLen = seqLen;
            blocksPerSlot = (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
            blockCfg = BLOCK_SIZE | (blocksPerSlot << 16);
            blockStride = BLOCK_SIZE * KV_DIM;
            int physical = blocksPerSlot + 3;
            Random rng = new Random(seed);
            keys = new HalfFloatArray(physical * blockStride);
            values = new HalfFloatArray(physical * blockStride);
            for (int i = 0; i < keys.getSize(); i++) {
                keys.set(i, new HalfFloat((float) rng.nextGaussian() * 0.5f));
                values.set(i, new HalfFloat((float) rng.nextGaussian()));
            }
            Integer[] pages = new Integer[physical];
            for (int i = 0; i < physical; i++) {
                pages[i] = i;
            }
            java.util.Collections.shuffle(Arrays.asList(pages), rng);
            blockTable = new IntArray(blocksPerSlot);
            for (int i = 0; i < blocksPerSlot; i++) {
                blockTable.set(i, pages[i]);
            }
        }

        int base(int p, int kvHead) {
            return blockTable.get(p / BLOCK_SIZE) * blockStride
                    + (p % BLOCK_SIZE) * KV_DIM
                    + kvHead * HEAD_SIZE;
        }
    }

    private static FloatArray queries(long seed) {
        Random rng = new Random(seed);
        FloatArray q = new FloatArray(HEADS * HEAD_SIZE);
        for (int i = 0; i < q.getSize(); i++) {
            float v = (float) rng.nextGaussian();
            if (rng.nextInt(64) == 0) {
                v *= 6.0f;
            }
            q.set(i, v);
        }
        return q;
    }

    private static double[] reference(Store store, FloatArray q) {
        double[] out = new double[HEADS * HEAD_SIZE];
        double invSqrt = 1.0 / Math.sqrt(HEAD_SIZE);
        for (int h = 0; h < HEADS; h++) {
            int kvHead = h / KV_MUL;
            double[] scores = new double[store.seqLen];
            double max = Double.NEGATIVE_INFINITY;
            for (int p = 0; p < store.seqLen; p++) {
                int base = store.base(p, kvHead);
                double s = 0;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    s += (double) q.get(h * HEAD_SIZE + d) * store.keys.get(base + d).getFloat32();
                }
                scores[p] = s * invSqrt;
                max = Math.max(max, scores[p]);
            }
            double sum = 0;
            for (int p = 0; p < store.seqLen; p++) {
                scores[p] = Math.exp(scores[p] - max);
                sum += scores[p];
            }
            for (int p = 0; p < store.seqLen; p++) {
                int base = store.base(p, kvHead);
                double w = scores[p] / sum;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    out[h * HEAD_SIZE + d] += w * store.values.get(base + d).getFloat32();
                }
            }
        }
        return out;
    }

    enum Kernel {
        LANE,
        WARP,
        BROKEN
    }

    private static FloatArray run(Kernel kernel, Store store, FloatArray q, int splits)
            throws Exception {
        FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
        xb.init(Float.NaN);
        FloatArray att = new FloatArray(HEADS * splits * (HEAD_SIZE + 2));
        att.init(Float.NaN);
        IntArray position = new IntArray(2);
        position.set(0, store.seqLen - 1);
        position.set(1, 0);
        String name = "d" + kernel + splits;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                q,
                                store.keys,
                                store.values,
                                position,
                                store.blockTable,
                                att,
                                xb);
        switch (kernel) {
            case LANE ->
                    graph.task(
                            "s",
                            Qwen35ReferenceKernels
                                    ::processHeadsFlashAttentionSplitKVFP16PagedWideHead,
                            new KernelContext(),
                            q,
                            store.keys,
                            store.values,
                            att,
                            HEADS,
                            HEAD_SIZE,
                            KV_DIM,
                            KV_MUL,
                            position,
                            0,
                            store.blockTable,
                            store.blockCfg,
                            store.blockStride,
                            splits);
            case WARP ->
                    graph.task(
                            "s",
                            TransformerPagedKvKernels
                                    ::processHeadsFlashAttentionSplitKVFP16PagedWarp,
                            new KernelContext(),
                            q,
                            store.keys,
                            store.values,
                            att,
                            HEADS,
                            HEAD_SIZE,
                            KV_DIM,
                            KV_MUL,
                            position,
                            0,
                            store.blockTable,
                            store.blockCfg,
                            store.blockStride,
                            splits);
            default ->
                    graph.task(
                            "s",
                            Qwen35DecodeAttentionWarpNumericsAccelTest::warpDroppingAProduct,
                            new KernelContext(),
                            q,
                            store.keys,
                            store.values,
                            att,
                            HEADS,
                            HEAD_SIZE,
                            KV_DIM,
                            KV_MUL,
                            position,
                            0,
                            store.blockTable,
                            store.blockCfg,
                            store.blockStride,
                            splits);
        }
        graph.task(
                        "c",
                        TransformerComputeKernelsLayered::combineSplitKVAttention,
                        new KernelContext(),
                        att,
                        xb,
                        HEADS,
                        HEAD_SIZE,
                        splits)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);
        WorkerGrid1D splitWorker = new WorkerGrid1D(HEADS * splits * LOCAL);
        splitWorker.setLocalWork(LOCAL, 1, 1);
        WorkerGrid1D combineWorker = new WorkerGrid1D(HEADS * 64);
        combineWorker.setLocalWork(64, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".s", splitWorker);
        scheduler.addWorkerGrid(name + ".c", combineWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        return xb;
    }

    private static double[] errors(FloatArray got, double[] ref, String what) {
        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < ref.length; i++) {
            assertTrue(what + ": not finite at " + i, Float.isFinite(got.get(i)));
            double e = got.get(i) - ref[i];
            num += e * e;
            den += ref[i] * ref[i];
            maxAbs = Math.max(maxAbs, Math.abs(e));
        }
        return new double[] {Math.sqrt(num / den), maxAbs};
    }

    @Test
    public void theWarpKernelIsAsCloseToFP64AsTheLaneKernel() throws Exception {
        assumeTrue("CUDA only (warp shuffles)", TensorCoreSupport.isTensorCoreCapableBackend());
        int[] seqLens = {1, 5, 17, 64, 100, 512, 513, 2048, 2049};
        int[] splitsList = {8, 16, 32};
        for (int seqLen : seqLens) {
            Store store = new Store(seqLen, 11L + seqLen);
            FloatArray q = queries(23L + seqLen);
            double[] ref = reference(store, q);
            for (int splits : splitsList) {
                double[] lane = errors(run(Kernel.LANE, store, q, splits), ref, "lane " + seqLen);
                double[] warp = errors(run(Kernel.WARP, store, q, splits), ref, "warp " + seqLen);
                System.out.printf(
                        Locale.ROOT,
                        "[decode-attn] seqLen %4d splits %2d: lane relL2 %.3e maxAbs %.3e | warp relL2 %.3e maxAbs %.3e%n",
                        seqLen,
                        splits,
                        lane[0],
                        lane[1],
                        warp[0],
                        warp[1]);
                assertTrue(
                        "seqLen "
                                + seqLen
                                + " splits "
                                + splits
                                + ": warp relL2 "
                                + warp[0]
                                + " vs lane "
                                + lane[0],
                        warp[0] <= Math.max(4.0 * lane[0], 1e-6));
                assertTrue(
                        "seqLen "
                                + seqLen
                                + " splits "
                                + splits
                                + ": warp maxAbs "
                                + warp[1]
                                + " vs lane "
                                + lane[1],
                        warp[1] <= Math.max(4.0 * lane[1], 1e-6));
            }
        }
    }

    @Test
    public void aDroppedProductFails() throws Exception {
        assumeTrue("CUDA only (warp shuffles)", TensorCoreSupport.isTensorCoreCapableBackend());
        Store store = new Store(300, 5L);
        FloatArray q = queries(6L);
        double[] ref = reference(store, q);
        double[] good = errors(run(Kernel.WARP, store, q, 8), ref, "warp");
        double[] bad = errors(run(Kernel.BROKEN, store, q, 8), ref, "broken");
        assertTrue(
                "the broken kernel was as accurate: " + bad[0] + " vs " + good[0],
                bad[0] > 10 * good[0]);
    }

    /** Opt in with JITLLM_KERNEL_SCREEN=true: both kernels + combine at depths 512 and 2048. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        for (int seqLen : new int[] {512, 2048}) {
            Store store = new Store(seqLen, 7L);
            FloatArray q = queries(8L);
            IntArray position = new IntArray(2);
            position.set(0, seqLen - 1);
            for (int splits : new int[] {8, 16, 32}) {
                for (Kernel kernel : new Kernel[] {Kernel.LANE, Kernel.WARP}) {
                    FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
                    FloatArray att = new FloatArray(HEADS * splits * (HEAD_SIZE + 2));
                    String name = "s" + kernel + splits;
                    TaskGraph graph =
                            new TaskGraph(name)
                                    .transferToDevice(
                                            DataTransferMode.FIRST_EXECUTION,
                                            q,
                                            store.keys,
                                            store.values,
                                            position,
                                            store.blockTable,
                                            att,
                                            xb);
                    if (kernel == Kernel.LANE) {
                        graph.task(
                                "s",
                                Qwen35ReferenceKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PagedWideHead,
                                new KernelContext(),
                                q,
                                store.keys,
                                store.values,
                                att,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                position,
                                0,
                                store.blockTable,
                                store.blockCfg,
                                store.blockStride,
                                splits);
                    } else {
                        graph.task(
                                "s",
                                TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PagedWarp,
                                new KernelContext(),
                                q,
                                store.keys,
                                store.values,
                                att,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                position,
                                0,
                                store.blockTable,
                                store.blockCfg,
                                store.blockStride,
                                splits);
                    }
                    graph.task(
                                    "c",
                                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                                    new KernelContext(),
                                    att,
                                    xb,
                                    HEADS,
                                    HEAD_SIZE,
                                    splits)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, xb);
                    WorkerGrid1D sw = new WorkerGrid1D(HEADS * splits * LOCAL);
                    sw.setLocalWork(LOCAL, 1, 1);
                    WorkerGrid1D cw = new WorkerGrid1D(HEADS * 64);
                    cw.setLocalWork(64, 1, 1);
                    GridScheduler sch = new GridScheduler();
                    sch.addWorkerGrid(name + ".s", sw);
                    sch.addWorkerGrid(name + ".c", cw);
                    try (TornadoExecutionPlan p = new TornadoExecutionPlan(graph.snapshot())) {
                        p.withGridScheduler(sch).withProfiler(ProfilerMode.SILENT);
                        for (int i = 0; i < 5; i++) {
                            p.execute();
                        }
                        long[] t = new long[21];
                        for (int i = 0; i < t.length; i++) {
                            t[i] = p.execute().getProfilerResult().getDeviceKernelTime();
                        }
                        Arrays.sort(t);
                        System.out.printf(
                                Locale.ROOT,
                                "[screen] %-5s splits %2d seqLen %4d: min %.1f median %.1f max %.1f us%n",
                                kernel,
                                splits,
                                seqLen,
                                t[0] / 1e3,
                                t[10] / 1e3,
                                t[20] / 1e3);
                    }
                }
            }
        }
    }

    /**
     * The warp kernel with one lane's last product dropped from the score: the negative control.
     */
    public static void warpDroppingAProduct(
            KernelContext context,
            FloatArray q,
            HalfFloatArray key_cache,
            HalfFloatArray value_cache,
            FloatArray att,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int nSplits) {
        int lane = context.localIdx;
        int g = context.groupIdx;
        int h = g / nSplits;
        int s = g - h * nSplits;
        if (h >= nHeads) {
            return;
        }
        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int seqLen = pos + 1;
        int chunk = (seqLen + nSplits - 1) / nSplits;
        int startPos = s * chunk;
        int endPos = Math.min(startPos + chunk, seqLen);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        float invSqrt = 1.0f / uk.ac.manchester.tornado.api.math.TornadoMath.sqrt(headSize);
        int headBase = h * nSplits * (headSize + 2);
        int outBase = headBase + s * headSize;
        int mBase = headBase + nSplits * headSize;
        int lBase = mBase + nSplits;
        int qBase = h * headSize + (lane << 1);
        float q0 = q.get(qBase),
                q1 = q.get(qBase + 1),
                q2 = q.get(qBase + 64),
                q3 = q.get(qBase + 65),
                q4 = q.get(qBase + 128),
                q5 = q.get(qBase + 129),
                q6 = q.get(qBase + 192);
        float m = Float.NEGATIVE_INFINITY, l = 0.0f;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (int p = startPos; p < endPos; p++) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHeadIdx * headSize
                            + (lane << 1);
            Half2 k0 = key_cache.getHalf2(base),
                    k1 = key_cache.getHalf2(base + 64),
                    k2 = key_cache.getHalf2(base + 128),
                    k3 = key_cache.getHalf2(base + 192);
            Half2 v0 = value_cache.getHalf2(base),
                    v1 = value_cache.getHalf2(base + 64),
                    v2 = value_cache.getHalf2(base + 128),
                    v3 = value_cache.getHalf2(base + 192);
            float partial =
                    q0 * Half2.lowFloat(k0)
                            + q1 * Half2.highFloat(k0)
                            + q2 * Half2.lowFloat(k1)
                            + q3 * Half2.highFloat(k1)
                            + q4 * Half2.lowFloat(k2)
                            + q5 * Half2.highFloat(k2)
                            + q6 * Half2.lowFloat(k3); // q7 term dropped
            partial += context.simdShuffleDown(partial, 16);
            partial += context.simdShuffleDown(partial, 8);
            partial += context.simdShuffleDown(partial, 4);
            partial += context.simdShuffleDown(partial, 2);
            partial += context.simdShuffleDown(partial, 1);
            float score = context.simdBroadcastFirst(partial) * invSqrt;
            float newM = Math.max(m, score);
            float corr =
                    (m == Float.NEGATIVE_INFINITY)
                            ? 0.0f
                            : uk.ac.manchester.tornado.api.math.TornadoMath.exp(m - newM);
            float e = uk.ac.manchester.tornado.api.math.TornadoMath.exp(score - newM);
            a0 = a0 * corr + e * Half2.lowFloat(v0);
            a1 = a1 * corr + e * Half2.highFloat(v0);
            a2 = a2 * corr + e * Half2.lowFloat(v1);
            a3 = a3 * corr + e * Half2.highFloat(v1);
            a4 = a4 * corr + e * Half2.lowFloat(v2);
            a5 = a5 * corr + e * Half2.highFloat(v2);
            a6 = a6 * corr + e * Half2.lowFloat(v3);
            a7 = a7 * corr + e * Half2.highFloat(v3);
            l = l * corr + e;
            m = newM;
        }
        int o = outBase + (lane << 1);
        att.set(o, a0);
        att.set(o + 1, a1);
        att.set(o + 64, a2);
        att.set(o + 65, a3);
        att.set(o + 128, a4);
        att.set(o + 129, a5);
        att.set(o + 192, a6);
        att.set(o + 193, a7);
        if (lane == 0) {
            att.set(mBase + s, m);
            att.set(lBase + s, l);
        }
    }
}
