package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.junit.BeforeClass;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The lane-cooperative split-KV decode attention for a 256-wide head ({@code
 * processHeadsFlashAttentionSplitKVPagedLaneHead256} over the FP32 cache and {@code
 * processHeadsFlashAttentionSplitKVFP16PagedLaneHead256} over the FP16 one), each followed by the
 * unchanged combine, against an FP64 reference computed from the stored keys and values, alongside
 * the per-head 8-position-tile kernels they replace on Metal.
 *
 * <p>Qwen3.5-0.8B geometry (8 query heads over 2 key/value heads, 256 wide), shuffled pages,
 * sequence lengths from one position to 2049, random queries with a few large logits, 4, 8 and 16
 * splits and 4, 8 and 16 warps per group. The candidate's error must not exceed the control's by
 * more than a small factor on any case; a negative control (a query dimension zeroed) must fail. An
 * opt-in screen times both at depths 680 and 2000.
 *
 * <p>Runs where {@link LaneAttentionPolicy#laneCooperativeAttentionHead256} admits the kernels
 * (Metal); skips elsewhere.
 */
// @formatter:on
public class Qwen35DecodeAttentionLaneHead256NumericsAccelTest {

    private static final int HEADS = 8;
    private static final int KV_HEADS = 2;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int HEAD_SIZE = 256;
    private static final int KV_DIM = KV_HEADS * HEAD_SIZE;
    private static final int BLOCK_SIZE = 16;

    @BeforeClass
    public static void onTheGpu() {
        System.setProperty("use.tornadovm", "true");
    }

    private static void assumeAdmitted() {
        assumeTrue(
                "the 256-wide lane-cooperative kernels are not admitted on this device",
                LaneAttentionPolicy.laneCooperativeAttentionHead256(HEAD_SIZE));
    }

    private static final class Store {
        final int blockCfg;
        final int blockStride;
        final IntArray blockTable;
        final FloatArray keys;
        final FloatArray values;
        final HalfFloatArray keysHalf;
        final HalfFloatArray valuesHalf;
        final int seqLen;

        Store(int seqLen, long seed) {
            this.seqLen = seqLen;
            int blocksPerSlot = (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
            blockCfg = BLOCK_SIZE | (blocksPerSlot << 16);
            blockStride = BLOCK_SIZE * KV_DIM;
            int physical = blocksPerSlot + 3;
            Random rng = new Random(seed);
            keys = new FloatArray(physical * blockStride);
            values = new FloatArray(physical * blockStride);
            keysHalf = new HalfFloatArray(physical * blockStride);
            valuesHalf = new HalfFloatArray(physical * blockStride);
            for (int i = 0; i < keys.getSize(); i++) {
                // Values representable in FP16, so both caches hold the same numbers and share
                // one reference.
                float k = new HalfFloat((float) rng.nextGaussian() * 0.5f).getFloat32();
                float v = new HalfFloat((float) rng.nextGaussian()).getFloat32();
                keys.set(i, k);
                values.set(i, v);
                keysHalf.set(i, new HalfFloat(k));
                valuesHalf.set(i, new HalfFloat(v));
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
                    s += (double) q.get(h * HEAD_SIZE + d) * store.keys.get(base + d);
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
                    out[h * HEAD_SIZE + d] += w * store.values.get(base + d);
                }
            }
        }
        return out;
    }

    enum Kernel {
        /** The per-head 8-position-tile kernel over the FP32 cache: the control. */
        TILE8,
        /** The per-head 8-position-tile kernel over the FP16 cache: the control. */
        TILE8_FP16,
        LANE256,
        LANE256_FP16
    }

    private static boolean split(Kernel kernel) {
        return kernel == Kernel.LANE256 || kernel == Kernel.LANE256_FP16;
    }

    private static TaskGraph graph(
            String name,
            Kernel kernel,
            Store store,
            FloatArray q,
            FloatArray att,
            FloatArray xb,
            IntArray position,
            int splits,
            int in,
            int out) {
        boolean half = kernel == Kernel.TILE8_FP16 || kernel == Kernel.LANE256_FP16;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                in,
                                q,
                                half ? store.keysHalf : store.keys,
                                half ? store.valuesHalf : store.values,
                                position,
                                store.blockTable,
                                att,
                                xb);
        switch (kernel) {
            case TILE8 ->
                    graph.task(
                            "s",
                            TransformerPagedKvKernels::processHeadsFlashAttentionPagedTile8,
                            new KernelContext(),
                            q,
                            store.keys,
                            store.values,
                            xb,
                            HEADS,
                            HEAD_SIZE,
                            KV_DIM,
                            KV_MUL,
                            position,
                            0,
                            store.blockTable,
                            store.blockCfg,
                            store.blockStride);
            case TILE8_FP16 ->
                    graph.task(
                            "s",
                            TransformerPagedKvKernels::processHeadsFlashAttentionFP16PagedTile8,
                            new KernelContext(),
                            q,
                            store.keysHalf,
                            store.valuesHalf,
                            xb,
                            HEADS,
                            HEAD_SIZE,
                            KV_DIM,
                            KV_MUL,
                            position,
                            0,
                            store.blockTable,
                            store.blockCfg,
                            store.blockStride);
            case LANE256 ->
                    graph.task(
                            "s",
                            TransformerPagedKvKernels
                                    ::processHeadsFlashAttentionSplitKVPagedLaneHead256,
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
            case LANE256_FP16 ->
                    graph.task(
                            "s",
                            TransformerPagedKvKernels
                                    ::processHeadsFlashAttentionSplitKVFP16PagedLaneHead256,
                            new KernelContext(),
                            q,
                            store.keysHalf,
                            store.valuesHalf,
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
        if (split(kernel)) {
            graph.task(
                    "c",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    new KernelContext(),
                    att,
                    xb,
                    HEADS,
                    HEAD_SIZE,
                    splits);
        }
        return graph.transferToHost(out, xb);
    }

    private static GridScheduler scheduler(String name, Kernel kernel, int splits, int warps) {
        GridScheduler scheduler = new GridScheduler();
        if (split(kernel)) {
            scheduler.addWorkerGrid(
                    name + ".s",
                    WorkerGridFactory.createLaneAttentionWorker(HEADS * splits, warps));
            scheduler.addWorkerGrid(
                    name + ".c", WorkerGridFactory.createAttentionWorker(HEADS, HEAD_SIZE));
        } else {
            WorkerGrid perHead = WorkerGridFactory.createAttentionWorker(HEADS, HEAD_SIZE);
            scheduler.addWorkerGrid(name + ".s", perHead);
        }
        return scheduler;
    }

    private static FloatArray run(Kernel kernel, Store store, FloatArray q, int splits, int warps)
            throws Exception {
        FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
        xb.init(Float.NaN);
        FloatArray att = new FloatArray(HEADS * Math.max(1, splits) * (HEAD_SIZE + 2));
        att.init(Float.NaN);
        IntArray position = new IntArray(2);
        position.set(0, store.seqLen - 1);
        position.set(1, 0);
        String name = "a" + kernel + splits + "w" + warps + "n" + store.seqLen;
        TaskGraph graph =
                graph(
                        name,
                        kernel,
                        store,
                        q,
                        att,
                        xb,
                        position,
                        splits,
                        DataTransferMode.EVERY_EXECUTION,
                        DataTransferMode.EVERY_EXECUTION);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler(name, kernel, splits, warps)).execute();
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
    public void theLaneKernelsAreAsCloseToFP64AsThePerHeadKernels() throws Exception {
        assumeAdmitted();
        int[] seqLens = {1, 5, 17, 64, 100, 513, 680, 2049};
        int[][] shapes = {{4, 8}, {8, 4}, {8, 8}, {8, 16}, {16, 8}};
        for (int seqLen : seqLens) {
            Store store = new Store(seqLen, 11L + seqLen);
            FloatArray q = queries(23L + seqLen);
            double[] ref = reference(store, q);
            for (Kernel[] pair :
                    new Kernel[][] {
                        {Kernel.TILE8, Kernel.LANE256}, {Kernel.TILE8_FP16, Kernel.LANE256_FP16}
                    }) {
                double[] control =
                        errors(run(pair[0], store, q, 1, 1), ref, pair[0] + " " + seqLen);
                for (int[] shape : shapes) {
                    int splits = shape[0];
                    int warps = shape[1];
                    double[] lane =
                            errors(
                                    run(pair[1], store, q, splits, warps),
                                    ref,
                                    pair[1] + " " + seqLen);
                    System.out.printf(
                            Locale.ROOT,
                            "[decode-attn-256] %-12s seqLen %4d splits %2d warps %2d: control"
                                    + " relL2 %.3e maxAbs %.3e | lane relL2 %.3e maxAbs %.3e%n",
                            pair[1],
                            seqLen,
                            splits,
                            warps,
                            control[0],
                            control[1],
                            lane[0],
                            lane[1]);
                    String where =
                            pair[1] + " seqLen " + seqLen + " splits " + splits + " warps " + warps;
                    assertTrue(
                            where + ": relL2 " + lane[0] + " vs control " + control[0],
                            lane[0] <= Math.max(4.0 * control[0], 1e-6));
                    assertTrue(
                            where + ": maxAbs " + lane[1] + " vs control " + control[1],
                            lane[1] <= Math.max(4.0 * control[1], 1e-6));
                }
            }
        }
    }

    @Test
    public void aWrongQueryFails() throws Exception {
        assumeAdmitted();
        Store store = new Store(300, 5L);
        FloatArray q = queries(6L);
        double[] ref = reference(store, q);
        double[] good = errors(run(Kernel.LANE256, store, q, 8, 8), ref, "lane");
        // The kernel is right; the query it is given is not: one dimension per head zeroed, which
        // the comparison must notice.
        FloatArray wrong = new FloatArray(q.getSize());
        for (int i = 0; i < q.getSize(); i++) {
            wrong.set(i, (i % HEAD_SIZE) == 193 ? 0.0f : q.get(i));
        }
        double[] bad = errors(run(Kernel.LANE256, store, wrong, 8, 8), ref, "wrong query");
        assertTrue(
                "a wrong query was as accurate: " + bad[0] + " vs " + good[0],
                bad[0] > 10 * good[0]);
    }

    /** Opt in with JITLLM_KERNEL_SCREEN=true: each kernel (+ combine) at depths 680 and 2000. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        assumeAdmitted();
        int[][] shapes = {{1, 1}, {4, 8}, {8, 4}, {8, 8}, {8, 16}, {16, 4}, {16, 8}, {32, 4}};
        for (int seqLen : new int[] {680, 2000}) {
            Store store = new Store(seqLen, 7L);
            FloatArray q = queries(8L);
            IntArray position = new IntArray(2);
            position.set(0, seqLen - 1);
            for (Kernel kernel : Kernel.values()) {
                for (int[] shape : shapes) {
                    if (split(kernel) == (shape[0] == 1)) {
                        continue;
                    }
                    int splits = shape[0];
                    int warps = shape[1];
                    FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
                    FloatArray att = new FloatArray(HEADS * splits * (HEAD_SIZE + 2));
                    String name = "s" + kernel + splits + "w" + warps + "n" + seqLen;
                    TaskGraph graph =
                            graph(
                                    name,
                                    kernel,
                                    store,
                                    q,
                                    att,
                                    xb,
                                    position,
                                    splits,
                                    DataTransferMode.FIRST_EXECUTION,
                                    DataTransferMode.UNDER_DEMAND);
                    try (TornadoExecutionPlan p = new TornadoExecutionPlan(graph.snapshot())) {
                        p.withGridScheduler(scheduler(name, kernel, splits, warps))
                                .withProfiler(ProfilerMode.SILENT);
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
                                "[screen-256] %-12s splits %2d warps %2d seqLen %4d: min %.1f"
                                        + " median %.1f max %.1f us%n",
                                kernel,
                                splits,
                                warps,
                                seqLen,
                                t[0] / 1e3,
                                t[10] / 1e3,
                                t[20] / 1e3);
                    }
                }
            }
        }
    }
}
