package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
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
 * The split-KV replacement for this family's decode attention, against the per-head kernel it
 * replaces.
 *
 * <p>Same algorithm, same FP16 key/value interpretation, same paged addressing; what differs is
 * that each head's position range is cut into {@code SPLITS} chunks scanned by separate workgroups
 * and merged by a combine pass. The merge is the standard online-softmax rescaling, so the
 * summation order differs from one workgroup walking the whole range and the two agree to rounding
 * rather than bit for bit.
 *
 * <p>The sequence lengths are chosen for the split boundaries rather than for realism: one position
 * (every split but the first empty), fewer positions than splits, a length that divides evenly, and
 * three that leave a partial last chunk. The head is 256 wide throughout, which is the width the
 * shared 128-wide kernel cannot take and the reason the wide-head variant exists.
 */
// @formatter:on
public class Qwen35SplitKvAttentionAccelTest {

    private static final int HEADS = 4;
    private static final int KV_HEADS = 1;
    private static final int KV_MUL = HEADS / KV_HEADS;
    private static final int SPLITS = 8;
    private static final int SPLIT_LOCAL = 32;
    private static final int BLOCK_SIZE = 64;

    private static float q(int head, int d) {
        return (float) Math.sin(0.017 * d + 0.31 * head);
    }

    private static float k(int pos, int d) {
        return (float) (Math.cos(0.011 * d + 0.007 * pos) * 0.5);
    }

    private static float v(int pos, int d) {
        return (float) (Math.sin(0.019 * d - 0.013 * pos) + 0.25);
    }

    @Test
    public void theSplitReplacementAgreesWithThePerHeadKernelAt256() throws Exception {
        assertAgreesWithPerHeadKernel(256);
    }

    // @formatter:off
    /**
     * The same comparison at half the width, which is the other end of the eligibility rule.
     *
     * <p>{@code headSize <= SPLIT_KV_MAX_HEAD} admits any head the wide-head kernel's arrays can
     * address, not only the 256 this family ships. Running it at 128 checks that the rule is a
     * property of the indexing rather than a coincidence of the width it was written for: the
     * per-lane accumulator row is {@code tid * headSize}, so a narrower head simply uses less of
     * the same allocation.
     */
    // @formatter:on
    @Test
    public void theSplitReplacementAgreesWithThePerHeadKernelAt128() throws Exception {
        assertAgreesWithPerHeadKernel(128);
    }

    private void assertAgreesWithPerHeadKernel(int headSize) throws Exception {
        assumeTrue("no tensor-core-capable device", TensorCoreSupport.isTensorCoreCapableBackend());
        final int HEAD_SIZE = headSize;
        final int KV_DIM = KV_HEADS * HEAD_SIZE;

        for (int seqLen : new int[] {1, 5, 64, 100, 128, 381}) {
            int blocksPerSlot = (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
            int blockCfg = BLOCK_SIZE | (blocksPerSlot << 16);
            int blockStride = BLOCK_SIZE * KV_DIM;

            IntArray blockTable = new IntArray(blocksPerSlot);
            for (int b = 0; b < blocksPerSlot; b++) {
                blockTable.set(b, b); // identity mapping, one slot
            }
            HalfFloatArray keyCache = new HalfFloatArray(blocksPerSlot * blockStride);
            HalfFloatArray valueCache = new HalfFloatArray(blocksPerSlot * blockStride);
            keyCache.init(new HalfFloat(0.0f));
            valueCache.init(new HalfFloat(0.0f));
            for (int p = 0; p < seqLen; p++) {
                int base = (p / BLOCK_SIZE) * blockStride + (p % BLOCK_SIZE) * KV_DIM;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    keyCache.set(base + d, new HalfFloat(k(p, d)));
                    valueCache.set(base + d, new HalfFloat(v(p, d)));
                }
            }
            FloatArray query = new FloatArray(HEADS * HEAD_SIZE);
            for (int h = 0; h < HEADS; h++) {
                for (int d = 0; d < HEAD_SIZE; d++) {
                    query.set(h * HEAD_SIZE + d, q(h, d));
                }
            }
            IntArray position = new IntArray(2);
            position.set(0, seqLen - 1); // pos; the kernels read seqLen = pos + 1
            position.set(1, 0); // slot

            FloatArray reference =
                    runPerHead(
                            query,
                            keyCache,
                            valueCache,
                            position,
                            blockTable,
                            blockCfg,
                            blockStride,
                            seqLen,
                            HEAD_SIZE,
                            KV_DIM);
            FloatArray candidate =
                    runSplit(
                            query,
                            keyCache,
                            valueCache,
                            position,
                            blockTable,
                            blockCfg,
                            blockStride,
                            seqLen,
                            HEAD_SIZE,
                            KV_DIM);

            double largest = 0;
            double worst = 0;
            int worstAt = -1;
            for (int i = 0; i < HEADS * HEAD_SIZE; i++) {
                float ref = reference.get(i);
                float got = candidate.get(i);
                assertTrue(
                        "seqLen " + seqLen + ": reference not finite at " + i, Float.isFinite(ref));
                assertTrue(
                        "seqLen " + seqLen + ": candidate not finite at " + i, Float.isFinite(got));
                largest = Math.max(largest, Math.abs(ref));
                if (Math.abs(ref - got) > worst) {
                    worst = Math.abs(ref - got);
                    worstAt = i;
                }
            }
            System.out.printf(
                    "[SPLITKV] head %d seqLen %4d: worst %.3g of largest %.3g (head %d, el %d)%n",
                    HEAD_SIZE, seqLen, worst, largest, worstAt / HEAD_SIZE, worstAt % HEAD_SIZE);
            assertTrue(
                    "seqLen " + seqLen + ": worst " + worst + " against largest " + largest,
                    worst <= 1e-5 * Math.max(largest, 1.0));
            assertTrue("seqLen " + seqLen + ": nothing was written", largest > 0);
        }
    }

    private static FloatArray runPerHead(
            FloatArray query,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray position,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int seqLen,
            int HEAD_SIZE,
            int KV_DIM)
            throws Exception {
        FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
        xb.init(Float.NaN);
        String name = "perhead" + HEAD_SIZE + "_" + seqLen;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                query,
                                keyCache,
                                valueCache,
                                position,
                                blockTable,
                                xb)
                        .task(
                                name,
                                TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                                new KernelContext(),
                                query,
                                keyCache,
                                valueCache,
                                xb,
                                HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                position,
                                0,
                                blockTable,
                                blockCfg,
                                blockStride)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);
        WorkerGrid1D worker = new WorkerGrid1D(HEADS * 64);
        worker.setLocalWork(64, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + "." + name, worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        return xb;
    }

    private static FloatArray runSplit(
            FloatArray query,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray position,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int seqLen,
            int HEAD_SIZE,
            int KV_DIM)
            throws Exception {
        FloatArray xb = new FloatArray(HEADS * HEAD_SIZE);
        xb.init(Float.NaN);
        FloatArray att = new FloatArray(HEADS * SPLITS * (HEAD_SIZE + 2));
        att.init(Float.NaN);
        String name = "split" + HEAD_SIZE + "_" + seqLen;
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
                                name + "s",
                                Qwen35ReferenceKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PagedWideHead,
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
                                name + "c",
                                TransformerComputeKernelsLayered::combineSplitKVAttention,
                                new KernelContext(),
                                att,
                                xb,
                                HEADS,
                                HEAD_SIZE,
                                SPLITS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);
        WorkerGrid1D splitWorker = new WorkerGrid1D(HEADS * SPLITS * SPLIT_LOCAL);
        splitWorker.setLocalWork(SPLIT_LOCAL, 1, 1);
        WorkerGrid1D combineWorker = new WorkerGrid1D(HEADS * 64);
        combineWorker.setLocalWork(64, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + "." + name + "s", splitWorker);
        scheduler.addWorkerGrid(name + "." + name + "c", combineWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        return xb;
    }
}
