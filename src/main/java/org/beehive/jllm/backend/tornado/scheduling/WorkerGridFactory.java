package org.beehive.jllm.backend.tornado.scheduling;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;

public class WorkerGridFactory {
    private static final int DEFAULT_WORK_GROUP_SIZE = 32;

    /** Single-threaded worker */
    public static WorkerGrid createSingleWorker() {
        WorkerGrid worker = new WorkerGrid1D(1);
        worker.setGlobalWork(1, 1, 1);
        worker.setLocalWork(1, 1, 1);
        return worker;
    }

    /** RMS norm worker: 1D grid for normalization */
    public static WorkerGrid createRmsNormWorker(int dim, int localSize) {
        WorkerGrid worker = new WorkerGrid1D(dim);
        worker.setGlobalWork(dim, 1, 1);
        worker.setLocalWork(localSize, 1, 1);
        return worker;
    }

    /** QKV matmul worker: combined projection output */
    public static WorkerGrid createQkvMatmulWorker(int opSize) {
        int global = opSize * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    public static WorkerGrid genericWorker(int globalWorkSize, int localWorkSize) {
        WorkerGrid worker = new WorkerGrid1D(globalWorkSize);
        worker.setLocalWork(localWorkSize, 1, 1);
        return worker;
    }

    /** RoPE worker: 2D grid for position encoding */
    public static WorkerGrid createRoPEWorker(int numberOfHeads, int headSize) {
        int ic = headSize / 2;
        WorkerGrid worker = new WorkerGrid2D(numberOfHeads, ic);
        worker.setGlobalWork(numberOfHeads, ic, 1);
        worker.setLocalWork(8, 1, 1);
        return worker;
    }

    /** Attention worker: compute all heads in parallel */
    public static WorkerGrid createAttentionWorker(int numberOfHeads, int headSize) {
        int optimalLocalSize = findOptimalLocalSize(headSize);
        WorkerGrid worker = new WorkerGrid1D(numberOfHeads);
        worker.setGlobalWork(numberOfHeads * optimalLocalSize, 1, 1);
        worker.setLocalWork(optimalLocalSize, 1, 1);
        return worker;
    }

    // @formatter:off
    /**
     * One 32-lane warp per group, for the lane-cooperative attention kernel.
     *
     * <p>Deliberately not {@link #createAttentionWorker}: that one sizes the group from the head
     * width ({@code min(headSize, 64)}), which is the shape the per-key kernel wants. The
     * lane-cooperative kernel wants a whole number of warps whatever the head width, because its
     * lane indices address head dimensions directly and its warp index strides the key range; a
     * group that is not a multiple of 32 would silently compute a wrong answer rather than fail.
     *
     * @param warpsPerGroup how many warps share one (head, split). More warps put more independent
     *     key streams in flight per block, which is what the kernel needs to hide KV-cache latency
     *     at depth; the kernel folds them together itself.
     */
    // @formatter:on
    public static WorkerGrid createLaneAttentionWorker(int groups, int warpsPerGroup) {
        int local = warpsPerGroup * 32;
        WorkerGrid worker = new WorkerGrid1D(groups * local);
        worker.setGlobalWork(groups * local, 1, 1);
        worker.setLocalWork(local, 1, 1);
        return worker;
    }

    /** FFN gate+up worker: combined projection */
    public static WorkerGrid createGateUpWorker(int hiddenDim) {
        int global = (2 * hiddenDim) * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    /** FFN down worker: final projection */
    public static WorkerGrid createDownWorker(int dim) {
        int global = dim * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    private static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) {
                    optimal = s;
                    break;
                }
            }
        }
        return optimal;
    }
}
