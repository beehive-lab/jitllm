package org.beehive.jitllm.backend.tornado.layers;

import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;

// @formatter:off
/**
 * The FP16 tensor-core GEMM of a batched prefill, with its depth split when the output is too
 * narrow to fill the device.
 *
 * <p>{@link TransformerBatchPrefillKernels#gemmMMA} gives one thread block per 128x128 output tile.
 * A projection into the model width at a chunk of 512 has 64 of them on a device of over a hundred
 * multiprocessors, so below {@link #MIN_BLOCKS} tiles the depth is cut into up to {@link
 * #MAX_SLICES} slices ({@link Gemma4BatchPrefillKernels#gemmMMASplitK}) whose partial products
 * {@link Gemma4BatchPrefillKernels#splitKReduce} then sums.
 *
 * <p>Under native libraries every GEMM is cuBLAS's ({@code cublasGemmExFP16FP32}), unsplit: the
 * weight row-major {@code [n][k]} is read transposed, as the Qwen3 F16 native projections read it.
 */
// @formatter:on
public final class SplitKGemm {

    /** Output tiles below which the depth is split. */
    static final int MIN_BLOCKS = 128;

    static final int MAX_SLICES = 4;

    /** {@code -Djitllm.prefill.splitK=false} keeps every GEMM whole. */
    private static final boolean ENABLED =
            !"false".equalsIgnoreCase(System.getProperty("jitllm.prefill.splitK", "true"));

    private final KernelContext context;
    private final boolean nativeGemm;
    private final FloatArray partial;
    private final Map<String, int[]> splitTasks = new LinkedHashMap<>();

    /**
     * @param rows the padded batch, the GEMMs' row count
     * @param widestSplit the widest output a split GEMM may write
     */
    public SplitKGemm(KernelContext context, int rows, int widestSplit, boolean nativeGemm) {
        this.context = context;
        this.nativeGemm = nativeGemm;
        this.partial =
                new FloatArray(Math.toIntExact(partialFloats(rows, widestSplit, nativeGemm)));
    }

    /** The partial-product buffer's floats: none to speak of when cuBLAS runs the GEMMs. */
    static long partialFloats(int rows, int widestSplit, boolean nativeGemm) {
        return nativeGemm ? 1 : (long) MAX_SLICES * rows * widestSplit;
    }

    /** The device bytes a split-K GEMM of this shape allocates. */
    public static long bytes(int rows, int widestSplit, boolean nativeGemm) {
        return partialFloats(rows, widestSplit, nativeGemm) * Float.BYTES;
    }

    /** Whether the GEMMs are cuBLAS's. */
    public boolean nativeGemm() {
        return nativeGemm;
    }

    /** The partial-product buffer, which every layer graph transfers or consumes. */
    public FloatArray partial() {
        return partial;
    }

    /** The depth slices for an {@code m x n} output over depth {@code k}: 1 for no split. */
    int slices(int m, int n, int k) {
        if (nativeGemm
                || !ENABLED
                || m % 128 != 0
                || (long) m * n * MAX_SLICES > partial.getSize()) {
            return 1;
        }
        int blocks = (m / 128) * (n / 128);
        int s = 1;
        while (s < MAX_SLICES && blocks * s < MIN_BLOCKS && (k / (s * 2)) % 16 == 0) {
            s *= 2;
        }
        return s;
    }

    /**
     * {@code c[m][n] = a[m][k] . b[n][k]^T} as task {@code task}. Unsplit, the grid is the
     * caller's; split, {@link #addGrids} registers it and the reduce task's.
     */
    public void gemm(
            TaskGraph layer,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k) {
        if (nativeGemm) {
            layer.libraryTask(
                    task,
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    n,
                    m,
                    k,
                    1.0f,
                    b,
                    k,
                    a,
                    k,
                    0.0f,
                    c,
                    n);
            return;
        }
        int s = slices(m, n, k);
        if (s == 1) {
            layer.task(task, TransformerBatchPrefillKernels::gemmMMA, context, a, b, c, m, n, k);
            return;
        }
        layer.task(
                task, Gemma4BatchPrefillKernels::gemmMMASplitK, context, a, b, partial, m, n, k, s);
        layer.task(
                task + "Reduce",
                Gemma4BatchPrefillKernels::splitKReduce,
                context,
                partial,
                c,
                m * n,
                s);
        splitTasks.put(layer.getTaskGraphName() + "." + task, new int[] {m, n, s});
    }

    /** Registers the split GEMMs' grids, replacing the caller's. Call after the caller's grids. */
    public void addGrids(GridScheduler scheduler) {
        splitTasks.forEach(
                (name, mns) -> {
                    WorkerGrid3D grid =
                            new WorkerGrid3D((mns[0] / 128) * 256, mns[1] / 128, mns[2]);
                    grid.setLocalWork(256, 1, 1);
                    scheduler.addWorkerGrid(name, grid);
                    scheduler.addWorkerGrid(
                            name + "Reduce", WorkerGridFactory.genericWorker(mns[0] * mns[1], 256));
                });
    }
}
