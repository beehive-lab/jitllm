package org.beehive.jitllm.backend.tornado.layers;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MMAKernels;
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

    /** The partial floats a split may use: what a plan of these rows allocates. */
    private final long capacity;

    /**
     * The slices chosen for each GEMM shape, by {@code n} and {@code k}: this instance's, or for a
     * narrowed one its primary's, which it takes instead of choosing.
     */
    private final Map<Long, Integer> chosen;

    private final boolean narrowed;

    private final Map<String, int[]> splitTasks = new LinkedHashMap<>();

    /**
     * @param rows the padded batch, the GEMMs' row count
     * @param widestSplit the widest output a split GEMM may write
     */
    public SplitKGemm(KernelContext context, int rows, int widestSplit, boolean nativeGemm) {
        this(
                context,
                nativeGemm,
                new FloatArray(Math.toIntExact(partialFloats(rows, widestSplit, nativeGemm))),
                partialFloats(rows, widestSplit, nativeGemm),
                new HashMap<>(),
                false);
    }

    private SplitKGemm(
            KernelContext context,
            boolean nativeGemm,
            FloatArray partial,
            long capacity,
            Map<Long, Integer> chosen,
            boolean narrowed) {
        this.context = context;
        this.nativeGemm = nativeGemm;
        this.partial = partial;
        this.capacity = Math.min(capacity, partial.getSize());
        this.chosen = chosen;
        this.narrowed = narrowed;
    }

    /**
     * These GEMMs over {@code rows}, for the narrow family ({@link PrefillFamily}): each shape
     * splits as this one split it, into this buffer, which the family's graphs bind from this
     * one's. Its rows still size the grids, the indexing and the capacity a split may use.
     *
     * <p>Example: a projection into a model 2,048 wide has 64 output tiles at 512 rows, and splits
     * in two; at 128 rows it has 16, and would split in four on its own rows, summing its depth in
     * another order than the plan does.
     */
    public SplitKGemm narrowed(int rows, int widestSplit) {
        return new SplitKGemm(
                context,
                nativeGemm,
                partial,
                partialFloats(rows, widestSplit, nativeGemm),
                chosen,
                true);
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

    /**
     * The depth slices for an {@code m x n} output over depth {@code k}: 1 for no split. A narrowed
     * instance takes the slices its primary chose for {@code n x k}, 1 where it chose none.
     */
    int slices(int m, int n, int k) {
        if (nativeGemm || !ENABLED || m % 128 != 0 || (long) m * n * MAX_SLICES > capacity) {
            return 1;
        }
        long shape = (long) n << 32 | k;
        if (narrowed) {
            return chosen.getOrDefault(shape, 1);
        }
        int blocks = (m / 128) * (n / 128);
        int s = 1;
        while (s < MAX_SLICES && blocks * s < MIN_BLOCKS && (k / (s * 2)) % 16 == 0) {
            s *= 2;
        }
        chosen.put(shape, s);
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

    /**
     * Whether {@link #gemmTiled} can multiply this shape: unsplit, or split into slices whose depth
     * {@link Qwen35MMAKernels#gemmMMATiledBStaged} steps through.
     */
    boolean tiled(int m, int n, int k) {
        int s = slices(m, n, k);
        return s == 1 || (k / s) % 32 == 0;
    }

    /**
     * {@link #gemm} over a weight decoded in the tile order of {@link
     * Qwen35MMAKernels#gemmMMATiledB}, split as {@link #gemm} splits it. Unsplit, the grid is the
     * caller's.
     */
    public void gemmTiled(
            TaskGraph layer,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k) {
        int s = slices(m, n, k);
        if (k % (32 * s) != 0) {
            layer.task(task, Qwen35MMAKernels::gemmMMATiledB, context, a, b, c, m, n, k);
            return;
        }
        if (s == 1) {
            layer.task(task, Qwen35MMAKernels::gemmMMATiledBStaged, context, a, b, c, m, n, k, 1);
            return;
        }
        layer.task(task, Qwen35MMAKernels::gemmMMATiledBStaged, context, a, b, partial, m, n, k, s);
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
