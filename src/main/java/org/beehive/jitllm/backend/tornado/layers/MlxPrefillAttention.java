package org.beehive.jitllm.backend.tornado.layers;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;

// @formatter:off
/**
 * Batched-prefill attention as MLX's fused causal attention, for any family whose layer runs RoPE
 * into the paged key/value cache and then attends from {@code wrapQBatch} into {@code wrapXbBatch}.
 *
 * <p>After RoPE has written the chunk's keys and values, each layer gathers its keys and values of
 * every position so far from the paged cache into contiguous FP16 {@code [head][position][headDim]}
 * buffers, packs the rotated queries the same way, runs {@link Mlx#scaledDotProductAttentionCached}
 * (which reads the context length from the start position when the task runs, so one graph serves
 * every chunk) and unpacks the output into {@code wrapXbBatch} for the output projection.
 *
 * <p>The buffers are shared by every layer graph: the first declares them, each later one binds
 * them from its predecessor.
 */
// @formatter:on
public final class MlxPrefillAttention {

    private final State state;
    private final KernelContext context;
    private final int batchSize;
    private final int heads;
    private final int kvHeads;
    private final int headDim;
    private final int qDim;
    private final int kvDim;
    private final float scale;

    /** Positions the gathered keys and values hold: the context, plus one padded chunk. */
    private final int positions;

    private final HalfFloatArray q;
    private final HalfFloatArray k;
    private final HalfFloatArray v;
    private final HalfFloatArray out;

    /**
     * The helper for a batched prefill that runs on MLX ({@link MlxPrefill#resolve}), or null when
     * {@code mlx} is empty and the family's own attention runs.
     */
    public static MlxPrefillAttention ifPresent(
            Optional<MlxPrefill> mlx,
            State state,
            KernelContext context,
            int batchSize,
            int contextLength,
            int heads,
            int kvHeads,
            int headDim) {
        return ifPresent(
                mlx,
                state,
                context,
                batchSize,
                contextLength,
                heads,
                kvHeads,
                headDim,
                (float) (1.0 / Math.sqrt(headDim)));
    }

    /**
     * {@link #ifPresent(Optional, State, KernelContext, int, int, int, int, int)} with its own
     * scale.
     */
    public static MlxPrefillAttention ifPresent(
            Optional<MlxPrefill> mlx,
            State state,
            KernelContext context,
            int batchSize,
            int contextLength,
            int heads,
            int kvHeads,
            int headDim,
            float scale) {
        if (mlx.isEmpty()) {
            return null;
        }
        return new MlxPrefillAttention(
                state, context, batchSize, contextLength, heads, kvHeads, headDim, scale);
    }

    public MlxPrefillAttention(
            State state,
            KernelContext context,
            int batchSize,
            int contextLength,
            int heads,
            int kvHeads,
            int headDim,
            float scale) {
        this.state = state;
        this.context = context;
        this.batchSize = batchSize;
        this.heads = heads;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.qDim = heads * headDim;
        this.kvDim = kvHeads * headDim;
        this.scale = scale;
        this.positions = contextLength + batchSize;
        this.q = new HalfFloatArray(batchSize * qDim);
        this.k = new HalfFloatArray(positions * kvDim);
        this.v = new HalfFloatArray(positions * kvDim);
        this.out = new HalfFloatArray(batchSize * qDim);
    }

    /**
     * Declares the buffers in layer graph {@code layerIndex}: uploaded by the first, bound after.
     */
    public void declare(TaskGraph layer, int layerIndex, String predecessor) {
        declare(layer, layerIndex == 0, predecessor);
    }

    /**
     * Declares the buffers in a layer graph: uploaded when it is the {@code first} that attends,
     * otherwise bound from graph {@code source}, which declared them already.
     */
    public void declare(TaskGraph layer, boolean first, String source) {
        if (first) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, q, k, v, out);
        } else {
            layer.consumeFromDevice(source, q, k, v, out);
        }
    }

    /**
     * The attention of layer {@code layerIndex} as task {@code task}, after the layer's RoPE task
     * has rotated {@code wrapQBatch} and written the chunk's keys and values to the cache.
     */
    public void attention(TaskGraph layer, String task, int layerIndex) {
        attention(layer, task, layerIndex, state.workspace.wrapQBatch, state.workspace.wrapXbBatch);
    }

    /**
     * {@link #attention(TaskGraph, String, int)} for a family whose rotated queries are in {@code
     * queries}, whose output goes to {@code output}, and whose layer addresses the cache as layer
     * {@code cacheLayer} — a family where only some layers hold a cache counts those densely.
     */
    public void attention(
            TaskGraph layer, String task, int cacheLayer, FloatArray queries, FloatArray output) {
        if (state.usesFp16KeyValueCache()) {
            layer.task(
                    task + "_gather_kv",
                    MlxPrefillKernels::gatherKvFP16,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    k,
                    v,
                    state.workspace.wrapBlockTable,
                    cacheLayer,
                    kvDim,
                    headDim,
                    positions,
                    batchSize,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    positions * kvDim);
        } else {
            layer.task(
                    task + "_gather_kv",
                    MlxPrefillKernels::gatherKvFP32,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    k,
                    v,
                    state.workspace.wrapBlockTable,
                    cacheLayer,
                    kvDim,
                    headDim,
                    positions,
                    batchSize,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    positions * kvDim);
        }
        layer.task(
                task + "_pack_q",
                MlxPrefillKernels::packHeads,
                context,
                queries,
                q,
                heads,
                batchSize,
                headDim,
                batchSize * qDim);
        layer.libraryTask(
                task,
                Mlx::scaledDotProductAttentionCached,
                q,
                k,
                v,
                out,
                state.workspace.batchStartPosHolder,
                1,
                heads,
                kvHeads,
                batchSize,
                positions,
                headDim,
                scale,
                true);
        layer.task(
                task + "_unpack",
                MlxPrefillKernels::unpackHeadsToFloat,
                context,
                out,
                output,
                heads,
                batchSize,
                headDim,
                batchSize * qDim);
    }

    /** Keeps the buffers on the device for the next layer graph. */
    public void persist(TaskGraph layer) {
        layer.persistOnDevice(q, k, v, out);
    }

    /** The grids of {@link #attention}'s kernels, under graph-qualified prefix {@code p}. */
    public void addGrids(GridScheduler scheduler, String p, String task) {
        scheduler.addWorkerGrid(p + task + "_gather_kv", elementwise(positions * kvDim));
        scheduler.addWorkerGrid(p + task + "_pack_q", elementwise(batchSize * qDim));
        scheduler.addWorkerGrid(p + task + "_unpack", elementwise(batchSize * qDim));
    }

    private static uk.ac.manchester.tornado.api.WorkerGrid elementwise(int total) {
        return WorkerGridFactory.genericWorker((total + 255) / 256 * 256, 256);
    }
}
