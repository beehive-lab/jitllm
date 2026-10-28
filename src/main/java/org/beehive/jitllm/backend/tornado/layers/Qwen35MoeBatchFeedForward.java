package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;

// @formatter:off
/**
 * The {@code qwen35moe} feed-forward of one layer over a prompt chunk, in place of the dense one in
 * {@link Qwen35BatchPrefillLayers}: from the normalized and quantized chunk into {@code
 * wrapXBatch}.
 *
 * <p>Batched routing, the assignments grouped by expert, the grouped expert GEMMs, the shared
 * expert as three ordinary int8 GEMMs over the chunk, and the weighted combine into the residual
 * stream. See {@link Qwen35MoeBatchKernels}. The layer graphs own the graph structure; this class
 * adds the tasks, names the weights and scratch they bind, and registers the tasks' grids.
 *
 * <p>Experts that keep the file's own representation (a {@code qwen2moe} Q4_0, Q4_1 or Q6_K file's)
 * run the grouped GEMMs for their type. A shared expert in such a representation runs them too, as
 * one expert over the whole chunk ({@code wrapMoeSharedTiles}, {@code wrapMoeSharedRows}), in place
 * of the Q8_0 GEMMs; a Q8_0 one keeps them.
 */
// @formatter:on
public final class Qwen35MoeBatchFeedForward implements Qwen35BatchFeedForward {

    /** The int8 GEMM's output tile, which the shared expert's widths must fill. */
    private static final int GEMM_TILE = 128;

    private final TornadoWorkspace workspace;
    private final Qwen35Experts experts;
    private final Qwen35ExpertWeights<TornadoTensor> tensors;
    private final int dim;
    private final int batchSize;

    /**
     * @param experts the experts' shape
     * @param tensors their tensors, per layer
     * @param dim the model width
     * @param batchSize the chunk width
     */
    public Qwen35MoeBatchFeedForward(
            TornadoWorkspace workspace,
            Qwen35Experts experts,
            Qwen35ExpertWeights<TornadoTensor> tensors,
            int dim,
            int batchSize) {
        if (experts.count() > Qwen35MoeBatchKernels.GROUP_THREADS
                || batchSize % Qwen35MoeBatchKernels.ROUTER_TILE != 0) {
            throw new UnsupportedOperationException(
                    "the batched expert path needs at most 1024 experts and a chunk width that is a"
                            + " multiple of 64");
        }
        this.workspace = workspace;
        this.experts = experts;
        this.tensors = tensors;
        this.dim = dim;
        this.batchSize = batchSize;
    }

    /** Adds layer {@code layerIndex}'s feed-forward tasks. */
    @Override
    public void appendTasks(TaskGraph layer, KernelContext context, int layerIndex) {
        appendRouter(layer, context, layerIndex);
        appendSharedGateInput(layer, context, layerIndex);
        appendTopK(layer, context);
        appendExperts(layer, context, layerIndex);
    }

    /** The router's logits for every token, tiled: the expert count a multiple of 64. */
    public void appendRouter(TaskGraph layer, KernelContext context, int layerIndex) {
        if (experts.count() % Qwen35MoeBatchKernels.ROUTER_TILE != 0) {
            throw new UnsupportedOperationException(
                    "the tiled router needs an expert count that is a multiple of 64");
        }
        var ws = workspace;
        layer.task(
                "moe_router",
                Qwen35MoeBatchKernels::routerTiled,
                context,
                ws.wrapNormedBatch,
                tensors.router()[layerIndex].asFloatArray(),
                ws.wrapMoeLogitsBatch,
                ws.batchStartPosHolder,
                dim,
                experts.count());
    }

    /** Every token's shared-expert gate. */
    public void appendSharedGateInput(TaskGraph layer, KernelContext context, int layerIndex) {
        var ws = workspace;
        layer.task(
                "moe_shared_gate_input",
                Qwen35MoeBatchKernels::sharedGateBatch,
                context,
                ws.wrapNormedBatch,
                tensors.sharedGateInput()[layerIndex].asFloatArray(),
                ws.wrapMoeSharedGateBatch,
                ws.batchStartPosHolder,
                dim);
    }

    /** Every token's experts and their weights, renormalized over the chosen ones. */
    public void appendTopK(TaskGraph layer, KernelContext context) {
        var ws = workspace;
        layer.task(
                "moe_topk",
                Qwen35MoeBatchKernels::routerTopKBatch,
                context,
                ws.wrapMoeLogitsBatch,
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                ws.batchStartPosHolder,
                experts.count(),
                experts.used());
    }

    /**
     * From the routed ids and weights on: the assignments grouped by expert, the grouped expert
     * GEMMs, the shared expert, and the weighted combine into the residual stream.
     */
    public void appendExperts(TaskGraph layer, KernelContext context, int layerIndex) {
        var ws = workspace;
        int hidden = experts.hiddenDim();
        int shared = experts.sharedHiddenDim();
        layer.task(
                "moe_group",
                Qwen35MoeBatchKernels::groupByExpert,
                context,
                ws.wrapMoeIdsBatch,
                ws.batchStartPosHolder,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                experts.count(),
                experts.used(),
                batchSize);
        TornadoTensor gate = tensors.gateExperts()[layerIndex];
        requireSameType(gate, tensors.upExperts()[layerIndex], "routed");
        layer.task(
                "moe_gate_up",
                switch (gate.dataType()) {
                    case Q8_0 -> Qwen35MoeBatchKernels::groupedGateUpQ8_0;
                    case Q4_0 -> Qwen35MoeBatchKernels::groupedGateUpQ4_0;
                    case Q4_1 -> Qwen35MoeBatchKernels::groupedGateUpQ4_1;
                    case Q4_K -> Qwen35MoeBatchKernels::groupedGateUpQ4_K;
                    case Q5_K -> Qwen35MoeBatchKernels::groupedGateUpQ5_K;
                    case Q6_K -> Qwen35MoeBatchKernels::groupedGateUpQ6_K;
                    default -> throw unsupported(gate, "routed gate and up");
                },
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                ws.wrapMoeSortedToken,
                ws.wrapMoeTiles,
                tensors.gateExperts()[layerIndex].asByteArray(),
                tensors.upExperts()[layerIndex].asByteArray(),
                ws.wrapMoeHiddenBatch,
                dim,
                hidden);
        layer.task(
                "moe_hidden_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                hidden);
        TornadoTensor down = tensors.downExperts()[layerIndex];
        layer.task(
                "moe_down",
                switch (down.dataType()) {
                    case Q8_0 -> Qwen35MoeBatchKernels::groupedDownQ8_0;
                    case Q4_0 -> Qwen35MoeBatchKernels::groupedDownQ4_0;
                    case Q4_1 -> Qwen35MoeBatchKernels::groupedDownQ4_1;
                    case Q4_K -> Qwen35MoeBatchKernels::groupedDownQ4_K;
                    case Q5_K -> Qwen35MoeBatchKernels::groupedDownQ5_K;
                    case Q6_K -> Qwen35MoeBatchKernels::groupedDownQ6_K;
                    default -> throw unsupported(down, "routed down");
                },
                context,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                ws.wrapMoeTiles,
                tensors.downExperts()[layerIndex].asByteArray(),
                ws.wrapMoeOutBatch,
                dim,
                hidden);
        // The shared expert, over every row of the chunk.
        if (nativeShared(layerIndex)) {
            appendNativeSharedExpert(layer, context, layerIndex);
        } else {
            appendSharedExpert(layer, context, layerIndex);
        }
        layer.task(
                "moe_combine",
                Qwen35MoeBatchKernels::combine,
                context,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedGateBatch,
                ws.wrapXBatch,
                ws.batchStartPosHolder,
                dim,
                experts.used());
    }

    /** The shared expert over Q8_0 weights: three int8 GEMMs over the chunk. */
    private void appendSharedExpert(TaskGraph layer, KernelContext context, int layerIndex) {
        var ws = workspace;
        int shared = experts.sharedHiddenDim();
        layer.task(
                "moe_shared_gate",
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                tensors.sharedGate()[layerIndex].asByteArray(),
                ws.wrapMoeSharedGateUp,
                ws.wrapMoeSharedGateUp,
                batchSize,
                shared,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                ws.wrapMoeSharedGateUp,
                1,
                ws.batchStartPosHolder,
                shared,
                0);
        layer.task(
                "moe_shared_up",
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                tensors.sharedUp()[layerIndex].asByteArray(),
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedGateUp,
                batchSize,
                shared,
                dim,
                Int8GemmKernels.EPILOGUE_SWIGLU,
                ws.wrapMoeSharedHidden,
                1,
                ws.batchStartPosHolder,
                shared,
                0);
        layer.task(
                "moe_shared_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                shared);
        layer.task(
                "moe_shared_down",
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                tensors.sharedDown()[layerIndex].asByteArray(),
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedOut,
                batchSize,
                dim,
                shared,
                Int8GemmKernels.EPILOGUE_STORE,
                ws.wrapMoeSharedOut,
                1,
                ws.batchStartPosHolder,
                dim,
                0);
    }

    /**
     * The shared expert over weights in their own representation: the routed experts' grouped
     * GEMMs, as one expert whose tiles cover the chunk in token order.
     */
    private void appendNativeSharedExpert(TaskGraph layer, KernelContext context, int layerIndex) {
        var ws = workspace;
        int shared = experts.sharedHiddenDim();
        TornadoTensor gate = tensors.sharedGate()[layerIndex];
        TornadoTensor down = tensors.sharedDown()[layerIndex];
        requireSameType(gate, tensors.sharedUp()[layerIndex], "shared");
        layer.task(
                "moe_shared_gate_up",
                switch (gate.dataType()) {
                    case Q8_0 -> Qwen35MoeBatchKernels::groupedGateUpQ8_0;
                    case Q4_0 -> Qwen35MoeBatchKernels::groupedGateUpQ4_0;
                    case Q4_1 -> Qwen35MoeBatchKernels::groupedGateUpQ4_1;
                    case Q4_K -> Qwen35MoeBatchKernels::groupedGateUpQ4_K;
                    case Q5_K -> Qwen35MoeBatchKernels::groupedGateUpQ5_K;
                    case Q6_K -> Qwen35MoeBatchKernels::groupedGateUpQ6_K;
                    default -> throw unsupported(gate, "shared gate and up");
                },
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                ws.wrapMoeSharedRows,
                ws.wrapMoeSharedTiles,
                gate.asByteArray(),
                tensors.sharedUp()[layerIndex].asByteArray(),
                ws.wrapMoeSharedHidden,
                dim,
                shared);
        layer.task(
                "moe_shared_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                shared);
        layer.task(
                "moe_shared_down",
                switch (down.dataType()) {
                    case Q8_0 -> Qwen35MoeBatchKernels::groupedDownQ8_0;
                    case Q4_0 -> Qwen35MoeBatchKernels::groupedDownQ4_0;
                    case Q4_1 -> Qwen35MoeBatchKernels::groupedDownQ4_1;
                    case Q4_K -> Qwen35MoeBatchKernels::groupedDownQ4_K;
                    case Q5_K -> Qwen35MoeBatchKernels::groupedDownQ5_K;
                    case Q6_K -> Qwen35MoeBatchKernels::groupedDownQ6_K;
                    default -> throw unsupported(down, "shared down");
                },
                context,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                ws.wrapMoeSharedTiles,
                down.asByteArray(),
                ws.wrapMoeSharedOut,
                dim,
                shared);
    }

    /** Whether layer {@code layerIndex}'s shared expert holds a weight other than Q8_0. */
    private boolean nativeShared(int layerIndex) {
        return tensors.sharedGate()[layerIndex].dataType() != DataType.Q8_0
                || tensors.sharedUp()[layerIndex].dataType() != DataType.Q8_0
                || tensors.sharedDown()[layerIndex].dataType() != DataType.Q8_0;
    }

    private boolean anyNativeShared() {
        for (int i = 0; i < tensors.sharedGate().length; i++) {
            if (nativeShared(i)) {
                return true;
            }
        }
        return false;
    }

    /** A gate and an up projection one fused GEMM decodes: they must share a type. */
    private static void requireSameType(TornadoTensor gate, TornadoTensor up, String which) {
        if (gate.dataType() != up.dataType()) {
            throw new UnsupportedOperationException(
                    "the batched "
                            + which
                            + " experts' gate and up projections are "
                            + gate.dataType()
                            + " and "
                            + up.dataType()
                            + ": one grouped GEMM decodes both, so they must share a type");
        }
    }

    private static IllegalStateException unsupported(TornadoTensor w, String which) {
        return new IllegalStateException(
                "no grouped tensor-core GEMM for " + which + " experts of type " + w.dataType());
    }

    /** Layer {@code layerIndex}'s expert tensors, in place of the dense gate, up and down. */
    @Override
    public List<Object> layerWeights(int layerIndex) {
        return List.of(
                tensors.router()[layerIndex],
                tensors.sharedGateInput()[layerIndex],
                tensors.gateExperts()[layerIndex],
                tensors.upExperts()[layerIndex],
                tensors.downExperts()[layerIndex],
                tensors.sharedGate()[layerIndex],
                tensors.sharedUp()[layerIndex],
                tensors.sharedDown()[layerIndex]);
    }

    /** The chunk's routing and expert scratch, bound by every layer graph in turn. */
    @Override
    public Object[] scratch() {
        var ws = workspace;
        Object[] expertScratch = {
            ws.wrapMoeLogitsBatch,
            ws.wrapMoeIdsBatch,
            ws.wrapMoeWeightsBatch,
            ws.wrapMoeSharedGateBatch,
            ws.wrapMoeSortedToken,
            ws.wrapMoePosition,
            ws.wrapMoeTiles,
            ws.wrapMoeHiddenBatch,
            ws.wrapMoeHiddenQ8,
            ws.wrapMoeHiddenScales,
            ws.wrapMoeOutBatch,
            ws.wrapMoeSharedGateUp,
            ws.wrapMoeSharedHidden,
            ws.wrapMoeSharedQ8,
            ws.wrapMoeSharedScales,
            ws.wrapMoeSharedOut
        };
        if (!anyNativeShared()) {
            return expertScratch;
        }
        Object[] scratch = java.util.Arrays.copyOf(expertScratch, expertScratch.length + 2);
        scratch[expertScratch.length] = ws.wrapMoeSharedTiles;
        scratch[expertScratch.length + 1] = ws.wrapMoeSharedRows;
        return scratch;
    }

    /** Grids of one layer's feed-forward tasks, whose qualified names start with {@code prefix}. */
    @Override
    public void addWorkerGrids(GridScheduler scheduler, String prefix) {
        addRouterAndTopKGrids(scheduler, prefix);
        addSharedGateInputGrid(scheduler, prefix);
        addExpertGrids(scheduler, prefix);
    }

    /** Grids of {@link #appendRouter} and {@link #appendTopK}. */
    public void addRouterAndTopKGrids(GridScheduler scheduler, String prefix) {
        WorkerGrid router =
                new WorkerGrid2D(
                        batchSize / Qwen35MoeBatchKernels.ROUTER_TILE * 256,
                        experts.count() / Qwen35MoeBatchKernels.ROUTER_TILE);
        router.setLocalWork(256, 1, 1);
        scheduler.addWorkerGrid(prefix + "moe_router", router);
        scheduler.addWorkerGrid(
                prefix + "moe_topk", WorkerGridFactory.genericWorker(batchSize * 32, 32));
    }

    /** Grid of {@link #appendSharedGateInput}. */
    public void addSharedGateInputGrid(GridScheduler scheduler, String prefix) {
        scheduler.addWorkerGrid(
                prefix + "moe_shared_gate_input",
                WorkerGridFactory.genericWorker(batchSize * 32, 128));
    }

    /**
     * Grids of {@link #appendExperts} in layer {@code layerIndex}, whose shared expert may differ.
     */
    public void addExpertGrids(GridScheduler scheduler, String prefix, int layerIndex) {
        if (!nativeShared(layerIndex)) {
            addExpertGrids(scheduler, prefix);
            return;
        }
        addRoutedExpertGrids(scheduler, prefix);
        int sharedTiles =
                (batchSize + Qwen35MoeBatchKernels.TILE_ROWS - 1) / Qwen35MoeBatchKernels.TILE_ROWS;
        WorkerGrid gateUp =
                new WorkerGrid2D(
                        experts.sharedHiddenDim()
                                / Qwen35MoeBatchKernels.TILE_COLS
                                * Qwen35MoeBatchKernels.GEMM_THREADS,
                        sharedTiles);
        gateUp.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        scheduler.addWorkerGrid(prefix + "moe_shared_gate_up", gateUp);
        scheduler.addWorkerGrid(
                prefix + "moe_shared_q8",
                WorkerGridFactory.genericWorker(batchSize * experts.sharedHiddenDim(), 128));
        WorkerGrid down =
                new WorkerGrid2D(
                        dim / Qwen35MoeBatchKernels.TILE_COLS * Qwen35MoeBatchKernels.GEMM_THREADS,
                        sharedTiles);
        down.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        scheduler.addWorkerGrid(prefix + "moe_shared_down", down);
    }

    /** Grids of {@link #appendExperts}, with the shared expert's Q8_0 GEMMs. */
    public void addExpertGrids(GridScheduler scheduler, String prefix) {
        addRoutedExpertGrids(scheduler, prefix);
        int shared = experts.sharedHiddenDim();
        scheduler.addWorkerGrid(
                prefix + "moe_shared_q8", WorkerGridFactory.genericWorker(batchSize * shared, 128));
        int local = Int8GemmKernels.Q8_GEMM_THREADS;
        addGemmGrid(scheduler, prefix + "moe_shared_gate", shared, local);
        addGemmGrid(scheduler, prefix + "moe_shared_up", shared, local);
        addGemmGrid(scheduler, prefix + "moe_shared_down", dim, local);
    }

    /** Grids of the grouping, the routed GEMMs, their activation's quantization and the combine. */
    private void addRoutedExpertGrids(GridScheduler scheduler, String prefix) {
        int assignments = batchSize * experts.used();
        int shared = experts.sharedHiddenDim();
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, experts.count());
        scheduler.addWorkerGrid(
                prefix + "moe_group",
                WorkerGridFactory.genericWorker(
                        Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS));
        WorkerGrid gateUp =
                new WorkerGrid2D(
                        experts.hiddenDim()
                                / Qwen35MoeBatchKernels.TILE_COLS
                                * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        gateUp.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        scheduler.addWorkerGrid(prefix + "moe_gate_up", gateUp);
        WorkerGrid down =
                new WorkerGrid2D(
                        dim / Qwen35MoeBatchKernels.TILE_COLS * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        down.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        scheduler.addWorkerGrid(prefix + "moe_down", down);
        scheduler.addWorkerGrid(
                prefix + "moe_hidden_q8",
                WorkerGridFactory.genericWorker(assignments * experts.hiddenDim(), 128));
        scheduler.addWorkerGrid(
                prefix + "moe_combine", WorkerGridFactory.genericWorker(batchSize * dim, 256));
    }

    /** An int8 GEMM's grid: a workgroup per output tile of the chunk. */
    private void addGemmGrid(GridScheduler scheduler, String task, int outputs, int local) {
        WorkerGrid gemm = new WorkerGrid2D((batchSize / GEMM_TILE) * local, outputs / GEMM_TILE);
        gemm.setLocalWork(local, 1, 1);
        scheduler.addWorkerGrid(task, gemm);
    }
}
