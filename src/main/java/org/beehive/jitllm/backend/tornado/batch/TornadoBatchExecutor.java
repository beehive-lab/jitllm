package org.beehive.jitllm.backend.tornado.batch;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kv.TornadoKvStore;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LlamaFP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.batch.BatchExecutor;
import org.beehive.jitllm.runtime.batch.BatchSlots;
import org.beehive.jitllm.runtime.kv.KvStorage;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The batched decode plan #129 proved, as a production executor the engine can drive.
 *
 * <p><b>What was promoted, and what was not.</b> {@code bench/BatchedDecodeEngine} contained two
 * things wound together: a batched decode <i>plan</i>, and a benchmark harness that drove it with
 * its own admission, its own free list and its own prompt handling. Only the plan belongs in
 * production — the harness's scheduling is what {@code engine.Scheduler} now is, and duplicating it
 * here would have given the codebase two admission policies to disagree with each other.
 *
 * <p>So this class owns exactly: the task graphs (activation, N decode layers, batched logits), the
 * per-step buffers, and the translation from {@link BatchSlots} to what the kernels read. The bench
 * keeps its own copy and its own numbers; nothing was deleted from it.
 *
 * <p><b>Slot identity.</b> The batched kernels index the block table at {@code batchIndex *
 * blocksPerSlot + …}, so a request's batch position <b>is</b> its KV slot. The scheduler follows
 * the lease rather than allocating a second index, and this executor asserts the two agree rather
 * than trusting them to.
 *
 * <p>FP16 weights and greedy on-device sampling only, matching what #129 demonstrated. Q8_0 and
 * temperature sampling are extensions, not omissions to be papered over.
 */
public final class TornadoBatchExecutor implements BatchExecutor, AutoCloseable {

    private static final int RMS_LOCAL = 256;

    private final int batchSize;
    private final int dim;
    private final Set<Integer> stopTokens;

    private final TornadoExecutionPlan plan;
    private final GridScheduler gridScheduler;
    private final int layerCount;
    private final int logitsGraphIndex;

    private final IntArray seqPositions;
    private final IntArray sampledTokens;
    private final MemorySegment embeddingTable;
    private final MemorySegment embeddingBatch;
    private final long dimBytes;
    private final TornadoKvStore store;
    private final int blocksPerSlot;

    private boolean closed;

    /**
     * The continuous-batch plan compiles lazily on the first step, independently of master plans.
     */
    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        var backend =
                uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime()
                        .getBackend(0);
        var module = TornadoExecutionPlan.class.getModule().getDescriptor();
        String version = module == null ? "unknown" : module.rawVersion().orElse("unknown");
        return new org.beehive.jitllm.runtime.backend.ExecutionInfo(
                backend.getBackendType().name(),
                backend.getDefaultDevice().getPhysicalDevice().getDeviceName(),
                "TornadoVM " + version,
                "continuous-batch-decode",
                1,
                store.keyPoolFP16() == null ? "FP32" : "FP16",
                "FP16 tensor-core MMA",
                "JIT kernels",
                false,
                false);
    }

    /**
     * @param model the model to decode with — FP16 Llama or Qwen3
     * @param state a state built against the engine's lease-backed KV, sized for this batch
     * @param store the shared KV store the engine's manager owns; its table is what the kernels
     *     walk
     * @param batchSize B, fixed here
     * @param blocksPerSlot table entries per slot, matching the pool
     */
    public TornadoBatchExecutor(
            Model model, State state, KvStorage storage, int batchSize, int blocksPerSlot) {
        Configuration config = model.configuration();
        TornadoWeights weights = (TornadoWeights) model.weights();
        boolean isQwen3 = config instanceof Qwen3Configuration;

        if (!org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend()) {
            throw new IllegalArgumentException(
                    "Continuous batching requires a CUDA device with tensor-core MMA support");
        }
        if (!(state instanceof LlamaState) && !(state instanceof Qwen3State)) {
            throw new IllegalArgumentException(
                    "Continuous batching supports FP16 Llama/Qwen3 only");
        }
        this.batchSize = batchSize;
        this.blocksPerSlot = blocksPerSlot;
        if (!(storage instanceof TornadoKvStore store)) {
            throw new IllegalStateException(
                    "the Tornado batch executor needs TornadoVM-backed"
                            + " key/value storage, got "
                            + storage.getClass().getName());
        }
        this.store = store;
        this.dim = config.dim();
        this.stopTokens = model.chatFormat().getStopTokens();

        int vocab = config.vocabularySize();
        int paddedB = (batchSize + 127) & ~127;

        this.seqPositions = new IntArray(batchSize);
        this.sampledTokens = new IntArray(paddedB);
        FloatArray finalScaleBatch = new FloatArray(batchSize);
        HalfFloatArray normedFinalFP16 = new HalfFloatArray(paddedB * dim);
        normedFinalFP16.init(new HalfFloat(0.0f));
        FloatArray logitsBatch = new FloatArray(paddedB * vocab);

        BatchPrefillActivation activation =
                new BatchPrefillActivation(state, config, batchSize, false);

        List<ImmutableTaskGraph> layerGraphs;
        String lastLayerId;
        Consumer<GridScheduler> updateLayerSchedule;
        if (isQwen3) {
            Qwen3FP16LayersBatchDecodeMMA layers =
                    new Qwen3FP16LayersBatchDecodeMMA(
                            (Qwen3State) state,
                            (Qwen3TornadoWeights) weights,
                            (Qwen3Configuration) config,
                            batchSize,
                            blocksPerSlot * store.blockSizeTokens(),
                            store.keyPool() != null ? store.keyPool() : store.keyPoolFP16(),
                            store.valuePool() != null ? store.valuePool() : store.valuePoolFP16(),
                            seqPositions,
                            store.blockTable(),
                            store.blockSizeTokens(),
                            blocksPerSlot);
            layerGraphs = layers.getLayerImmutableTaskGraphs();
            lastLayerId = layers.getLastLayerTaskGraphID();
            updateLayerSchedule = layers::updateGridScheduler;
        } else {
            LlamaFP16LayersBatchDecodeMMA layers =
                    new LlamaFP16LayersBatchDecodeMMA(
                            (LlamaState) state,
                            (LlamaTornadoWeights) weights,
                            (LlamaConfiguration) config,
                            batchSize,
                            blocksPerSlot * store.blockSizeTokens(),
                            store.keyPool() != null ? store.keyPool() : store.keyPoolFP16(),
                            store.valuePool() != null ? store.valuePool() : store.valuePoolFP16(),
                            seqPositions,
                            store.blockTable(),
                            store.blockSizeTokens(),
                            blocksPerSlot);
            layerGraphs = layers.getLayerImmutableTaskGraphs();
            lastLayerId = layers.getLastLayerTaskGraphID();
            updateLayerSchedule = layers::updateGridScheduler;
        }

        KernelContext logitsContext = new KernelContext();
        HalfFloatArray wclsHalf = weights.wclsByteArray.asHalfFloatArray();
        FloatArray rmsFinal = weights.rms_final_weight_as_floatArray.asFloatArray();
        // Greedy on-device argmax: only B token ids cross to the host, not the whole logits
        // tensor, which is tens of megabytes per step.
        TaskGraph logits =
                new TaskGraph("batchLogits")
                        .consumeFromDevice(lastLayerId, state.workspace.wrapXBatch)
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                logitsContext,
                                normedFinalFP16,
                                finalScaleBatch,
                                wclsHalf,
                                rmsFinal,
                                sampledTokens)
                        .task(
                                "rms_reduce",
                                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                                logitsContext,
                                state.workspace.wrapXBatch,
                                finalScaleBatch,
                                dim,
                                config.rmsNormEps(),
                                RMS_LOCAL)
                        .task(
                                "rms_apply",
                                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                                logitsContext,
                                normedFinalFP16,
                                state.workspace.wrapXBatch,
                                rmsFinal,
                                finalScaleBatch,
                                dim)
                        .task(
                                "vocab",
                                TransformerBatchPrefillKernels::gemmMMA,
                                logitsContext,
                                normedFinalFP16,
                                wclsHalf,
                                logitsBatch,
                                paddedB,
                                vocab,
                                dim)
                        .task(
                                "argmax",
                                TransformerBatchPrefillKernels::batchedArgmaxLogits,
                                logitsContext,
                                logitsBatch,
                                sampledTokens,
                                vocab)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, sampledTokens);

        GridScheduler schedule = new GridScheduler();
        activation.updateGridScheduler(schedule);
        updateLayerSchedule.accept(schedule);
        schedule.addWorkerGrid("batchLogits.rms_reduce", worker(batchSize * RMS_LOCAL, RMS_LOCAL));
        schedule.addWorkerGrid("batchLogits.rms_apply", worker(batchSize * dim, 256));
        schedule.addWorkerGrid("batchLogits.vocab", mmaGrid(paddedB, vocab));
        schedule.addWorkerGrid("batchLogits.argmax", worker(batchSize * 256, 256));
        this.gridScheduler = schedule;

        List<ImmutableTaskGraph> all = new ArrayList<>();
        all.add(activation.getImmutableTaskGraph());
        all.addAll(layerGraphs);
        all.add(logits.snapshot());
        this.layerCount = layerGraphs.size();
        this.logitsGraphIndex = 1 + layerCount;
        this.plan = new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]));
        var roles = new java.util.HashMap<Integer, String>();
        roles.put(0, "batch activation");
        org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.label(
                roles, 1, layerCount, "layers");
        roles.put(logitsGraphIndex, "logits and sampling");
        org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.printIfRequested(
                new org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.Chain(
                        "continuous batching (" + batchSize + " slots)",
                        all,
                        schedule,
                        List.of(
                                new org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.Phase(
                                        "engine step",
                                        "per engine step, all slots together",
                                        org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter
                                                .span(0, logitsGraphIndex))),
                        roles),
                model);

        this.embeddingTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
        this.embeddingBatch = state.workspace.embeddingXBatch.getSegment();
        this.dimBytes = (long) dim * Short.BYTES;
    }

    @Override
    public int maxBatchSize() {
        return batchSize;
    }

    @Override
    public int[] decodeStep(BatchSlots batch) {
        if (closed) {
            throw new IllegalStateException("this executor is closed");
        }
        if (batch.width() != batchSize) {
            throw new IllegalArgumentException(
                    "the batch is fixed at "
                            + batchSize
                            + " slots but "
                            + batch.width()
                            + " were given");
        }

        for (int slot = 0; slot < batchSize; slot++) {
            boolean active = batch.active()[slot];
            // An inactive slot still runs every kernel, so it needs a valid embedding and a valid
            // position. Its KV writes land in the pool's scratch block, which is what the block
            // table publishes for an unmapped slot.
            int token = active ? batch.tokens()[slot] : 0;
            MemorySegment.copy(
                    embeddingTable,
                    token * dimBytes,
                    embeddingBatch,
                    (long) slot * dimBytes,
                    dimBytes);
            seqPositions.set(slot, active ? batch.positions()[slot] : 0);

            if (active && batch.kvSlots()[slot] != slot) {
                throw new IllegalStateException(
                        "batch position "
                                + slot
                                + " holds a lease at KV"
                                + " slot "
                                + batch.kvSlots()[slot]
                                + ". The kernels index the block table"
                                + " by batch position, so the two must be the same number — otherwise this"
                                + " slot reads another sequence's mapping, and both indices are valid");
            }
        }

        plan.withGraph(0).withGridScheduler(gridScheduler).execute();
        for (int layer = 0; layer < layerCount; layer++) {
            plan.withGraph(1 + layer).withGridScheduler(gridScheduler).execute();
        }
        plan.withGraph(logitsGraphIndex).withGridScheduler(gridScheduler).execute();

        int[] tokens = new int[batchSize];
        for (int slot = 0; slot < batchSize; slot++) {
            tokens[slot] = sampledTokens.get(slot);
        }
        return tokens;
    }

    @Override
    public boolean isStopToken(int token) {
        return stopTokens.contains(token);
    }

    /** The store whose table this executor's kernels walk. */
    public TornadoKvStore store() {
        return store;
    }

    public int blocksPerSlot() {
        return blocksPerSlot;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            plan.freeDeviceMemory();
        }
    }

    private static WorkerGrid worker(int global, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    private static WorkerGrid mmaGrid(int paddedM, int n) {
        WorkerGrid2D grid = new WorkerGrid2D(paddedM / 128 * 256, n / 128);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }
}
