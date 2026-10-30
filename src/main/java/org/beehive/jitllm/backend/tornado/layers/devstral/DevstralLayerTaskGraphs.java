package org.beehive.jitllm.backend.tornado.layers.devstral;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * What Devstral's quantized decode layers share whatever their weights hold: the buffers each layer
 * graph binds, RoPE over the precomputed (YaRN) tables into the key/value cache, and decode
 * attention.
 *
 * <p>Devstral's layer is Llama's with a Q wider than the model ({@code qDim = heads * headDim}, not
 * {@code dim}), so the projections are sized by {@code qDim} where Llama's use {@code dim}. The
 * selection rules are Llama's ({@code LlamaQ8_0FFNLayers}):
 *
 * <ul>
 *   <li>the projections read the activation quantized to Q8 blocks with packed integer dot products
 *       where {@link PackedActivationTasks#enabled()} holds (CUDA), and the floating-point kernels
 *       elsewhere;
 *   <li>attention over the FP16 cache on the NVIDIA path is split across several workgroups per
 *       head (flash-decoding) and merged by a combine pass, taking the lane-cooperative kernel
 *       where {@link LaneAttentionPolicy} has one for the head width. One workgroup per head
 *       otherwise, as before.
 * </ul>
 */
// @formatter:on
public abstract class DevstralLayerTaskGraphs
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, DevstralConfiguration> {

    /** Lanes per output row of the packed projections: a multiple of 32, as Llama's. */
    protected static final int PACKED_LOCAL = 64;

    protected DevstralLayerTaskGraphs(
            State state,
            LlamaTornadoWeights weights,
            DevstralConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    /**
     * Whether the projections run as packed integer dot products over the activation quantized to
     * Q8 blocks: the device lowers {@code dp4a} and the state carries the quantization scratch.
     */
    protected boolean packed() {
        return PackedActivationTasks.enabled() && state.workspace.wrapXbQuants != null;
    }

    /** The key cache in the representation the state holds: FP16 where it allocated one. */
    protected Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state holds. */
    protected Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    /** The graph whose {@code wrapX} this layer reads, or {@code null} for the unnamed consume. */
    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    /** The graph that has already uploaded this layer's weights, or {@code null} to upload here. */
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }

    /** {@code objects} consumed from {@code source}, or from the previous graph when it is null. */
    protected static void consume(TaskGraph graph, String source, Object... objects) {
        if (source != null) {
            graph.consumeFromDevice(source, objects);
        } else {
            graph.consumeFromDevice(objects);
        }
    }

    /**
     * Binds the scratch the selected kernels share across layers: the quantized activation of the
     * packed projections and the split-KV partial outputs. Uploaded once by layer 0 and passed
     * along the graph chain, so every layer addresses the same device buffers.
     */
    protected void bindSharedScratch(TaskGraph layer, int layerIndex) {
        java.util.List<Object> scratch = new java.util.ArrayList<>(4);
        if (packed()) {
            scratch.addAll(java.util.List.of(PackedActivationTasks.buffers(state.workspace)));
        }
        if (attentionSplits() > 0) {
            scratch.add(state.workspace.wrapAttSplit);
        }
        if (scratch.isEmpty()) {
            return;
        }
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch.toArray());
        } else {
            consume(layer, predecessorGraphName(layerIndex), scratch.toArray());
        }
    }

    // @formatter:off
    /** RoPE on Q and K over the precomputed (YaRN-scaled) tables, K and V appended to the cache. */
    protected void ropeAndKeyValueCache(TaskGraph layer, int layerIndex) {
        if (useFp16KVCache()) {
            layer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedFP16Paged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            layer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedPaged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
    }

    /**
     * Split-KV partitions per head, or 0 for one workgroup per head. Only over the FP16 cache on
     * the NVIDIA path, as Llama: by default where the lane-cooperative kernel runs, or as asked
     * with {@code -Djitllm.attention.splitKv=true} (and {@code .count}).
     */
    protected int attentionSplits() {
        if (!useFp16KVCache() || schedulerType != SchedulerType.NVIDIA) {
            return 0;
        }
        var partitions = state.executionPolicy().splitKvPartitions();
        if (partitions.isEmpty()) {
            return laneAttention(config.headSize()) ? State.SPLIT_KV : 0;
        }
        int splits = partitions.getAsInt();
        if (splits > State.SPLIT_KV) {
            throw new IllegalArgumentException(
                    "split-KV working partitions "
                            + splits
                            + " exceed the "
                            + State.SPLIT_KV
                            + " the attention scratch was sized for;"
                            + " raise jitllm.attention.splitKv.count, which is the capacity");
        }
        return splits;
    }

    /** Decode attention, its output in {@code wrapXb}. */
    protected void configureAttention(TaskGraph layer, int layerIndex) {
        int splits = attentionSplits();
        if (splits > 0) {
            // Flash-decoding: every head's positions split across several workgroups, merged by
            // a combine pass. More workgroups than heads, which is what keeps the device busy.
            LaneAttentionPolicy.Kernel kernel =
                    laneAttention(config.headSize())
                            ? LaneAttentionPolicy.kernel(config.headSize())
                            : packedHalf2Attention
                                    ? TransformerPagedKvKernels
                                            ::processHeadsFlashAttentionSplitKVFP16PackedPaged
                                    : TransformerPagedKvKernels
                                            ::processHeadsFlashAttentionSplitKVFP16Paged;
            layer.task(
                    "attention",
                    kernel,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapAttSplit,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    splits);
            layer.task(
                    "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    state.workspace.wrapAttSplit,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    splits);
            return;
        }
        if (schedulerType == SchedulerType.NVIDIA) {
            if (useFp16KVCache()) {
                layer.task(
                        "attention",
                        TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                        context,
                        state.workspace.wrapQ,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapXb,
                        config.numberOfHeads(),
                        config.headSize(),
                        config.kvDim(),
                        config.kvMul(),
                        state.workspace.positionHolder,
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride);
                return;
            }
            layer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            return;
        }
        layer.task(
                "attention",
                TransformerPagedKvKernels::processHeadsParallelPaged,
                state.workspace.wrapQ,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                state.workspace.wrapXb,
                config.numberOfHeads(),
                config.headSize(),
                config.kvDim(),
                config.kvMul(),
                config.contextLength(),
                state.workspace.positionHolder,
                state.workspace.wrapAtt,
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride);
    }

    // @formatter:on

    /** Registers the attention tasks' grids for {@code layerIndex}, matching the kernel chosen. */
    protected void addAttentionWorkerGrids(GridScheduler scheduler, int layerIndex) {
        String prefix = GraphNames.layer(layerIndex) + ".";
        int splits = attentionSplits();
        if (splits == 0) {
            scheduler.addWorkerGrid(
                    prefix + "attention",
                    WorkerGridFactory.createAttentionWorker(
                            config.numberOfHeads(), config.headSize()));
            return;
        }
        int groups = config.numberOfHeads() * splits;
        scheduler.addWorkerGrid(
                prefix + "attention",
                laneAttention(config.headSize())
                        ? LaneAttentionPolicy.worker(groups, config.headSize())
                        : WorkerGridFactory.createAttentionWorker(groups, config.headSize()));
        scheduler.addWorkerGrid(
                prefix + "attention_combine",
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize()));
    }

    /**
     * The grids every Devstral layer shares: the RMS reductions, RoPE and attention. A subclass
     * adds its projections' and RMS apply's.
     */
    protected void addSharedWorkerGrids(GridScheduler scheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);
        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.qDim() / 2, 512);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String prefix = GraphNames.layer(i) + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduceWorker);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduceWorker);
            scheduler.addWorkerGrid(prefix + "rope_and_kv_cache", ropeWithCacheWorker);
            addAttentionWorkerGrids(scheduler, i);
        }
    }
}
