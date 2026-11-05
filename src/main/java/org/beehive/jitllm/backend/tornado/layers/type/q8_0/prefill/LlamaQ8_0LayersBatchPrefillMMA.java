package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.LlamaPrefillRegions;
import org.beehive.jitllm.backend.tornado.layers.TensorCoreBatchAttention;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.ops.Implementations;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan ({@link
 * TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing {@code batchSize} tokens
 * simultaneously via {@link TransformerBatchPrefillKernels}.
 *
 * <p>Tensor-core layer pipeline (12 tasks, all GEMMs on MMA, Q8_0 weights dequantized to FP16 in
 * the GEMM staging registers — W8A16):
 *
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_rope_kv         RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device after every
 * layer so the subsequent single-token decode layers can consume it.
 */
public class LlamaQ8_0LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeProjections() {
        return "FP16 tensor-core MMA (Q8_0 weights dequantized)";
    }

    @Override
    public String describeNativeLibraries() {
        return plan.libraries();
    }

    // Local size for the parallel RMS reductions (one workgroup per token).
    protected static final int RMS_LOCAL_SIZE = 256;

    protected final State state;
    protected final TornadoWeights weights;
    protected final Configuration config;
    protected final KernelContext context = new KernelContext();
    protected final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM). The GEMM-adjacent
    // buffers in State are allocated at this padded size; rows >= batchSize are
    // computed but never consumed. All non-GEMM kernels use the true batchSize.
    protected final int paddedBatch;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /** The attention as tensor-core GEMMs, or null for the flash-attention kernel. */
    private final TensorCoreBatchAttention tcAttention;

    /**
     * Each layer's operations, split where a family runs tasks of its own ({@link
     * LlamaPrefillRegions}).
     */
    private final List<LlamaPrefillRegions.Layer> regions;

    /**
     * Runs those operations: the projections with each weight decoded to FP16 ({@link
     * FP16Projections}), then cuBLAS or the JIT tensor-core GEMM.
     */
    private final WorkPlan plan;

    /** The workspace buffers behind the activation names the operations use. */
    private final Operands operands;

    public LlamaQ8_0LayersBatchPrefillMMA(
            State state, TornadoWeights weights, Configuration config, int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        if (batchSize % 128 != 0
                && (Boolean.getBoolean("jitllm.verbose")
                        || Boolean.getBoolean("jitllm.EnableTimingForTornadoVMInit"))) {
            System.err.printf(
                    "[jitllm] prefill batch %d padded to %d for tensor-core tiles; "
                            + "GEMM efficiency is %d/%d — use a multiple of 128 for best throughput.%n",
                    batchSize, paddedBatch, batchSize, paddedBatch);
        }
        int qDim = config.numberOfHeads() * config.headSize();
        this.tcAttention =
                TensorCoreBatchAttention.applies(useFp16KVCache(), paddedBatch, config.headSize())
                        ? new TensorCoreBatchAttention(
                                state,
                                qDim,
                                config.kvDim(),
                                config.headSize(),
                                config.numberOfHeads(),
                                config.kvMul(),
                                paddedBatch,
                                config.contextLength())
                        : null;
        this.regions =
                LlamaPrefillRegions.of(
                        config,
                        batchSize,
                        paddedBatch,
                        fusedProjections(),
                        denseFeedForward(),
                        (role, layer) -> weights.tensor(role, layer).dataType());
        List<Region> all = LlamaPrefillRegions.all(regions);
        this.plan =
                WorkPlan.of(
                        all,
                        Implementations.tensorCorePrefill(),
                        Target.of(state.executionPolicy()),
                        context,
                        state.workspace.prepacking);
        this.operands =
                Operands.of(
                        Map.ofEntries(
                                Map.entry("x", state.workspace.wrapXBatch),
                                Map.entry("attnIn", state.workspace.wrapXbFP16Batch),
                                Map.entry("attnIn.scale", state.workspace.attnScaleBatch),
                                Map.entry("q", state.workspace.qkvResultBatch),
                                Map.entry("k", state.workspace.qkvResultBatch),
                                Map.entry("v", state.workspace.qkvResultBatch),
                                Map.entry("qkv", state.workspace.qkvResultBatch),
                                Map.entry("attn", state.workspace.attnOutFP16),
                                Map.entry("o", state.workspace.woOut),
                                Map.entry("ffnIn", state.workspace.normedXFFNFP16),
                                Map.entry("ffnIn.scale", state.workspace.ffnScaleBatch),
                                Map.entry("gate", state.workspace.gateUpResultBatch),
                                Map.entry("up", state.workspace.gateUpResultBatch),
                                Map.entry("gateUp", state.workspace.gateUpResultBatch),
                                Map.entry("hidden", state.workspace.wrapHbFP16Batch),
                                Map.entry("down", state.workspace.w2Out)),
                        weights::tensor,
                        all);
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    /**
     * Whether q|k|v and gate|up are one weight each, as in Phi-3. Called from the constructor, so
     * an override returns a constant.
     */
    protected boolean fusedProjections() {
        return false;
    }

    /**
     * Whether the layers have the dense gate, up and down projections; Qwen2-MoE's experts replace
     * them ({@link #feedForward}). Called from the constructor, so an override returns a constant.
     */
    protected boolean denseFeedForward() {
        return true;
    }

    /** What these layers allocate for one session at {@code batchSize}; see the constructor. */
    public static BatchPrefillWorkspace workspace(
            Configuration config, int batchSize, ExecutionPolicy policy, boolean fp16Cache) {
        int padded = (batchSize + 127) & ~127;
        int qDim = config.numberOfHeads() * config.headSize();
        long attention =
                TensorCoreBatchAttention.applies(fp16Cache, padded, config.headSize())
                        ? TensorCoreBatchAttention.bytes(
                                qDim,
                                config.kvDim(),
                                config.headSize(),
                                config.numberOfHeads(),
                                padded,
                                config.contextLength())
                        : 0;
        // Asked on a tensor-core device only (TensorCoreSupport.isTensorCoreCapableBackend), so
        // the projections are planned for one; the weight types do not change the sizes.
        return WorkPlan.needs(
                        LlamaPrefillRegions.all(
                                LlamaPrefillRegions.of(
                                        config,
                                        batchSize,
                                        padded,
                                        false,
                                        true,
                                        (role, layer) -> DataType.Q8_0)),
                        Implementations.tensorCorePrefill(),
                        new Target(policy, DeviceCapabilities.of(DeviceCapability.TENSOR_CORE_MMA)))
                .plus(attention);
    }

    // @formatter:off
    /**
     * Whether the cache is half precision: the representation the state holds, which the decode
     * layers after this prefill bind as well. Batched prefill runs on the tensor-core device only.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    private Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = GraphNames.batchPrefillLayer(layerIndex);
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            // Allocate persistent GPU-side intermediates once
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, plan.buffers());
            if (tcAttention != null) {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, tcAttention.buffers());
            }
            // wrapXBatch produced by the prefillActivation graph and persists in device memory
            // to consume it from there we should use the explicit uniqueTaskGraph name
            // the no-arg form would use current graph name, which causes NPE without CUDA Graphs
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
        } else {
            // for the same reasons as above, we should use the explicit uniqueTaskGraph name to
            // consume
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            batchPrefillLayer.consumeFromDevice(pred, plan.buffers());
            if (tcAttention != null) {
                batchPrefillLayer.consumeFromDevice(pred, tcAttention.buffers());
            }
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex - 1), state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once
        batchPrefillLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION, layerWeights(layerIndex));
        Object[] extraWeights = extraLayerWeights(layerIndex);
        if (extraWeights.length > 0) {
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, extraWeights);
        }

        int dim = config.dim();
        int kvDim = config.kvDim();
        int qDim = config.numberOfHeads() * config.headSize();

        atLayerStart(batchPrefillLayer, layerIndex);

        // ── Attention Block ────────────────────────────────────────────────────
        // RMS norm, then Q, K and V in ONE tensor-core GEMM → packed [q|k|v] rows: no grid
        // starvation on the skinny GQA projections, and the A operand is read once, not thrice.
        emit(batchPrefillLayer, regions.get(layerIndex).beforeAttention());

        afterQkvProjection(batchPrefillLayer, layerIndex);

        // Same fused QKV layout and FP16 attention output as the FP16-weight layers, so the
        // half-precision cache takes their kernels: the weights never reach these two tasks.
        if (useFp16KVCache()) {
            if (splitHalfRope()) {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            } else {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        TransformerPagedKvBatchPrefillKernels
                                ::batchedRopeWithKVCachePackedFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        weights.freq_cis_realFlat.asFloatArray(),
                        weights.freq_cis_imagFlat.asFloatArray(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }

            if (tcAttention != null) {
                tcAttention.appendTasks(batchPrefillLayer, context, layerIndex);
            } else {
                // Register-partitioned P·V accumulation + direct FP16 emission
                // (replaces batchedFlashAttention + attnCast).
                batchPrefillLayer.task(
                        "batch_attention",
                        TransformerPagedKvBatchPrefillKernels
                                ::batchedFlashAttentionFP16OutKVFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }
        } else {
            if (splitHalfRope()) {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedPaged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            } else {
                batchPrefillLayer.task(
                        "batch_rope_kv",
                        TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedPaged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        weights.freq_cis_realFlat.asFloatArray(),
                        weights.freq_cis_imagFlat.asFloatArray(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }

            // Register-partitioned P·V accumulation + direct FP16 emission
            // (replaces batchedFlashAttention + attnCast).
            batchPrefillLayer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionFP16OutPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        emit(batchPrefillLayer, regions.get(layerIndex).attentionOutput());
        afterAttentionOutputProjection(batchPrefillLayer, layerIndex);

        feedForward(batchPrefillLayer, layerIndex);

        // Persist wrapXBatch for the next layer, and KV cache so the decode
        // layers can consume it via the activation graph pass-through.
        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());

        return batchPrefillLayer;
    }

    // @formatter:on

    /** The per-layer weights this layer's tasks read, uploaded once. */
    protected Object[] layerWeights(int layerIndex) {
        return new Object[] {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].deviceArray(),
            weights.wkLayered[layerIndex].deviceArray(),
            weights.wvLayered[layerIndex].deviceArray(),
            weights.woLayered[layerIndex].deviceArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].deviceArray(),
            weights.w2Layered[layerIndex].deviceArray(),
            weights.w3Layered[layerIndex].deviceArray(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    /**
     * The feed-forward block: the attention output added to the stream, RMS norm, gate and up,
     * SwiGLU, down, and the residual add. A family with another feed-forward replaces it whole.
     */
    protected void feedForward(TaskGraph batchPrefillLayer, int layerIndex) {
        LlamaPrefillRegions.Layer layer = regions.get(layerIndex);
        // x += woOut is fused into the FFN RMS reduction (drops the woResid task); W1 and W3 in
        // ONE tensor-core launch → packed [gate|up] rows.
        emit(batchPrefillLayer, layer.feedForwardInput());
        swiglu(batchPrefillLayer, layerIndex);
        emit(batchPrefillLayer, layer.down());
        afterDownProjection(batchPrefillLayer, layerIndex);
        emit(batchPrefillLayer, layer.residual());
    }

    /** Adds the tasks of {@code region}'s operations, as the plan split them. */
    private void emit(TaskGraph batchPrefillLayer, Region region) {
        plan.emit(batchPrefillLayer, region, operands, state.workspace.weightBindings);
    }

    /**
     * The grids of {@link #feedForward}'s own tasks in layer {@code layerIndex}, under its prefix;
     * the plan registers those of the operations it ran.
     */
    protected void addFeedForwardGrids(GridScheduler scheduler, String p, int layerIndex) {
        scheduler.addWorkerGrid(p + "swiglu", elementwiseGrid(batchSize * config.hiddenDim()));
    }

    /** Per-layer weights a family adds to the shared projection set, uploaded with it. */
    protected Object[] extraLayerWeights(int layerIndex) {
        return new Object[0];
    }

    /** Tasks a family runs between the fused QKV GEMM and RoPE, over the packed rows. */
    protected void afterQkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs at the start of a layer, before the attention RMS norm. */
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs on {@code woOut} before it is added to the stream. */
    protected void afterAttentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /** Tasks a family runs on {@code w2Out} before it is added to the stream. */
    protected void afterDownProjection(TaskGraph batchPrefillLayer, int layerIndex) {}

    /**
     * SwiGLU over the packed {@code [gate|up]} rows into {@code wrapHbFP16Batch}, the FP16 operand
     * of the down projection. A family whose activations can leave the FP16 range scales them here.
     */
    protected void swiglu(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                config.hiddenDim());
    }

    /**
     * The grids of the family's own tasks in layer {@code layerIndex}, under its prefix {@code p}.
     */
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {}

    /**
     * Whether RoPE pairs element {@code i} with {@code i + headSize/2} (NeoX), computed from {@link
     * #ropeTheta()}, instead of Llama's adjacent pairs over the precomputed tables.
     */
    protected boolean splitHalfRope() {
        return false;
    }

    /** The RoPE base for {@link #splitHalfRope()}. */
    protected float ropeTheta() {
        throw new UnsupportedOperationException("adjacent-pair RoPE reads the precomputed tables");
    }

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    protected static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128; // BM
        int nBlocks = N / 128; // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1); // groupIdx∈[0,mBlocks), groupIdy∈[0,nBlocks)
        return g;
    }

    protected static WorkerGrid elementwiseGrid(int n) { // n must be a multiple of 256
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int qDim = config.numberOfHeads() * config.headSize();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention (flash): B*nHeads workgroups × min(headSize,128) threads.
        // The kernel requires headSize <= 2*localSize.
        int attnLocal = Math.min(headSz, 128);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = GraphNames.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            addFamilyTaskGrids(scheduler, p, i);
            if (tcAttention != null) {
                tcAttention.addGrids(scheduler, GraphNames.batchPrefillLayer(i));
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            addFeedForwardGrids(scheduler, p, i);
        }
        plan.addGrids(scheduler);
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
