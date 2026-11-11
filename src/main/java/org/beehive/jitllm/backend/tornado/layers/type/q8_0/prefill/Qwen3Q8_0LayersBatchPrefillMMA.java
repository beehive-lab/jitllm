package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PrefillFamily;
import org.beehive.jitllm.backend.tornado.layers.Qwen3PrefillRegions;
import org.beehive.jitllm.backend.tornado.layers.TensorCoreBatchAttention;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.ops.Implementations;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Qwen3 batched-prefill transformer-layer TaskGraphs on the tensor-core (MMA) pipeline. Mirrors
 * {@link LlamaFP16LayersBatchPrefillMMA} with the Qwen3 architectural additions: per-head Q/K RMS
 * normalization between the QKV projection and RoPE, split-half RoPE pairing, and qDim-shaped
 * attention (qDim = nHeads * headDim, which may differ from the model dim).
 *
 * <p>Tensor-core layer pipeline (13 tasks, all GEMMs on MMA, Q8_0 weights dequantized to FP16 in
 * the GEMM staging registers — W8A16):
 *
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_qk_rmsnorm      per-head Q/K RMS norm over the packed buffer
 *   batch_rope_kv         split-half RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM [dim × qDim] → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>Requires dim, qDim, kvDim, and hidDim to be multiples of 128 (holds for all standard Qwen3
 * checkpoints).
 */
public class Qwen3Q8_0LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeProjections() {
        return "FP16 tensor-core MMA (Q8_0 weights dequantized)";
    }

    @Override
    public String describeNativeLibraries() {
        return plan.libraries();
    }

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context;
    private final PrefillFamily family;
    private final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM); see the Llama
    // planner for the padding rationale. Non-GEMM kernels use the true batchSize.
    private final int paddedBatch;
    private final int nHeadKv;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /** The attention as tensor-core GEMMs, or null for the flash-attention kernel. */
    private final TensorCoreBatchAttention tcAttention;

    /** Each layer's operations before and after attention. */
    private final List<Qwen3PrefillRegions.Layer> regions;

    /** Runs those operations: the tensor-core prefill on cuBLAS or on the JIT GEMM. */
    private final WorkPlan plan;

    /** The workspace buffers behind the activation names the operations use. */
    private final Operands operands;

    private static long queryWidth(Qwen3Configuration config) {
        return config.queryDim();
    }

    private static long keyValueWidth(Qwen3Configuration config) {
        return config.kvDim();
    }

    /** What these layers allocate for one session at {@code batchSize}; see the constructor. */
    public static BatchPrefillWorkspace workspace(
            Qwen3Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache,
            DataType weights) {
        int padded = (batchSize + 127) & ~127;
        int headSize = config.numberOfHeadsValue();
        long attention =
                TensorCoreBatchAttention.applies(fp16Cache, padded, headSize)
                        ? TensorCoreBatchAttention.bytes(
                                (int) queryWidth(config),
                                (int) keyValueWidth(config),
                                headSize,
                                config.numberOfHeads(),
                                padded,
                                config.contextLength())
                        : 0;
        return WorkPlan.needs(
                        Qwen3PrefillRegions.all(
                                Qwen3PrefillRegions.of(
                                        config, batchSize, padded, (role, layer) -> weights)),
                        Implementations.tensorCorePrefill(),
                        Target.of(policy))
                .plus(attention);
    }

    public Qwen3Q8_0LayersBatchPrefillMMA(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.state = state;
        this.family = PrefillFamily.of(state);
        // The narrow family takes its primary's context, attention and projections.
        Qwen3Q8_0LayersBatchPrefillMMA primary =
                family.primary(Qwen3Q8_0LayersBatchPrefillMMA.class);
        this.context = primary != null ? primary.context : new KernelContext();
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
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.queryDim();
        this.kvDim = config.kvDim();
        this.gqa = config.numberOfHeads() / nHeadKv;
        this.tcAttention =
                TensorCoreBatchAttention.applies(useFp16KVCache(), paddedBatch, nEmbdHead)
                        ? primary != null
                                ? primary.tcAttention.narrowed(paddedBatch)
                                : new TensorCoreBatchAttention(
                                        state,
                                        qDim,
                                        kvDim,
                                        nEmbdHead,
                                        config.numberOfHeads(),
                                        gqa,
                                        paddedBatch,
                                        config.contextLength())
                        : null;
        this.regions =
                Qwen3PrefillRegions.of(
                        config,
                        batchSize,
                        paddedBatch,
                        (role, layer) -> weights.tensor(role, layer).dataType());
        List<Region> all = Qwen3PrefillRegions.all(regions);
        this.plan =
                WorkPlan.of(
                        primary != null ? primary.plan : null,
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
                                Map.entry("attn", state.workspace.attnOutFP16),
                                Map.entry("o", state.workspace.woOut),
                                Map.entry("ffnIn", state.workspace.normedXFFNFP16),
                                Map.entry("ffnIn.scale", state.workspace.ffnScaleBatch),
                                Map.entry("gate", state.workspace.gateUpResultBatch),
                                Map.entry("up", state.workspace.gateUpResultBatch),
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

        TaskGraph batchPrefillLayer = family.taskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
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
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
        } else {
            String pred = family.batchPrefillLayer(layerIndex - 1);
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
            batchPrefillLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once
        UnifiedMemoryWeights.upload(
                batchPrefillLayer,
                state.workspace,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].deviceArray(),
                weights.wkLayered[layerIndex].deviceArray(),
                weights.wvLayered[layerIndex].deviceArray(),
                weights.woLayered[layerIndex].deviceArray(),
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].deviceArray(),
                weights.w2Layered[layerIndex].deviceArray(),
                weights.w3Layered[layerIndex].deviceArray());
        // The q and k norms are read by this graph's own task, so decode reuses them. The plan's
        // steps record the weights their tasks read.
        state.workspace.weightBindings.record(
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(), graphName);
        state.workspace.weightBindings.record(
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(), graphName);

        // ── Attention Block ────────────────────────────────────────────────────
        // RMS norm, then q, k and v as one GEMM into [q|k|v] rows.
        plan.emit(
                batchPrefillLayer,
                regions.get(layerIndex).beforeAttention(),
                operands,
                state.workspace.weightBindings);

        // Qwen3: per-head RMS norm on Q and K before RoPE
        batchPrefillLayer.task(
                "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNormPackedSimdgroup,
                context,
                state.workspace.qkvResultBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                nHeadKv,
                nEmbdHead,
                qDim,
                kvDim,
                config.rmsNormEps());

        // Same packed QKV layout and FP16 attention output as the FP16-weight layers, so the
        // half-precision cache takes their kernels: the weights never reach these two tasks.
        if (useFp16KVCache()) {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);

            if (tcAttention != null) {
                tcAttention.appendTasks(batchPrefillLayer, context, layerIndex);
            } else {
                // Register-partitioned flash attention over the packed buffer.
                // The 'dim' parameter doubles as the packed-Q stride base and the
                // attnOutFP16 row width — both are qDim for Qwen3.
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
                        nEmbdHead,
                        kvDim,
                        gqa,
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        qDim);
            }
        } else {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3PackedPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);

            // Register-partitioned flash attention over the packed buffer.
            // The 'dim' parameter doubles as the packed-Q stride base and the
            // attnOutFP16 row width — both are qDim for Qwen3.
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
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        // Output projection, residual add, RMS norm, gate|up, SwiGLU, down, residual add.
        plan.emit(
                batchPrefillLayer,
                regions.get(layerIndex).afterAttention(),
                operands,
                state.workspace.weightBindings);

        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());

        return batchPrefillLayer;
    }

    // @formatter:on

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int nHeads = config.numberOfHeads();

        // Q/K per-head RMS norm: one warp per (token, head), four per workgroup when they divide
        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (nHeads + nHeadKv) * 32,
                        (nHeads + nHeadKv) % 4 == 0 ? 128 : 32);

        // Split-half RoPE: B*(qDim/2) threads
        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention: B*nHeads workgroups × min(nEmbdHead,128) threads
        int attnLocal = Math.min(nEmbdHead, 128);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = family.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (tcAttention != null) {
                tcAttention.addGrids(scheduler, family.batchPrefillLayer(i));
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
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
