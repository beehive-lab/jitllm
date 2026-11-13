package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.GroupedExpertMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.GroupedExpertSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.Lfm2BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MMAKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TensorCoreAttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The {@code lfm2moe} layers over a prompt chunk, one task graph per layer: the chunk's rows in
 * {@code wrapXBatch}, their keys and values appended to the cache, the convolution windows carried
 * to where decode continues.
 *
 * <p>The same two choices per layer as {@link Lfm2Layers}: attention or the short convolution,
 * dense or routed feed-forward. The dense projections are SIMD-group GEMMs ({@link
 * QuantizedBatchProjections}); attention, the per-head norms and the rotation are the kernels the
 * Qwen3.5 prefill runs on Metal. The routed feed-forward sorts the chunk's assignments by expert
 * ({@code Qwen35MoeBatchKernels.groupByExpert}) and runs one GEMM per tile of an expert's rows
 * ({@link GroupedExpertSimdgroupKernels}), so each selected expert's weights are read once per tile
 * rather than once per token: a chunk of 512 tokens over 64 experts gives each about 32 rows.
 *
 * <p>On Metal the GEMMs are SIMD-group matrices; elsewhere they are the tiled batch kernels,
 * decoding each weight once per eight rows, with the experts' as grouped tiled kernels. With native
 * libraries on CUDA ({@link Target#cublas}) the attention and convolution projections are cuBLAS
 * FP16 GEMMs over the weight decoded into an FP16 scratch, one projection at a time; the experts
 * stay on the grouped kernels, since the cuBLAS binding has no grouped GEMM, and the two dense
 * feed-forward layers on the JIT kernels. The narrow chunk family is declined.
 */
// @formatter:on
public class Lfm2BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    private static final int ELEMENTWISE_LOCAL = 128;

    /** Lanes of a tiled (non-SIMD-group) batch projection or grouped expert GEMM. */
    private static final int TILED_LOCAL = 128;

    /** Whether the GEMMs are SIMD-group matrices (Metal) rather than tiled batch kernels. */
    private final boolean simdgroup = BatchPrefillGemmPolicy.simdgroup();

    private static final int ATTENTION_LOCAL = Qwen35BatchKernels.ATTENTION_STAGE_LANES;

    private final Lfm2State state;
    private final Lfm2TornadoWeights weights;
    private final Lfm2Configuration config;
    private final int batchSize;
    private final KernelContext context = new KernelContext();
    private final QuantizedBatchProjections projections;

    /**
     * Whether the grouped expert GEMMs run on tensor cores ({@link GroupedExpertMmaKernels}): a
     * CUDA device with them, and expert widths that fill whole 128-output blocks.
     */
    private final boolean mmaExperts;

    /** Whether the dense projections run on cuBLAS ({@link Target#cublas}). */
    private final boolean cublas;

    /**
     * Whether attention runs on tensor cores ({@link
     * TensorCoreAttentionKernels#flashAttentionHead64}): see {@link #flashAttention}. Null keys and
     * values otherwise.
     */
    private final boolean flash;

    private final HalfFloatArray flashKeys;
    private final HalfFloatArray flashValues;

    /** The decode and conversion tasks ahead of the cuBLAS GEMMs, by qualified name, with lanes. */
    private final Map<String, Integer> dequantTasks = new LinkedHashMap<>();

    private final List<ImmutableTaskGraph> graphs;
    private final String lastLayerTaskGraphID;

    public Lfm2BatchPrefillLayers(
            Lfm2State state, Lfm2TornadoWeights weights, Lfm2Configuration config, int batchSize) {
        PrefillFamily family = PrefillFamily.of(state);
        if (family.isNarrow()) {
            throw PrefillFamily.decline("the lfm2moe layers build no narrow family yet");
        }
        if (config.expertHiddenDim() > config.dim()) {
            throw new UnsupportedOperationException(
                    "the batched lfm2moe experts keep their up rows in the output rows, which"
                            + " needs an expert width no larger than the model's");
        }
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.projections =
                new QuantizedBatchProjections(
                        context, state.workspace, batchSize, simdgroup, true, TILED_LOCAL);
        this.cublas = Target.of(state.executionPolicy()).cublas();
        this.mmaExperts =
                !simdgroup
                        && TensorCoreSupport.isTensorCoreCapableBackend()
                        && config.expertHiddenDim() % 128 == 0
                        && config.dim() % 128 == 0;
        if (mmaExperts) {
            // The FP16 rows share the cuBLAS weight scratch where there is one: the experts run
            // after the layer's projections, which leave it free.
            long halves =
                    Math.max(
                            (long) batchSize * config.dim(),
                            (long) assignments() * config.expertHiddenDim());
            HalfFloatArray scratch = state.workspace.contiguousF16WeightsScratch;
            state.workspace.wrapMoeHalvesBatch =
                    cublas && scratch != null && scratch.getSize() >= halves
                            ? scratch
                            : TornadoWorkspaces.halfFloats(Math.toIntExact(halves));
            state.workspace.wrapMoeBlockSums =
                    TornadoWorkspaces.floats(Math.toIntExact(halves / 32));
        }
        if (cublas) {
            int widest = Math.toIntExact(nativeScratchBytes(config) / 2);
            HalfFloatArray current = state.workspace.contiguousF16WeightsScratch;
            if (current == null || current.getSize() < widest) {
                state.workspace.contiguousF16WeightsScratch = TornadoWorkspaces.halfFloats(widest);
            }
        }
        this.flash =
                flashAttention(cublas, state.usesFp16KeyValueCache(), config)
                        && batchSize % TensorCoreAttentionKernels.FLASH_ROWS == 0;
        int keyValueHalves = config.kvDim() * flashPositions(config);
        this.flashKeys = flash ? TornadoWorkspaces.halfFloats(keyValueHalves) : null;
        this.flashValues = flash ? TornadoWorkspaces.halfFloats(keyValueHalves) : null;
        this.graphs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::buildLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
        this.lastLayerTaskGraphID = GraphNames.batchPrefillLayer(config.numberOfLayers() - 1);
    }

    /**
     * The bytes {@code Lfm2State} allocates for a chunk of {@code batch} rows beyond the staging
     * every family has: the normalized chunk, the convolution's, and the mixture's routing and
     * rows. Example: LFM2.5-24B-A2B at 512 rows, about 45 MB.
     */
    public static long workspaceBytes(Lfm2Configuration config, int batch) {
        long dim = config.dim();
        long assignments = (long) batch * config.expertUsedCount();
        long hidden = config.expertHiddenDim();
        long floats =
                batch * dim
                        + (long) batch * config.shortConvProjectionDim()
                        + batch * dim
                        + (long) batch * config.expertCount()
                        + assignments
                        + assignments * hidden
                        + assignments * Math.max(dim, hidden);
        long ints =
                3 * assignments
                        + 1
                        + 3L
                                * Qwen35MoeBatchKernels.maxTiles(
                                        Math.toIntExact(assignments), config.expertCount());
        return 4 * (floats + ints);
    }

    /**
     * The FP16 scratch the cuBLAS projections decode each weight into: the largest attention or
     * convolution projection. Example: LFM2.5-24B-A2B's conv in_proj, 6144 x 2048, 25 MB.
     */
    public static long nativeScratchBytes(Lfm2Configuration config) {
        long dim = config.dim();
        return 2 * dim * Math.max(dim, config.shortConvProjectionDim());
    }

    /**
     * Whether attention runs on tensor cores: cuBLAS output projections (they read the attention's
     * FP16 rows), the FP16 cache, heads 64 wide, and a device with tensor cores.
     */
    public static boolean flashAttention(
            boolean cublas, boolean fp16Cache, Lfm2Configuration config) {
        return cublas
                && fp16Cache
                && config.headSize() == 64
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /**
     * The keys and values {@link TensorCoreAttentionKernels#gatherPagedKeysTransposed} copies the
     * cache into, FP16, when {@link #flashAttention} holds. Example: LFM2.5-24B-A2B at 4096
     * positions, 8 MB.
     */
    public static long flashAttentionBytes(Lfm2Configuration config) {
        return 2L * config.kvDim() * flashPositions(config) * Short.BYTES;
    }

    /** The context's positions rounded up to whole 128-key tiles. */
    private static int flashPositions(Lfm2Configuration config) {
        return (config.contextLength() + 127) / 128 * 128;
    }

    @Override
    public String describeProjections() {
        return cublas
                ? "cuBLAS FP16 attention and convolution projections, JIT kernels elsewhere"
                : "JIT kernels (SIMD-group GEMMs, grouped expert GEMMs)";
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    private boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    private Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    private static TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "lfm2moe layer " + layer + " must carry " + role + " for a batched prefill");
        }
        return tensor;
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    private TaskGraph buildLayer(int layerIndex) {
        var ws = state.workspace;
        TaskGraph layer = new TaskGraph(GraphNames.batchPrefillLayer(layerIndex));
        String predecessor =
                layerIndex == 0
                        ? GraphNames.PREFILL_ACTIVATION
                        : GraphNames.batchPrefillLayer(layerIndex - 1);
        layer.consumeFromDevice(predecessor, ws.wrapXBatch);
        configureTransfers(layer, layerIndex, predecessor);
        bindLayerWeights(layer, layerIndex);

        rmsReduce(layer, "attn_rms_reduce", ws.attnScaleBatch);
        layer.task(
                "attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm").asFloatArray(),
                ws.attnScaleBatch,
                config.dim());

        if (config.isAttentionLayer(layerIndex)) {
            attentionBranch(layer, layerIndex);
        } else {
            shortConvBranch(layer, layerIndex);
        }

        rmsReduce(layer, "ffn_rms_reduce", ws.ffnScaleBatch);
        if (config.isMoeLayer(layerIndex)) {
            mixtureOfExperts(layer, layerIndex);
        } else {
            projections.gateUp(
                    layer,
                    require(weights.w1Layered, layerIndex, "ffn_gate"),
                    require(weights.w3Layered, layerIndex, "ffn_up"),
                    require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm").asFloatArray(),
                    config.dim(),
                    config.hiddenDim());
            projections.projection(
                    layer,
                    "ffn_down_proj",
                    ws.wrapHbBatch,
                    ws.wrapXBatch,
                    require(weights.w2Layered, layerIndex, "ffn_down"),
                    config.hiddenDim(),
                    config.dim(),
                    true);
        }

        layer.persistOnDevice(
                ws.wrapXBatch, keyStore(), valueStore(), ws.wrapBlockTable, ws.wrapConvState);
        return layer;
    }

    private void rmsReduce(TaskGraph layer, String name, FloatArray scale) {
        layer.task(
                name,
                Qwen35BatchKernels::rmsReduceBatchGroup,
                context,
                state.workspace.wrapXBatch,
                scale,
                config.dim(),
                config.rmsNormEps());
    }

    /** Attention over the chunk, from {@code wrapNormedBatch} back into {@code wrapXBatch}. */
    private void attentionBranch(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        final int dim = config.dim();
        final int kvDim = config.kvDim();
        final int headDim = config.headSize();
        store(
                layer,
                "attn_q_proj",
                ws.wrapNormedBatch,
                ws.wrapQBatch,
                require(weights.wqLayered, layerIndex, "attn_q"),
                dim,
                dim);
        store(
                layer,
                "attn_k_proj",
                ws.wrapNormedBatch,
                ws.wrapKBatch,
                require(weights.wkLayered, layerIndex, "attn_k"),
                dim,
                kvDim);
        store(
                layer,
                "attn_v_proj",
                ws.wrapNormedBatch,
                ws.wrapVBatch,
                require(weights.wvLayered, layerIndex, "attn_v"),
                dim,
                kvDim);
        layer.task(
                "attn_qk_norm",
                Qwen35BatchKernels::fusedQKRmsNormBatchGroup,
                context,
                ws.wrapQBatch,
                ws.wrapKBatch,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.rmsNormEps(),
                ws.batchStartPosHolder);
        layer.task(
                "attn_rope",
                Qwen35BatchKernels::ropeNeoxPartialBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapQBatch,
                ws.wrapKBatch,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim);
        int kvLayer = config.keyValueLayerIndex(layerIndex);
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchFP16Paged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            if (flashLayer(layerIndex)) {
                flashAttentionTasks(layer, kvLayer);
                cublasGemm(
                        layer,
                        "attn_output_proj",
                        ws.wrapXbFP16Batch,
                        ws.wrapXBatch,
                        weights.woLayered[layerIndex],
                        dim,
                        dim,
                        true);
                return;
            }
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchFP16Paged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchPaged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchPaged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        }
        project(
                layer,
                "attn_output_proj",
                ws.wrapXbBatch,
                ws.wrapXBatch,
                require(weights.woLayered, layerIndex, "attn_output"),
                dim,
                dim,
                true);
    }

    /** Whether layer {@code layerIndex} runs {@link #flashAttentionTasks}. */
    private boolean flashLayer(int layerIndex) {
        return flash && cublasForm(require(weights.woLayered, layerIndex, "attn_output"));
    }

    /**
     * Attention on tensor cores, from {@code wrapQBatch} into {@code wrapXbFP16Batch}, the FP16
     * rows the output projection's cuBLAS GEMM reads: the queries converted in place there, the
     * layer's cache up to the chunk's last position gathered into {@code flashKeys} and {@code
     * flashValues}, then {@link TensorCoreAttentionKernels#flashAttentionHead64}. A workgroup reads
     * its rows of one head's queries before it writes the same rows of that head's output, and no
     * other workgroup reads them, so the output can overwrite the queries.
     */
    private void flashAttentionTasks(TaskGraph layer, int kvLayer) {
        var ws = state.workspace;
        int qDim = config.dim();
        layer.task(
                "attn_q16",
                TensorCoreAttentionKernels::queriesToHalf,
                context,
                ws.wrapQBatch,
                ws.wrapXbFP16Batch,
                qDim,
                qDim,
                batchSize * qDim);
        layer.task(
                "attn_gather",
                TensorCoreAttentionKernels::gatherPagedKeysTransposed,
                context,
                ws.batchStartPosHolder,
                ws.wrapKeyCacheFP16,
                ws.wrapValueCacheFP16,
                flashKeys,
                flashValues,
                kvLayer,
                config.kvDim(),
                config.headSize(),
                ws.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                flashPositions(config));
        layer.task(
                "attention",
                TensorCoreAttentionKernels::flashAttentionHead64,
                context,
                ws.batchStartPosHolder,
                ws.wrapXbFP16Batch,
                flashKeys,
                flashValues,
                ws.wrapXbFP16Batch,
                config.numberOfHeads(),
                config.kvMul(),
                qDim,
                batchSize / TensorCoreAttentionKernels.FLASH_ROWS,
                flashPositions(config),
                (float) (1.0 / Math.sqrt(config.headSize())));
    }

    /**
     * {@code out = w · x} over the chunk. A Q8_0 weight off SIMD-group matrices has only the tiled
     * GEMM that adds into its output, so the output is zeroed first; a prepacked one stores.
     */
    private void store(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        if (cublasForm(w)) {
            cublasProjection(layer, name, x, out, w, n, d, false);
        } else if (!simdgroup && w.dataType() == DataType.Q8_0 && !projections.prepacked(w)) {
            layer.task(
                    name + "_zero",
                    Lfm2BatchKernels::zeroRows,
                    context,
                    state.workspace.batchStartPosHolder,
                    out,
                    d);
            projections.projection(layer, name, x, out, w, n, d, true);
        } else {
            projections.projection(layer, name, x, out, w, n, d, false);
        }
    }

    /**
     * {@code out (+)= w · x} over the chunk: on cuBLAS where it has the form, the JIT kernels else.
     */
    private void project(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean residual) {
        if (cublasForm(w)) {
            cublasProjection(layer, name, x, out, w, n, d, residual);
        } else {
            projections.projection(layer, name, x, out, w, n, d, residual);
        }
    }

    /**
     * Whether attention or convolution weight {@code w} runs on cuBLAS: native libraries, and a
     * type with an FP16 decode. The dense feed-forward stays on the JIT kernels, whose weights
     * would need a scratch twice the size.
     */
    private boolean cublasForm(TornadoTensor w) {
        DataType type = w.dataType();
        return cublas && (type == DataType.Q8_0 || type == DataType.Q4_K || type == DataType.Q6_K);
    }

    /**
     * {@code out (+)= w · x} as cuBLAS's FP16 GEMM with FP32 accumulation: {@code x} converted to
     * FP16 by {@code name + "_x16"}, {@code w} decoded into the FP16 scratch by {@code name +
     * "_dequant"}, then one GEMM over the chunk, adding into {@code out} for a residual projection.
     */
    private void cublasProjection(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean residual) {
        HalfFloatArray x16 = state.workspace.wrapXbFP16Batch;
        dequantTasks.put(layer.getTaskGraphName() + "." + name + "_x16", batchSize * n);
        layer.task(name + "_x16", Qwen35MMAKernels::convertToFP16, context, x, x16);
        cublasGemm(layer, name, x16, out, w, n, d, residual);
    }

    /**
     * {@link #cublasProjection} over FP16 rows {@code x16} already in place: {@code w} decoded into
     * the FP16 scratch by {@code name + "_dequant"}, then the GEMM.
     */
    private void cublasGemm(
            TaskGraph layer,
            String name,
            HalfFloatArray x16,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean residual) {
        HalfFloatArray weights16 = state.workspace.contiguousF16WeightsScratch;
        int elements = n * d;
        // Q8_0 decodes a lane per weight, the K-quants four.
        int lanes = w.dataType() == DataType.Q8_0 ? elements : elements / 4;
        dequantTasks.put(layer.getTaskGraphName() + "." + name + "_dequant", lanes);
        layer.task(
                name + "_dequant",
                switch (w.dataType()) {
                    case Q8_0 -> TransformerBatchPrefillKernels::dequantizeQ8_0ToFP16;
                    case Q4_K -> TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16;
                    default -> TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16;
                },
                context,
                w.asByteArray(),
                weights16,
                0,
                elements);
        layer.libraryTask(
                name,
                uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16FP32,
                1,
                0,
                d,
                batchSize,
                n,
                1.0f,
                weights16,
                n,
                x16,
                n,
                residual ? 1.0f : 0.0f,
                out,
                d);
    }

    /** The gated short convolution over the chunk, rows in order, into {@code wrapXBatch}. */
    private void shortConvBranch(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        store(
                layer,
                "conv_in_proj",
                ws.wrapNormedBatch,
                ws.wrapSsmQkvBatch,
                require(weights.convInProj, layerIndex, "shortconv.in_proj"),
                config.dim(),
                config.shortConvProjectionDim());
        layer.task(
                "conv",
                Lfm2BatchKernels::shortConvBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapSsmQkvBatch,
                require(weights.convKernel, layerIndex, "shortconv.conv").asFloatArray(),
                ws.wrapConvState,
                ws.wrapSsmConvOutBatch,
                config.dim(),
                config.shortConvCache(),
                config.convLayerIndex(layerIndex) * config.convStateSize());
        layer.task(
                "conv_window",
                Lfm2BatchKernels::shortConvBatchWindow,
                context,
                ws.batchStartPosHolder,
                ws.wrapSsmQkvBatch,
                ws.wrapConvState,
                config.dim(),
                config.shortConvCache(),
                config.convLayerIndex(layerIndex) * config.convStateSize());
        project(
                layer,
                "conv_out_proj",
                ws.wrapSsmConvOutBatch,
                ws.wrapXBatch,
                require(weights.convOutProj, layerIndex, "shortconv.out_proj"),
                config.dim(),
                config.dim(),
                true);
    }

    /** The routed feed-forward over the chunk, into {@code wrapXBatch}. */
    private void mixtureOfExperts(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        final int dim = config.dim();
        final int hidden = config.expertHiddenDim();
        final int used = config.expertUsedCount();
        layer.task(
                "ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm").asFloatArray(),
                ws.ffnScaleBatch,
                dim);
        layer.task(
                "moe_router",
                Lfm2BatchKernels::routerBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapNormedBatch,
                require(weights.ffnGateInp, layerIndex, "ffn_gate_inp").asFloatArray(),
                ws.wrapMoeLogitsBatch,
                dim,
                config.expertCount());
        layer.task(
                "moe_select",
                Lfm2BatchKernels::selectExpertsSigmoidBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeLogitsBatch,
                require(weights.expProbsBias, layerIndex, "exp_probs_b").asFloatArray(),
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                config.expertCount(),
                used);
        layer.task(
                "moe_group",
                Qwen35MoeBatchKernels::groupByExpert,
                context,
                ws.wrapMoeIdsBatch,
                ws.batchStartPosHolder,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                config.expertCount(),
                used,
                batchSize);
        TornadoTensor gate = require(weights.ffnGateExps, layerIndex, "ffn_gate_exps");
        TornadoTensor up = require(weights.ffnUpExps, layerIndex, "ffn_up_exps");
        TornadoTensor down = require(weights.ffnDownExps, layerIndex, "ffn_down_exps");
        if (mmaExperts) {
            layer.task(
                    "moe_x16",
                    Lfm2BatchKernels::halvesWithBlockSums,
                    context,
                    ws.wrapNormedBatch,
                    ws.wrapMoeHalvesBatch,
                    ws.wrapMoeBlockSums,
                    batchSize * dim / 32);
        }
        grouped(layer, "moe_gate", gate, ws.wrapNormedBatch, ws.wrapMoeHiddenBatch, 1, dim, hidden);
        grouped(layer, "moe_up", up, ws.wrapNormedBatch, ws.wrapMoeOutBatch, 1, dim, hidden);
        layer.task(
                "moe_swiglu",
                Lfm2BatchKernels::siluMultiplyAssignments,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeOutBatch,
                hidden,
                used);
        if (mmaExperts) {
            layer.task(
                    "moe_h16",
                    Lfm2BatchKernels::halvesWithBlockSums,
                    context,
                    ws.wrapMoeHiddenBatch,
                    ws.wrapMoeHalvesBatch,
                    ws.wrapMoeBlockSums,
                    assignments() * hidden / 32);
        }
        grouped(layer, "moe_down", down, ws.wrapMoeHiddenBatch, ws.wrapMoeOutBatch, 0, hidden, dim);
        layer.task(
                "moe_combine",
                Lfm2BatchKernels::combineBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapXBatch,
                dim,
                used);
    }

    /** A chunk's expert assignments: {@code batchSize} rows, each to the used experts. */
    private int assignments() {
        return batchSize * config.expertUsedCount();
    }

    /** One grouped expert GEMM over the tiles {@code moe_group} wrote. */
    private void grouped(
            TaskGraph layer,
            String name,
            TornadoTensor w,
            FloatArray in,
            FloatArray out,
            int gather,
            int n,
            int d) {
        var ws = state.workspace;
        if (mmaExperts) {
            layer.task(
                    name,
                    switch (w.dataType()) {
                        case Q4_K -> GroupedExpertMmaKernels::groupedExpertMmaQ4_K;
                        case Q5_K -> GroupedExpertMmaKernels::groupedExpertMmaQ5_K;
                        case Q6_K -> GroupedExpertMmaKernels::groupedExpertMmaQ6_K;
                        case Q8_0 -> GroupedExpertMmaKernels::groupedExpertMmaQ8_0;
                        default ->
                                throw new UnsupportedOperationException(
                                        "no batched lfm2moe expert GEMM for " + w.dataType());
                    },
                    context,
                    ws.wrapMoeHalvesBatch,
                    ws.wrapMoeBlockSums,
                    out,
                    w.asByteArray(),
                    ws.wrapMoeTiles,
                    ws.wrapMoeSortedToken,
                    gather,
                    n,
                    d);
            return;
        }
        if (!simdgroup) {
            layer.task(
                    name,
                    switch (w.dataType()) {
                        case Q4_K -> TransformerComputeKernelsQ4_K::groupedExpertTiledBatchQ4_K;
                        case Q5_K -> TransformerComputeKernelsQ5_K::groupedExpertTiledBatchQ5_K;
                        case Q6_K -> TransformerComputeKernelsQ6_K::groupedExpertTiledBatchQ6_K;
                        case Q8_0 -> TransformerBatchPrefillKernels::groupedExpertTiledBatchQ8_0;
                        default ->
                                throw new UnsupportedOperationException(
                                        "no batched lfm2moe expert GEMM for " + w.dataType());
                    },
                    context,
                    in,
                    out,
                    w.asByteArray(),
                    ws.wrapMoeTiles,
                    ws.wrapMoeSortedToken,
                    gather,
                    n,
                    d,
                    TILED_LOCAL);
            return;
        }
        boolean halfTiles = groupedHalfTiles(w.dataType(), d);
        layer.task(
                name,
                switch (w.dataType()) {
                    case Q4_K ->
                            halfTiles
                                    ? GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_KHalfTiles
                                    : GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_K;
                    case Q5_K ->
                            halfTiles
                                    ? GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_KHalfTiles
                                    : GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_K;
                    case Q6_K ->
                            halfTiles
                                    ? GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_KHalfTiles
                                    : GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_K;
                    case Q8_0 -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ8_0;
                    default ->
                            throw new UnsupportedOperationException(
                                    "no batched lfm2moe expert GEMM for " + w.dataType());
                },
                context,
                in,
                out,
                w.asByteArray(),
                ws.wrapMoeTiles,
                ws.wrapMoeSortedToken,
                gather,
                n,
                d);
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    /**
     * This layer's weights: read in place where the device shares memory with the host, uploaded
     * once otherwise and recorded, so the decode graphs bind the same buffers.
     */
    private void bindLayerWeights(TaskGraph layer, int layerIndex) {
        Lfm2Layers.bindPrepacked(layer, state.workspace, weights, layerIndex);
        List<Object> tensors = new ArrayList<>();
        tensors.add(require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));
        tensors.add(require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm"));
        if (config.isMoeLayer(layerIndex)) {
            tensors.add(require(weights.ffnGateInp, layerIndex, "ffn_gate_inp"));
            tensors.add(require(weights.expProbsBias, layerIndex, "exp_probs_b"));
            tensors.add(require(weights.ffnGateExps, layerIndex, "ffn_gate_exps"));
            tensors.add(require(weights.ffnUpExps, layerIndex, "ffn_up_exps"));
            tensors.add(require(weights.ffnDownExps, layerIndex, "ffn_down_exps"));
        } else {
            tensors.add(require(weights.w1Layered, layerIndex, "ffn_gate"));
            tensors.add(require(weights.w2Layered, layerIndex, "ffn_down"));
            tensors.add(require(weights.w3Layered, layerIndex, "ffn_up"));
        }
        if (config.isAttentionLayer(layerIndex)) {
            tensors.add(require(weights.wqLayered, layerIndex, "attn_q"));
            tensors.add(require(weights.wkLayered, layerIndex, "attn_k"));
            tensors.add(require(weights.wvLayered, layerIndex, "attn_v"));
            tensors.add(require(weights.woLayered, layerIndex, "attn_output"));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm"));
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm"));
        } else {
            tensors.add(require(weights.convInProj, layerIndex, "shortconv.in_proj"));
            tensors.add(require(weights.convKernel, layerIndex, "shortconv.conv"));
            tensors.add(require(weights.convOutProj, layerIndex, "shortconv.out_proj"));
        }
        tensors.removeIf(
                t -> t instanceof TornadoTensor w && state.workspace.prepacking.prepacked(w));
        Object[] uploaded =
                UnifiedMemoryWeights.consume(
                        layer, state.workspace, WeightPrepacker.deviceArrays(tensors.toArray()));
        if (uploaded.length > 0) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, uploaded);
            for (Object w : uploaded) {
                state.workspace.weightBindings.record(w, layer.getTaskGraphName());
            }
        }
    }

    /**
     * The chunk's scratch, uploaded once by layer 0 and bound from the previous layer after it; the
     * chunk holder and block table per chunk. The convolution windows are uploaded once and never
     * read back: they are the session's history, advanced here and then by decode.
     */
    private void configureTransfers(TaskGraph layer, int layerIndex, String predecessor) {
        var ws = state.workspace;
        Object[] scratch = {
            context,
            ws.wrapNormedBatch,
            ws.wrapQBatch,
            ws.wrapKBatch,
            ws.wrapVBatch,
            ws.wrapXbBatch,
            ws.wrapHbBatch,
            ws.ffnUpResult,
            ws.attnScaleBatch,
            ws.ffnScaleBatch,
            ws.wrapSsmQkvBatch,
            ws.wrapSsmConvOutBatch,
            ws.wrapMoeLogitsBatch,
            ws.wrapMoeIdsBatch,
            ws.wrapMoeWeightsBatch,
            ws.wrapMoeSortedToken,
            ws.wrapMoePosition,
            ws.wrapMoeTiles,
            ws.wrapMoeHiddenBatch,
            ws.wrapMoeOutBatch,
            keyStore(),
            valueStore(),
            ws.wrapConvState
        };
        if (cublas) {
            scratch = java.util.Arrays.copyOf(scratch, scratch.length + 2);
            scratch[scratch.length - 2] = ws.contiguousF16WeightsScratch;
            scratch[scratch.length - 1] = ws.wrapXbFP16Batch;
        }
        if (flash) {
            scratch = java.util.Arrays.copyOf(scratch, scratch.length + 2);
            scratch[scratch.length - 2] = flashKeys;
            scratch[scratch.length - 1] = flashValues;
        }
        if (mmaExperts) {
            boolean shared = ws.wrapMoeHalvesBatch == ws.contiguousF16WeightsScratch;
            scratch = java.util.Arrays.copyOf(scratch, scratch.length + (shared ? 1 : 2));
            scratch[scratch.length - 1] = ws.wrapMoeBlockSums;
            if (!shared) {
                scratch[scratch.length - 2] = ws.wrapMoeHalvesBatch;
            }
        }
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.wrapBlockTable);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch);
        } else {
            layer.consumeFromDevice(predecessor, scratch);
            layer.consumeFromDevice(predecessor, ws.batchStartPosHolder, ws.wrapBlockTable);
        }
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    /** The grid of a grouped expert GEMM to {@code d} outputs of a tensor in {@code type}. */
    /**
     * Whether a grouped SIMD-group GEMM to {@code d} outputs of a K-quant tensor stages FP16 weight
     * tiles: over the chunk's whole tile table its grid is wide enough, as the dense GEMMs' rule
     * says ({@code TransformerBatchPrefillSimdgroupKernels.halfTiles}).
     */
    private boolean groupedHalfTiles(DataType type, int d) {
        int maxTiles =
                Qwen35MoeBatchKernels.maxTiles(
                        batchSize * config.expertUsedCount(), config.expertCount());
        return type != DataType.Q8_0
                && TransformerBatchPrefillSimdgroupKernels.halfTiles((d + 63) / 64, maxTiles * 64);
    }

    private WorkerGrid groupedWorker(DataType type, int d, int maxTiles) {
        if (mmaExperts) {
            int threads = GroupedExpertMmaKernels.THREADS;
            int workgroups =
                    type == DataType.Q4_K || type == DataType.Q5_K
                            ? GroupedExpertMmaKernels.splitWorkgroups(maxTiles, d)
                            : maxTiles * (d / 128);
            return WorkerGridFactory.genericWorker(workgroups * threads, threads);
        }
        if (simdgroup) {
            int threads =
                    TransformerBatchPrefillSimdgroupKernels.threads(groupedHalfTiles(type, d));
            return WorkerGridFactory.genericWorker(((d + 63) / 64) * maxTiles * threads, threads);
        }
        int colTile =
                switch (type) {
                    case Q4_K -> TransformerComputeKernelsQ4_K.colTile();
                    case Q5_K -> TransformerComputeKernelsQ5_K.colTile();
                    case Q8_0 -> TransformerBatchPrefillKernels.GROUPED_Q8_COL_TILE;
                    default -> TransformerComputeKernelsQ6_K.colTile();
                };
        int subTiles =
                type == DataType.Q8_0
                        ? TransformerBatchPrefillKernels.GROUPED_Q8_SUB_TILES
                        : TransformerComputeKernelsQ4_K.GROUPED_SUB_TILES;
        return WorkerGridFactory.genericWorker(
                maxTiles * subTiles * ((d + colTile - 1) / colTile) * TILED_LOCAL, TILED_LOCAL);
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsReduce =
                WorkerGridFactory.genericWorker(
                        batchSize * Qwen35BatchKernels.RMS_GROUP_LOCAL,
                        Qwen35BatchKernels.RMS_GROUP_LOCAL);
        WorkerGrid rmsApply =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        int headDim = config.headSize();
        int qkHeads = batchSize * (config.numberOfHeads() + config.numberOfKeyValueHeads());
        WorkerGrid qkNorm = WorkerGridFactory.genericWorker(qkHeads * headDim, headDim);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * (headDim / 2), 32);
        WorkerGrid kvAppend =
                WorkerGridFactory.genericWorker(batchSize * config.kvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * ATTENTION_LOCAL, ATTENTION_LOCAL);
        WorkerGrid conv =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid convWindow = WorkerGridFactory.genericWorker(config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid flashQueries =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid flashGather =
                WorkerGridFactory.genericWorker(
                        flashPositions(config) * config.kvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid1D flashGrid =
                new WorkerGrid1D(
                        batchSize
                                / TensorCoreAttentionKernels.FLASH_ROWS
                                * config.numberOfHeads()
                                * TensorCoreAttentionKernels.FLASH_THREADS);
        flashGrid.setLocalWork(TensorCoreAttentionKernels.FLASH_THREADS, 1, 1);

        int used = config.expertUsedCount();
        int assignments = batchSize * used;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, config.expertCount());
        int routerJobs = batchSize * config.expertCount();
        WorkerGrid router =
                WorkerGridFactory.genericWorker(
                        ((routerJobs + Lfm2BatchKernels.ROUTER_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.ROUTER_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.ROUTER_SIMDGROUPS,
                        32 * Lfm2BatchKernels.ROUTER_SIMDGROUPS);
        WorkerGrid select =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Lfm2BatchKernels.SELECT_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.SELECT_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.SELECT_SIMDGROUPS,
                        32 * Lfm2BatchKernels.SELECT_SIMDGROUPS);
        WorkerGrid group =
                WorkerGridFactory.genericWorker(
                        Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS);
        int hidden = config.expertHiddenDim();
        WorkerGrid zero =
                WorkerGridFactory.genericWorker(
                        ((batchSize * Math.max(config.dim(), config.shortConvProjectionDim()) + 255)
                                        / 256)
                                * 256,
                        256);
        WorkerGrid swiglu =
                WorkerGridFactory.genericWorker(((assignments * hidden + 255) / 256) * 256, 256);
        WorkerGrid combine =
                WorkerGridFactory.genericWorker(
                        ((batchSize * config.dim() + 255) / 256) * 256, 256);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = GraphNames.batchPrefillLayer(layer) + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            if (config.isAttentionLayer(layer)) {
                projectionGrid(
                        scheduler, prefix + "attn_q_proj", weights.wqLayered[layer], config.dim());
                projectionGrid(
                        scheduler,
                        prefix + "attn_k_proj",
                        weights.wkLayered[layer],
                        config.kvDim());
                projectionGrid(
                        scheduler,
                        prefix + "attn_v_proj",
                        weights.wvLayered[layer],
                        config.kvDim());
                for (String store : new String[] {"attn_q_proj", "attn_k_proj", "attn_v_proj"}) {
                    scheduler.addWorkerGrid(prefix + store + "_zero", zero);
                }
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                if (flashLayer(layer)) {
                    scheduler.addWorkerGrid(prefix + "attn_q16", flashQueries);
                    scheduler.addWorkerGrid(prefix + "attn_gather", flashGather);
                    scheduler.addWorkerGrid(prefix + "attention", flashGrid);
                } else {
                    scheduler.addWorkerGrid(prefix + "attention", attention);
                }
                projectionGrid(
                        scheduler,
                        prefix + "attn_output_proj",
                        weights.woLayered[layer],
                        config.dim());
            } else {
                projectionGrid(
                        scheduler,
                        prefix + "conv_in_proj",
                        weights.convInProj[layer],
                        config.shortConvProjectionDim());
                scheduler.addWorkerGrid(prefix + "conv_in_proj_zero", zero);
                scheduler.addWorkerGrid(prefix + "conv", conv);
                scheduler.addWorkerGrid(prefix + "conv_window", convWindow);
                projectionGrid(
                        scheduler,
                        prefix + "conv_out_proj",
                        weights.convOutProj[layer],
                        config.dim());
            }
            if (config.isMoeLayer(layer)) {
                scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
                scheduler.addWorkerGrid(prefix + "moe_router", router);
                scheduler.addWorkerGrid(prefix + "moe_select", select);
                scheduler.addWorkerGrid(prefix + "moe_group", group);
                WorkerGrid gateUp =
                        groupedWorker(weights.ffnGateExps[layer].dataType(), hidden, maxTiles);
                scheduler.addWorkerGrid(prefix + "moe_gate", gateUp);
                scheduler.addWorkerGrid(prefix + "moe_up", gateUp);
                scheduler.addWorkerGrid(prefix + "moe_swiglu", swiglu);
                if (mmaExperts) {
                    int inBlocks = batchSize * config.dim() / 32;
                    int hiddenBlocks = assignments * hidden / 32;
                    scheduler.addWorkerGrid(prefix + "moe_x16", halvesWorker(inBlocks));
                    scheduler.addWorkerGrid(prefix + "moe_h16", halvesWorker(hiddenBlocks));
                }
                scheduler.addWorkerGrid(
                        prefix + "moe_down",
                        groupedWorker(
                                weights.ffnDownExps[layer].dataType(), config.dim(), maxTiles));
                scheduler.addWorkerGrid(prefix + "moe_combine", combine);
            } else {
                TornadoTensor w1 = weights.w1Layered[layer];
                if (projections.fusedGateUp(w1, weights.w3Layered[layer]) && simdgroup) {
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            projections.gateUpWorker(w1, config.hiddenDim()));
                } else if (projections.fusedGateUp(w1, weights.w3Layered[layer])) {
                    // The tiled fused kernels: Q8_0's applies the norm as it stages, the others
                    // after the norm's own task.
                    if (w1.dataType() != DataType.Q8_0) {
                        scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", rmsApply);
                    }
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            projections.worker(w1, config.hiddenDim()));
                } else {
                    scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", rmsApply);
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate", projections.worker(w1, config.hiddenDim()));
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_up",
                            projections.worker(weights.w3Layered[layer], config.hiddenDim()));
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            WorkerGridFactory.genericWorker(
                                    ((batchSize * config.hiddenDim() + 255) / 256) * 256, 256));
                }
                scheduler.addWorkerGrid(
                        prefix + "ffn_down_proj",
                        projections.worker(weights.w2Layered[layer], config.dim()));
            }
        }
        dequantTasks.forEach(
                (task, elements) ->
                        scheduler.addWorkerGrid(
                                task,
                                WorkerGridFactory.genericWorker(
                                        ((elements + 255) / 256) * 256, 256)));
    }

    /** The grid of {@link Lfm2BatchKernels#halvesWithBlockSums} over {@code blocks} blocks. */
    private static WorkerGrid halvesWorker(int blocks) {
        int groups = (blocks + Lfm2BatchKernels.HALVES_BLOCKS - 1) / Lfm2BatchKernels.HALVES_BLOCKS;
        return WorkerGridFactory.genericWorker(
                groups * Lfm2BatchKernels.HALVES_THREADS, Lfm2BatchKernels.HALVES_THREADS);
    }

    /** The grid of projection {@code task}, none for a cuBLAS GEMM (its decode task has one). */
    private void projectionGrid(GridScheduler scheduler, String task, TornadoTensor w, int rows) {
        if (!cublasForm(w)) {
            scheduler.addWorkerGrid(task, projections.worker(w, rows));
        }
    }
}
