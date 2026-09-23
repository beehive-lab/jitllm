package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4AttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The batched-prefill transformer layers for Gemma 4, Q8_0 weights, on the tensor cores.
 *
 * <p><b>Why this exists.</b> Until it did, this family's prompt processing was its decode loop run
 * once per prompt token, so every token read every weight: five hundred and twelve sweeps of the
 * model to ingest five hundred and twelve tokens. A chunk-wide graph reads each weight once for the
 * whole chunk, which is the entire difference between a matrix-vector product and a matrix-matrix
 * one, and it is why the reference implementation is two orders of magnitude faster at this and not
 * at decode.
 *
 * <p><b>What is this family's and what is not.</b> The GEMMs are the repository's own — {@code
 * gemmMMAQ8}, {@code gemmMMAQKVQ8}, {@code gemmMMAGateUpQ8} and {@code gemmMMA} are written in
 * terms of M, N and K and know nothing about any architecture — and so are the RMS reductions and
 * the FP32-to-FP16 cast. Everything that is Gemma 4's is in {@link Gemma4BatchPrefillKernels}: the
 * sandwich norms, the query/key/value head norms, the table-driven NeoX rotation into this family's
 * flat KV cache, the sliding window, the GeGLU, and the per-layer-embedding block.
 *
 * <p><b>Two head widths and two feed-forward widths.</b> Four layers in five attend through a
 * 256-wide head and a 512-position window; the fifth attends through a 512-wide head over the whole
 * context. Blocks 0-14 have a 6144-wide feed-forward and blocks 15-34 a 12288-wide one. Every
 * chunk-wide buffer is therefore allocated at the widest and addressed with the layer's own stride,
 * and every worker grid is built per layer rather than once — a grid keyed on a task name is wrong
 * the moment one name maps to two shapes.
 *
 * <p><b>Twenty layers project no key and no value.</b> {@code shared_kv_layers} is 20, so fifteen
 * layers own a cache and the rest read an earlier layer's. Those twenty run the query projection
 * alone, rotate the query alone, and their attention addresses the cache their source layer filled.
 *
 * <p><b>Numerically this is not the decode path.</b> The GEMM operands are FP16 with FP32
 * accumulation, and the two per-layer-embedding projections — which this file's weights hold in
 * FP32 — are narrowed to FP16 once at construction so they can be a GEMM's B operand at all. That
 * is a new approximation on the prefill path and it is gated as one, at the decision level, not by
 * a bound that was widened to admit it.
 */
// @formatter:on
public class Gemma4BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeProjections() {
        return "FP16 tensor-core MMA (quantized weights dequantized where needed)";
    }

    /** One workgroup per token for the RMS reductions, as the other MMA prefill families use. */
    private static final int RMS_LOCAL_SIZE = 256;

    /** Lanes per (row, head) in the per-head norms and in attention. */
    private static final int HEAD_LOCAL_SIZE = 128;

    // @formatter:off
    /**
     * Which of the two attention kernels this plan dispatches.
     *
     * <p>An exact-comparison switch, not a tuning knob: the two are bit-identical by construction
     * and the property exists so one build can measure both. The retained kernel is the staged one.
     */
    // @formatter:on
    private static final boolean STAGED_ATTENTION =
            !Boolean.getBoolean("jitllm.gemma4.unstagedAttention");

    // @formatter:off
    /**
     * Depth slices for the two projections whose output is too narrow to fill the device.
     *
     * <p>Four: it takes {@code w2Proj} and {@code woProj} from forty-eight thread blocks to a
     * hundred and ninety-two on a hundred-and-twenty-eight-SM device, and four divides every depth
     * they are given — 6144 and 12288 for the feed-forward, 2048 and 4096 for the attention output
     * — into whole 32-weight blocks. A property selects one slice, which is the unsplit kernel, for
     * exact comparison; not a tuning knob.
     */
    // @formatter:on
    public static final int SPLIT_K_SLICES = 4;

    private static final boolean SPLIT_K = !Boolean.getBoolean("jitllm.gemma4.noSplitK");

    // @formatter:off
    /**
     * Whether a projection's weights are decoded to FP16 before its GEMM.
     *
     * <p>Exact-comparison switches, not tuning knobs. The Q8_0 GEMMs convert every staged operand
     * in software because TornadoVM reaches the hardware conversion only through a store to a half
     * array (TornadoVM #1096, #1097), and that conversion is most of what those kernels do.
     */
    // @formatter:on
    private static final boolean DEQUANT_GATE_UP =
            !Boolean.getBoolean("jitllm.gemma4.noDequantGateUp");

    private static final boolean DEQUANT_PROJECTIONS =
            !Boolean.getBoolean("jitllm.gemma4.noDequantProjections");

    /** The split-K pair separately, so its own A/B does not also revert the query/key/value one. */
    private static final boolean DEQUANT_SPLIT_K =
            DEQUANT_PROJECTIONS && !Boolean.getBoolean("jitllm.gemma4.noDequantSplitK");

    private final Gemma4State state;
    private final Gemma4TornadoWeights weights;
    private final Gemma4Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;

    /** Whether the state holds its key/value cache in half precision; decided at allocation. */
    private final boolean fp16KeyValue;

    private final int paddedBatch;

    private final int nHead;
    private final int nHeadKv;
    private final int kvMul;
    private final int dim;
    private final int nEmbdPerLayer;
    private final int perLayerTotal;
    private final float embedScale;
    private final float perLayerProjScale;
    private final float perLayerInputScale;

    /**
     * The two per-layer-embedding projections in FP16.
     *
     * <p>This file holds {@code inp_gate} and {@code proj} in FP32, and the tensor-core GEMM's B
     * operand is FP16. They are narrowed once here rather than per chunk: together they are 786,432
     * elements a layer, 55 MB across the trunk in FP16, which is less than one chunk's worth of
     * re-narrowing traffic and is paid at plan construction instead of inside the timed window.
     */
    private final HalfFloatArray[] pleGateF16;

    private final HalfFloatArray[] pleProjF16;

    /**
     * The per-layer model projection in FP16, narrowed the same way as the two above. The file
     * holds it as BF16, which materializes to F16 — but a file that held it as anything else would
     * have met an unchecked cast here instead of the refusal by name this class promises everywhere
     * else.
     */
    private final HalfFloatArray pleModelProjF16;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Gemma4BatchPrefillLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.fp16KeyValue = state.usesFp16KeyValueCache();
        this.paddedBatch = (batchSize + 127) & ~127;
        this.nHead = config.numberOfHeads();
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.kvMul = config.kvMul();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));

        int layers = config.numberOfLayers();
        this.pleGateF16 = new HalfFloatArray[layers];
        this.pleProjF16 = new HalfFloatArray[layers];
        for (int l = 0; l < layers; l++) {
            int pleElements = dim * nEmbdPerLayer;
            pleGateF16[l] =
                    narrowToF16(weights.perLayerInpGate[l], pleElements, "blk." + l + ".inp_gate");
            pleProjF16[l] = narrowToF16(weights.perLayerProj[l], pleElements, "blk." + l + ".proj");
        }
        this.pleModelProjF16 =
                narrowToF16(weights.perLayerModelProj, perLayerTotal * dim, "per_layer_model_proj");

        List<ImmutableTaskGraph> graphs = new ArrayList<>(layers);
        for (int l = 0; l < layers; l++) {
            graphs.add(createBatchPrefillLayerTaskGraph(l).snapshot());
        }
        this.layerITGs = List.copyOf(graphs);
    }

    // @formatter:off
    /**
     * The B operand of a tensor-core GEMM in FP16, from whatever the file holds.
     *
     * <p>Refused by name rather than by a fallthrough: reading a Q8_0 block layout as FP16 halves
     * is a plausible-looking activation and wrong output, and a quantization this method does not
     * know is a missing case and not a reason to guess.
     */
    // @formatter:on
    private static HalfFloatArray narrowToF16(TornadoTensor t, int elements, String name) {
        HalfFloatArray out = new HalfFloatArray(elements);
        switch (t.dataType()) {
            case F16 -> {
                var src = t.asHalfFloatArray();
                for (int i = 0; i < elements; i++) {
                    out.set(i, src.get(i));
                }
            }
            case F32 -> {
                var src = t.asFloatArray();
                for (int i = 0; i < elements; i++) {
                    out.set(i, new HalfFloat(src.get(i)));
                }
            }
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 batched prefill has no tensor-core operand for "
                                    + t.dataType()
                                    + " ("
                                    + name
                                    + "); the per-layer-embedding projections must be F32 or F16");
        }
        return out;
    }

    // @formatter:off
    /**
     * Refuses a projection this plan has no decoder for, by name.
     *
     * <p>Three block layouts reach the trunk: 34 bytes to thirty-two weights for Q8_0, 18 for Q4_0,
     * and 20 for Q4_1 — which this family needs for exactly four tensors, {@code ffn_down} on
     * blocks 0-3 of the Q4_0 file. Reading one as another is a plausible-looking activation and
     * wrong output, so an unknown representation is a missing case and not something to guess at.
     */
    // @formatter:on
    private static void requireDecodable(TornadoTensor t, String name) {
        DataType type = t.dataType();
        if (type != DataType.Q8_0 && type != DataType.Q4_0 && type != DataType.Q4_1) {
            throw new UnsupportedOperationException(
                    "gemma4 batched prefill has no decoder for "
                            + type
                            + " ("
                            + name
                            + "); the trunk must be Q8_0, Q4_0 or Q4_1");
        }
    }

    /** Whether every one of these is Q8_0, which is what the direct Q8_0 GEMMs require. */
    private static boolean allQ8(TornadoTensor... tensors) {
        for (TornadoTensor t : tensors) {
            if (t.dataType() != DataType.Q8_0) {
                return false;
            }
        }
        return true;
    }

    // @formatter:off
    /**
     * Decodes one projection into the shared FP16 scratch, choosing the decoder from the tensor's
     * own representation rather than from the model's.
     */
    // @formatter:on
    private void addDequant(TaskGraph layer, String taskName, TornadoTensor w, int destOffset) {
        switch (w.dataType()) {
            case Q8_0 ->
                    layer.task(
                            taskName,
                            Gemma4BatchPrefillKernels::dequantizeQ8ToFP16,
                            context,
                            w.asByteArray(),
                            state.workspace.weightsF16Scratch,
                            destOffset);
            case Q4_0 ->
                    layer.task(
                            taskName,
                            Gemma4BatchPrefillKernels::dequantizeQ4_0ToFP16,
                            context,
                            w.asByteArray(),
                            state.workspace.weightsF16Scratch,
                            destOffset);
            case Q4_1 ->
                    layer.task(
                            taskName,
                            Gemma4BatchPrefillKernels::dequantizeQ4_1ToFP16,
                            context,
                            w.asByteArray(),
                            state.workspace.weightsF16Scratch,
                            destOffset);
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 batched prefill has no decoder for " + w.dataType());
        }
    }

    // @formatter:off
    /**
     * Whether a layer's prefill attention runs on the tensor cores: over the FP16 cache, on a
     * tensor-core device, for heads a whole number of 256-wide halves wide.
     */
    // @formatter:on
    private boolean tensorCoreAttention(int layerIndex) {
        return fp16KeyValue
                && config.headDim(layerIndex) % Gemma4AttentionKernels.TC_HEAD == 0
                && org.beehive.jitllm.backend.tornado.TensorCoreSupport
                        .isTensorCoreCapableBackend();
    }

    /** The key cache in the representation the state allocated. */
    private Object keyCache() {
        return fp16KeyValue ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state allocated. */
    private Object valueCache() {
        return fp16KeyValue ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    // @formatter:off
    /**
     * This layer's RoPE pair: uploaded by the first layer of its kind, bound from that graph by
     * every later one.
     *
     * <p>From that graph and not the previous one: a graph that declares an array none of its tasks
     * reads never allocates it, and a sliding layer's graph never reads the full-attention pair.
     * Without this every layer graph held its own device copy of its pair.
     */
    // @formatter:on
    private void bindRopeTables(
            TaskGraph layer, boolean isSwa, FloatArray freqCisReal, FloatArray freqCisImag) {
        int firstUser = -1;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSwa(l) == isSwa) {
                firstUser = l;
                break;
            }
        }
        String graphName = layer.getTaskGraphName();
        if (graphName.equals("batchPrefillLayer_" + firstUser)) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, freqCisReal, freqCisImag);
        } else {
            layer.consumeFromDevice("batchPrefillLayer_" + firstUser, freqCisReal, freqCisImag);
        }
    }

    /** The packed [q|k|v] row stride this layer's projection writes. */
    private int qkvStride(int layerIndex) {
        int headDim = config.headDim(layerIndex);
        int qDim = nHead * headDim;
        return config.hasOwnKv(layerIndex) ? qDim + 2 * nHeadKv * headDim : qDim;
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);

        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int qDim = nHead * headDim;
        final int kvDim = nHeadKv * headDim;
        final int ffnLen = config.feedForwardLength(layerIndex);
        final int cacheBaseOffset = state.cacheLayerBaseOffset[layerIndex];
        final int windowSize = isSwa ? config.slidingWindowSize() : config.contextLength();
        final var freqCisReal =
                (isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull).asFloatArray();
        final var freqCisImag =
                (isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull).asFloatArray();
        final int peOffset = layerIndex * nEmbdPerLayer;
        final int stride = qkvStride(layerIndex);

        requireDecodable(weights.wqLayered[layerIndex], "blk." + layerIndex + ".attn_q");
        requireDecodable(weights.woLayered[layerIndex], "blk." + layerIndex + ".attn_output");
        requireDecodable(weights.w1Layered[layerIndex], "blk." + layerIndex + ".ffn_gate");
        requireDecodable(weights.w3Layered[layerIndex], "blk." + layerIndex + ".ffn_up");
        requireDecodable(weights.w2Layered[layerIndex], "blk." + layerIndex + ".ffn_down");

        // ── Transfers ──────────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapPerLayerTokenEmbedRowBatch);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.branchScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    state.workspace.splitKPartialBatch,
                    state.workspace.weightsF16Scratch,
                    state.workspace.woOut,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    keyCache(),
                    valueCache(),
                    state.workspace.wrapXFP16Batch,
                    state.workspace.wrapPerLayerInputsBatch,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    state.workspace.wrapPerLayerGateBatch,
                    state.workspace.wrapPerLayerGateFP16Batch,
                    state.workspace.wrapPerLayerOutBatch);
            if (fp16KeyValue) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        state.workspace.attnOutF32Batch,
                        state.workspace.attnProbStageBatch);
            }
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    pleModelProjF16,
                    weights.perLayerProjNorm.asFloatArray());
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.branchScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    state.workspace.splitKPartialBatch,
                    state.workspace.weightsF16Scratch,
                    state.workspace.woOut,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            layer.consumeFromDevice(
                    pred,
                    keyCache(),
                    valueCache(),
                    state.workspace.wrapXFP16Batch,
                    state.workspace.wrapPerLayerInputsBatch,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    state.workspace.wrapPerLayerGateBatch,
                    state.workspace.wrapPerLayerGateFP16Batch,
                    state.workspace.wrapPerLayerOutBatch);
            if (fp16KeyValue) {
                // From the graph that allocated them, not the previous one: the full-attention
                // layers never hand them to a task.
                layer.consumeFromDevice(
                        "batchPrefillLayer_0",
                        state.workspace.attnOutF32Batch,
                        state.workspace.attnProbStageBatch);
            }
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.attnQNorm[layerIndex].asFloatArray(),
                weights.attnPostNorm[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                pleGateF16[layerIndex],
                pleProjF16[layerIndex],
                weights.perLayerPostNorm[layerIndex].asFloatArray());
        if (hasOwnKv) {
            requireDecodable(weights.wkLayered[layerIndex], "blk." + layerIndex + ".attn_k");
            requireDecodable(weights.wvLayered[layerIndex], "blk." + layerIndex + ".attn_v");
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    weights.attnKNorm[layerIndex].asFloatArray());
        }
        if (weights.layerOutputScale[layerIndex] != null) {
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }

        bindRopeTables(layer, isSwa, freqCisReal, freqCisImag);
        if (layerIndex == 0) {
            appendPleSetup(layer);
        }

        // ── Attention ──────────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        if (hasOwnKv) {
            boolean qkvDirect =
                    !DEQUANT_PROJECTIONS
                            && allQ8(
                                    weights.wqLayered[layerIndex],
                                    weights.wkLayered[layerIndex],
                                    weights.wvLayered[layerIndex]);
            if (!qkvDirect) {
                // Query, key and value laid end to end make one [qDim + 2*kvDim, dim] matrix, so a
                // single plain GEMM writes exactly the packed [q|k|v] row the head norms and the
                // rotation already read.
                addDequant(layer, "qDequant", weights.wqLayered[layerIndex], 0);
                addDequant(layer, "kDequant", weights.wkLayered[layerIndex], qDim * dim);
                addDequant(layer, "vDequant", weights.wvLayered[layerIndex], (qDim + kvDim) * dim);
                layer.task(
                        "qkvProj",
                        TransformerBatchPrefillKernels::gemmMMA,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.weightsF16Scratch,
                        state.workspace.qkvResultBatch,
                        paddedBatch,
                        qDim + 2 * kvDim,
                        dim);
            } else {
                layer.task(
                        "qkvProj",
                        TransformerBatchPrefillKernels::gemmMMAQKVQ8,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        weights.wqLayered[layerIndex].asByteArray(),
                        weights.wkLayered[layerIndex].asByteArray(),
                        weights.wvLayered[layerIndex].asByteArray(),
                        state.workspace.qkvResultBatch,
                        paddedBatch,
                        qDim,
                        kvDim,
                        dim);
            }
            layer.task(
                    "batch_qkv_norm",
                    Gemma4BatchPrefillKernels::batchedQkvHeadNorms,
                    context,
                    state.workspace.qkvResultBatch,
                    weights.attnQNorm[layerIndex].asFloatArray(),
                    weights.attnKNorm[layerIndex].asFloatArray(),
                    nHead,
                    nHeadKv,
                    headDim,
                    stride,
                    HEAD_LOCAL_SIZE,
                    config.rmsNormEps());
            if (fp16KeyValue) {
                layer.task(
                        "batch_rope_kv",
                        Gemma4AttentionKernels::batchedRopeAndCacheFP16,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        freqCisReal,
                        freqCisImag,
                        nHead,
                        nHeadKv,
                        headDim,
                        kvDim,
                        stride,
                        cacheBaseOffset);
            } else {
                layer.task(
                        "batch_rope_kv",
                        Gemma4BatchPrefillKernels::batchedRopeAndCache,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        freqCisReal,
                        freqCisImag,
                        nHead,
                        nHeadKv,
                        headDim,
                        kvDim,
                        stride,
                        cacheBaseOffset);
            }
        } else {
            if (!DEQUANT_PROJECTIONS && allQ8(weights.wqLayered[layerIndex])) {
                layer.task(
                        "qkvProj",
                        TransformerBatchPrefillKernels::gemmMMAQ8,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        weights.wqLayered[layerIndex].asByteArray(),
                        state.workspace.qkvResultBatch,
                        paddedBatch,
                        qDim,
                        dim);
            } else {
                addDequant(layer, "qDequant", weights.wqLayered[layerIndex], 0);
                layer.task(
                        "qkvProj",
                        TransformerBatchPrefillKernels::gemmMMA,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.weightsF16Scratch,
                        state.workspace.qkvResultBatch,
                        paddedBatch,
                        qDim,
                        dim);
            }
            layer.task(
                    "batch_qkv_norm",
                    Gemma4BatchPrefillKernels::batchedQHeadNorm,
                    context,
                    state.workspace.qkvResultBatch,
                    weights.attnQNorm[layerIndex].asFloatArray(),
                    nHead,
                    headDim,
                    stride,
                    HEAD_LOCAL_SIZE,
                    config.rmsNormEps());
            layer.task(
                    "batch_rope_kv",
                    Gemma4BatchPrefillKernels::batchedRopeQOnly,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    freqCisReal,
                    freqCisImag,
                    nHead,
                    headDim,
                    stride);
        }

        if (tensorCoreAttention(layerIndex)) {
            layer.task(
                    "batch_attention",
                    Gemma4AttentionKernels::attentionPrefillTensorCoreFP16,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutF32Batch,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    state.workspace.attnProbStageBatch,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    stride,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    Gemma4AttentionKernels.tcScoreKeys(windowSize, config.contextLength()));
        } else if (fp16KeyValue) {
            layer.task(
                    "batch_attention",
                    Gemma4AttentionKernels::batchedSlidingWindowAttentionStagedFP16,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    stride,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    HEAD_LOCAL_SIZE);
        } else {
            layer.task(
                    "batch_attention",
                    STAGED_ATTENTION
                            ? Gemma4BatchPrefillKernels::batchedSlidingWindowAttentionStaged
                            : Gemma4BatchPrefillKernels::batchedSlidingWindowAttention,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    stride,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    HEAD_LOCAL_SIZE);
        }

        if (SPLIT_K && (DEQUANT_SPLIT_K || !allQ8(weights.woLayered[layerIndex]))) {
            addDequant(layer, "woDequant", weights.woLayered[layerIndex], 0);
            layer.task(
                    "woProj",
                    Gemma4BatchPrefillKernels::gemmMMASplitK,
                    context,
                    state.workspace.attnOutFP16,
                    state.workspace.weightsF16Scratch,
                    state.workspace.splitKPartialBatch,
                    paddedBatch,
                    dim,
                    qDim,
                    SPLIT_K_SLICES);
            layer.task(
                    "woReduce",
                    Gemma4BatchPrefillKernels::splitKReduce,
                    context,
                    state.workspace.splitKPartialBatch,
                    state.workspace.woOut,
                    paddedBatch * dim,
                    SPLIT_K_SLICES);
        } else if (SPLIT_K) {
            layer.task(
                    "woProj",
                    Gemma4BatchPrefillKernels::gemmMMAQ8SplitK,
                    context,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asByteArray(),
                    state.workspace.splitKPartialBatch,
                    paddedBatch,
                    dim,
                    qDim,
                    SPLIT_K_SLICES);
            layer.task(
                    "woReduce",
                    Gemma4BatchPrefillKernels::splitKReduce,
                    context,
                    state.workspace.splitKPartialBatch,
                    state.workspace.woOut,
                    paddedBatch * dim,
                    SPLIT_K_SLICES);
        } else {
            layer.task(
                    "woProj",
                    TransformerBatchPrefillKernels::gemmMMAQ8,
                    context,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asByteArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    qDim);
        }
        layer.task(
                "batch_post_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.woOut,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_post_attn_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                weights.attnPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        // ── Feed-forward ───────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        if (DEQUANT_GATE_UP
                || !allQ8(weights.w1Layered[layerIndex], weights.w3Layered[layerIndex])) {
            // Gate into the first half of the scratch and up into the second, so the pair is one
            // contiguous [2*ffnLen, dim] matrix and one GEMM produces the packed [gate|up] rows the
            // activation expects — no second kernel, and the same output layout as before.
            addDequant(layer, "gateDequant", weights.w1Layered[layerIndex], 0);
            addDequant(layer, "upDequant", weights.w3Layered[layerIndex], ffnLen * dim);
            layer.task(
                    "gateUpProj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.normedXFFNFP16,
                    state.workspace.weightsF16Scratch,
                    state.workspace.gateUpResultBatch,
                    paddedBatch,
                    2 * ffnLen,
                    dim);
        } else {
            layer.task(
                    "gateUpProj",
                    TransformerBatchPrefillKernels::gemmMMAGateUpQ8,
                    context,
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    state.workspace.gateUpResultBatch,
                    paddedBatch,
                    ffnLen,
                    dim);
        }
        layer.task(
                "batch_geglu",
                Gemma4BatchPrefillKernels::batchedGeGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                ffnLen);
        if (SPLIT_K && (DEQUANT_SPLIT_K || !allQ8(weights.w2Layered[layerIndex]))) {
            addDequant(layer, "w2Dequant", weights.w2Layered[layerIndex], 0);
            layer.task(
                    "w2Proj",
                    Gemma4BatchPrefillKernels::gemmMMASplitK,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.weightsF16Scratch,
                    state.workspace.splitKPartialBatch,
                    paddedBatch,
                    dim,
                    ffnLen,
                    SPLIT_K_SLICES);
            layer.task(
                    "w2Reduce",
                    Gemma4BatchPrefillKernels::splitKReduce,
                    context,
                    state.workspace.splitKPartialBatch,
                    state.workspace.w2Out,
                    paddedBatch * dim,
                    SPLIT_K_SLICES);
        } else if (SPLIT_K) {
            layer.task(
                    "w2Proj",
                    Gemma4BatchPrefillKernels::gemmMMAQ8SplitK,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asByteArray(),
                    state.workspace.splitKPartialBatch,
                    paddedBatch,
                    dim,
                    ffnLen,
                    SPLIT_K_SLICES);
            layer.task(
                    "w2Reduce",
                    Gemma4BatchPrefillKernels::splitKReduce,
                    context,
                    state.workspace.splitKPartialBatch,
                    state.workspace.w2Out,
                    paddedBatch * dim,
                    SPLIT_K_SLICES);
        } else {
            layer.task(
                    "w2Proj",
                    TransformerBatchPrefillKernels::gemmMMAQ8,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asByteArray(),
                    state.workspace.w2Out,
                    paddedBatch,
                    dim,
                    ffnLen);
        }
        layer.task(
                "batch_post_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.w2Out,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_post_ffn_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out,
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        // ── Per-layer embedding ────────────────────────────────────────────────
        layer.task(
                "batch_x_cast",
                TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapXFP16Batch);
        layer.task(
                "pleGateProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapXFP16Batch,
                pleGateF16[layerIndex],
                state.workspace.wrapPerLayerGateBatch,
                paddedBatch,
                nEmbdPerLayer,
                dim);
        layer.task(
                "batch_ple_gate_gelu",
                Gemma4BatchPrefillKernels::batchedPleGateGeluMul,
                context,
                state.workspace.wrapPerLayerGateFP16Batch,
                state.workspace.wrapPerLayerGateBatch,
                state.workspace.wrapPerLayerInputsBatch,
                peOffset,
                nEmbdPerLayer,
                perLayerTotal);
        layer.task(
                "pleProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapPerLayerGateFP16Batch,
                pleProjF16[layerIndex],
                state.workspace.wrapPerLayerOutBatch,
                paddedBatch,
                dim,
                nEmbdPerLayer);
        layer.task(
                "batch_ple_post_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapPerLayerOutBatch,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ple_post_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapPerLayerOutBatch,
                weights.perLayerPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        if (weights.layerOutputScale[layerIndex] != null) {
            layer.task(
                    "batch_layer_output_scale",
                    Gemma4BatchPrefillKernels::batchedScaleInPlaceFromTensor,
                    context,
                    state.workspace.wrapXBatch,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }

        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache);
        return layer;
    }

    /**
     * The chunk-wide form of layer 0's one-time-per-token per-layer-embedding setup: scale the
     * embeddings, project them to the per-layer inputs, normalize each layer's segment, and merge
     * with the host-gathered per-layer token embedding rows.
     */
    private void appendPleSetup(TaskGraph layer) {
        layer.task(
                "batch_scale_embedding",
                Gemma4BatchPrefillKernels::batchedScaleInPlace,
                context,
                state.workspace.wrapXBatch,
                embedScale);
        layer.task(
                "batch_embed_cast",
                TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapXFP16Batch);
        layer.task(
                "pleModelProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapXFP16Batch,
                pleModelProjF16,
                state.workspace.wrapPerLayerProjScratchBatch,
                paddedBatch,
                perLayerTotal,
                dim);
        layer.task(
                "batch_ple_proj_scale_norm",
                Gemma4BatchPrefillKernels::batchedPleProjScaleAndNormalize,
                context,
                state.workspace.wrapPerLayerProjScratchBatch,
                weights.perLayerProjNorm.asFloatArray(),
                nEmbdPerLayer,
                HEAD_LOCAL_SIZE,
                perLayerProjScale,
                config.rmsNormEps());
        layer.task(
                "batch_ple_merge",
                Gemma4BatchPrefillKernels::batchedAddAndScale,
                context,
                state.workspace.wrapPerLayerInputsBatch,
                state.workspace.wrapPerLayerProjScratchBatch,
                state.workspace.wrapPerLayerTokenEmbedRowBatch,
                perLayerInputScale);
    }

    // @formatter:on

    /** The {@code gemmMMA} family: 256 threads per block, one block per (M-tile, N-tile). */
    private static WorkerGrid mmaGrid(int paddedM, int n) {
        WorkerGrid2D g = new WorkerGrid2D((paddedM / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** The split-K family: the {@code gemmMMA} grid with a third dimension, one slice per plane. */
    private static WorkerGrid mmaSplitKGrid(int paddedM, int n) {
        WorkerGrid3D g = new WorkerGrid3D((paddedM / 128) * 256, n / 128, SPLIT_K_SLICES);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid elementwise(int n, int local) {
        int l = Math.min(local, n);
        while (l > 1 && n % l != 0) {
            l--;
        }
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(l, 1, 1);
        return g;
    }

    // @formatter:off
    /**
     * Worker grids, built per layer.
     *
     * <p>Per layer and not once: this family's head width, feed-forward width and whether a layer
     * projects a key at all differ between layers sharing a task name, and a grid keyed on a task
     * name is wrong the moment one name maps to two shapes.
     */
    // @formatter:on
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid dimApplyWorker = elementwise(batchSize * dim, 256);
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim);
        WorkerGrid pleGateWorker = mmaGrid(paddedBatch, nEmbdPerLayer);
        WorkerGrid pleGateGeluWorker = elementwise(batchSize * nEmbdPerLayer, 256);
        WorkerGrid reduceWorker = elementwise(paddedBatch * dim, 256);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            String p = "batchPrefillLayer_" + l + ".";
            int headDim = config.headDim(l);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(l);
            boolean hasOwnKv = config.hasOwnKv(l);

            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", dimApplyWorker);
            scheduler.addWorkerGrid(
                    p + "qkvProj", mmaGrid(paddedBatch, hasOwnKv ? qDim + 2 * kvDim : qDim));
            boolean qkvDirect =
                    !DEQUANT_PROJECTIONS
                            && (hasOwnKv
                                    ? allQ8(
                                            weights.wqLayered[l],
                                            weights.wkLayered[l],
                                            weights.wvLayered[l])
                                    : allQ8(weights.wqLayered[l]));
            if (!qkvDirect) {
                scheduler.addWorkerGrid(p + "qDequant", elementwise(qDim * dim, 256));
                if (hasOwnKv) {
                    scheduler.addWorkerGrid(p + "kDequant", elementwise(kvDim * dim, 256));
                    scheduler.addWorkerGrid(p + "vDequant", elementwise(kvDim * dim, 256));
                }
            }

            int normSlots = hasOwnKv ? nHead + 2 * nHeadKv : nHead;
            scheduler.addWorkerGrid(
                    p + "batch_qkv_norm",
                    WorkerGridFactory.genericWorker(
                            batchSize * normSlots * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
            scheduler.addWorkerGrid(
                    p + "batch_rope_kv", elementwise(batchSize * nHead * (headDim / 2), 256));
            // Over the PADDED rows, not the real ones. The output projection is a GEMM at
            // M = paddedBatch and reads every row of attnOutFP16; launching attention at batchSize
            // would leave the rows between the two untouched, and the kernel's own guard — which
            // writes zeros and returns — would never run for them. It matters only when the chunk
            // width is not a multiple of 128, which nothing rounds it to.
            scheduler.addWorkerGrid(
                    p + "batch_attention",
                    tensorCoreAttention(l)
                            ? WorkerGridFactory.genericWorker(
                                    paddedBatch
                                            / Gemma4AttentionKernels.TC_QUERIES
                                            * nHead
                                            * Gemma4AttentionKernels.TC_LANES,
                                    Gemma4AttentionKernels.TC_LANES)
                            : WorkerGridFactory.genericWorker(
                                    paddedBatch * nHead * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
            if (SPLIT_K) {
                scheduler.addWorkerGrid(p + "woProj", mmaSplitKGrid(paddedBatch, dim));
                scheduler.addWorkerGrid(p + "woReduce", reduceWorker);
                if (DEQUANT_SPLIT_K || !allQ8(weights.woLayered[l])) {
                    scheduler.addWorkerGrid(p + "woDequant", elementwise(dim * qDim, 256));
                }
            } else {
                scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_post_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_post_attn_apply", dimApplyWorker);

            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", dimApplyWorker);
            scheduler.addWorkerGrid(p + "gateUpProj", mmaGrid(paddedBatch, 2 * ffnLen));
            if (DEQUANT_GATE_UP || !allQ8(weights.w1Layered[l], weights.w3Layered[l])) {
                WorkerGrid dequantWorker = elementwise(ffnLen * dim, 256);
                scheduler.addWorkerGrid(p + "gateDequant", dequantWorker);
                scheduler.addWorkerGrid(p + "upDequant", dequantWorker);
            }
            scheduler.addWorkerGrid(p + "batch_geglu", elementwise(batchSize * ffnLen, 256));
            if (SPLIT_K) {
                scheduler.addWorkerGrid(p + "w2Proj", mmaSplitKGrid(paddedBatch, dim));
                scheduler.addWorkerGrid(p + "w2Reduce", reduceWorker);
                if (DEQUANT_SPLIT_K || !allQ8(weights.w2Layered[l])) {
                    scheduler.addWorkerGrid(p + "w2Dequant", elementwise(dim * ffnLen, 256));
                }
            } else {
                scheduler.addWorkerGrid(p + "w2Proj", mmaDimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_post_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_post_ffn_apply", dimApplyWorker);

            scheduler.addWorkerGrid(p + "batch_x_cast", dimApplyWorker);
            scheduler.addWorkerGrid(p + "pleGateProj", pleGateWorker);
            scheduler.addWorkerGrid(p + "batch_ple_gate_gelu", pleGateGeluWorker);
            scheduler.addWorkerGrid(p + "pleProj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_ple_post_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ple_post_apply", dimApplyWorker);
            if (weights.layerOutputScale[l] != null) {
                scheduler.addWorkerGrid(p + "batch_layer_output_scale", dimApplyWorker);
            }
        }

        String p0 = "batchPrefillLayer_0.";
        scheduler.addWorkerGrid(p0 + "batch_scale_embedding", dimApplyWorker);
        scheduler.addWorkerGrid(p0 + "batch_embed_cast", dimApplyWorker);
        scheduler.addWorkerGrid(p0 + "pleModelProj", mmaGrid(paddedBatch, perLayerTotal));
        scheduler.addWorkerGrid(
                p0 + "batch_ple_proj_scale_norm",
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfLayers() * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p0 + "batch_ple_merge", elementwise(batchSize * perLayerTotal, 256));
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
