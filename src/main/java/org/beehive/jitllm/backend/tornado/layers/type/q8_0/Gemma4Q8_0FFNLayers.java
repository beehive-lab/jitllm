package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4AttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0Packed;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.Gemma4BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Gemma4Q8_0FFNLayers: Q8_0 transformer-layer task graphs for the Gemma 4 architecture.
 *
 * <p>Structurally identical to {@code Gemma4FP16FFNLayers}. The main attention/FFN projections
 * (Q/K/V/O, FFN gate/up/down) are Q8_0 byte arrays consumed via {@code matrixVectorGenericQ8Byte} /
 * {@link Gemma4Kernels#fusedGateUpGeGLUQ8}. The per-layer-embedding (PLE) projections ({@code
 * inp_gate}, {@code proj}, {@code per_layer_model_proj}) are <em>not</em> uniformly Q8_0 in a Q8_0
 * GGUF -- they may be stored at F32 or F16 -- so they are routed through {@link #addProjection}
 * which selects the matmul kernel from each tensor's {@link TornadoTensor#type()}. The
 * (un-quantized) norm and scale weights remain F32.
 *
 * <p>Gemma 4's layers differ enough from the "Llama-like" models that nothing here is fused the way
 * {@code Qwen3FP16FFNLayers} is -- each layer carries its own Q/K-norm and a "sandwich" of pre/post
 * normalization around both attention and FFN, attention head dimensions and RoPE tables differ
 * between sliding-window and full-attention layers (and are baked into each layer's task graph as
 * compile-time constants -- see {@link Gemma4Configuration#headDim}), some layers reuse an earlier
 * layer's KV cache instead of computing their own, the FFN uses GeGLU, and every layer mixes in a
 * per-layer embedding (PLE) contribution. See {@link InferenceCore#forwardJavaGemma4} for the
 * reference computation each task mirrors.
 *
 * <p>Layer 0's task graph additionally carries one-time-per-token setup that the reference
 * implementation performs before the layer loop: scaling the token embedding by {@code sqrt(dim)},
 * and computing the per-layer-embedding inputs ({@code perLayerInputs}) from the per-layer model
 * projection and the (host-gathered) per-layer token embedding row -- see {@link
 * #appendPLESetupTasks}.
 */
public class Gemma4Q8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Gemma4TornadoWeights, Gemma4Configuration> {

    /**
     * Local memory size for per-head Q/K/V-norm reductions; must evenly divide both head dimensions
     * (256, 512).
     */
    private static final int HEAD_NORM_LOCAL_SIZE = 64;

    /**
     * Lanes' worth of reduction scratch the attention kernel allocates.
     *
     * <p>The invariant is {@code ATTENTION_LOCAL_SIZE >= } the launched workgroup size, not
     * equality with it. It is passed as the kernel's {@code localMemSize} and the kernel derives
     * every bound from {@code context.localGroupSizeX}, so a value above the launch over-allocates
     * shared memory and a value below it corrupts the reduction. {@code createAttentionWorker}
     * picks {@code min(headDim, 64)} for the single-pass kernel, so 256 is slack, not a match — an
     * earlier comment here claimed they had to be equal, which would have invited someone to lower
     * this to 64 and under-size the scratch for the split kernel, which does launch at 256.
     */
    private static final int ATTENTION_LOCAL_SIZE = 256;

    private final Gemma4State gemma4State;
    private final int nHead;
    private final int nHeadKv;
    private final int kvMul;
    private final int dim;
    private final int nEmbdPerLayer;
    private final int perLayerTotal;
    private final float embedScale;
    private final float perLayerTokEmbedScale;
    private final float perLayerProjScale;
    private final float perLayerInputScale;

    // @formatter:off
    /**
     * Whether these are the decode layers of the batched plan.
     *
     * <p>A flag rather than a subclass: the two differences are which graph layer 0 names as the
     * producer of its activation and of the key/value caches, and a class per plan for a difference
     * that small is what the dispatch ledger exists to stop.
     */
    // @formatter:on
    private final boolean batchedPlan;

    /** Whether the state holds its key/value cache in half precision; decided at allocation. */
    private final boolean fp16KeyValue;

    /** The graph a later stage's first layer takes its activation from; null on the first stage. */
    private final String stageInput;

    public Gemma4Q8_0FFNLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            SchedulerType schedulerType) {
        this(state, weights, config, schedulerType, false);
    }

    public Gemma4Q8_0FFNLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            SchedulerType schedulerType,
            boolean batchedPlan) {
        this(state, weights, config, schedulerType, batchedPlan, 0, config.numberOfLayers(), null);
    }

    // @formatter:off
    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     *
     * <p>A stage after the first takes its activation from {@code stageInput}, the graph that
     * received it from the previous stage, and in a batched plan its key/value cache from its own
     * last batch-prefill graph. The cache keeps the whole model's offsets ({@code
     * Gemma4State.cacheLayerBaseOffset}), so a stage addresses its layers as on one device. A model
     * with per-layer input embeddings is not split: layer 0 computes them for every layer.
     */
    // @formatter:on
    public Gemma4Q8_0FFNLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            SchedulerType schedulerType,
            boolean batchedPlan,
            int firstLayer,
            int endLayer,
            String stageInput) {
        super(state, weights, config, schedulerType);
        if (firstLayer > 0 && config.hasPerLayerEmbeddings()) {
            throw new UnsupportedOperationException(
                    "a gemma4 model with per-layer input embeddings is not split across devices:"
                            + " layer 0 computes them for every layer");
        }
        restrictToLayers(firstLayer, endLayer);
        this.stageInput = stageInput;
        this.batchedPlan = batchedPlan;
        this.gemma4State = state;
        this.fp16KeyValue = state.usesFp16KeyValueCache();
        this.nHead = config.numberOfHeads();
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.kvMul = config.kvMul();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerTokEmbedScale = (float) Math.sqrt(nEmbdPerLayer);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));
        setupFFNLayers();
    }

    /**
     * Whether a layer's decode attention takes the grouped kernel over the half-precision cache:
     * the shape it is written for. Other layers over that cache take the split-window kernel.
     */
    private boolean groupDecode(int layerIndex) {
        return fp16KeyValue
                && Gemma4AttentionKernels.decodeGroupFits(
                        config.kvMul(layerIndex), config.headDimSwa(), config.headDimFull());
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                  TASK GRAPH
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * Lanes per output row for a row-per-workgroup projection.
     *
     * <p>One warp per row leaves a 1536-row projection with 1536 warps, which is not enough
     * resident work to hide DRAM latency on this device. Measured with Nsight Compute on
     * gemma-4-E2B-it-Q8_0: {@code matrixVectorGenericQ8Byte} at grid 1536 x 32 reached 27.4% of
     * DRAM peak, while {@code fusedGateUpGeGLUQ8} at grid 6144 x 32 reached 74.9% running the
     * identical inner routine. The routine already reduces over an arbitrary block, so the fix is
     * the launch shape, not the kernel.
     *
     * <p>Derived from the projection's own row count, decided at graph build. Wide projections
     * already have the warps and keep the narrow block.
     */
    private static int projectionLocalSize(int rows) {
        if (rows >= 4096) {
            return LOCAL_WORK_GROUP_SIZE_ALLOC;
        }
        return rows >= 1024 ? 128 : 256;
    }

    /**
     * Whether the packed-integer projections may run here: the device must lower {@code dp4a} AND
     * evaluate {@code simdShuffleDown} correctly ({@code PACKED_INTEGER_DOT} bundles the two
     * deliberately), and the weights must be the representation those kernels decode.
     */
    private boolean packedFor(int layerIndex) {
        DataType type = weights.wqLayered[layerIndex].dataType();
        return (type == DataType.Q4_0
                        || type == DataType.Q8_0
                        || type == DataType.Q4_1
                        || type == DataType.Q4_K
                        || type == DataType.Q5_K
                        || type == DataType.Q6_K)
                && packedIntegerDot();
    }

    /**
     * Whether a layer projects gate and up apart and multiplies them by their own task: they are
     * not one representation with a fused GeGLU kernel (Q8_0, Q4_0). A K-quant file's layers do.
     */
    private boolean splitGateUp(int layerIndex) {
        DataType gate = weights.w1Layered[layerIndex].dataType();
        return gate != weights.w3Layered[layerIndex].dataType()
                || (gate != DataType.Q8_0 && gate != DataType.Q4_0);
    }

    /** Whether some layer of this stage runs gate and up apart, which needs {@code wrapHb2}. */
    private boolean anySplitGateUp() {
        for (int i = firstLayer; i < endLayer(config.numberOfLayers()); i++) {
            if (splitGateUp(i)) {
                return true;
            }
        }
        return false;
    }

    /** Whether this device runs the packed-integer projections at all. */
    private static boolean packedIntegerDot() {
        // An escape hatch for exact comparison, matching the one qwen35 keeps: the packed path
        // quantizes the activation to eight bits, so a build that needs to be compared against one
        // that does not needs to be able to turn it off. Not a tuning knob.
        if (!"true"
                .equalsIgnoreCase(System.getProperty("jitllm.gemma4.packedIntegerDot", "true"))) {
            return false;
        }
        return TornadoDevices.target().capabilities().supports(DeviceCapability.PACKED_INTEGER_DOT);
    }

    /** How many slices the window is cut into, or 1 to run the single-pass kernel. */
    private int attentionSplits() {
        return config.attentionSplits();
    }

    /**
     * Adjacent layers to a decode graph.
     *
     * <p>Every task graph is one submission, and CUDA graphs do not change that -- they are applied
     * per graph, so thirty-five layer graphs stay thirty-five launches. Grouping divides the
     * submissions without building one graph for the whole trunk.
     */
    private static final int LAYERS_PER_GRAPH = 4;

    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    /** A task's name inside its graph. */
    private String tn(int layerIndex, String task) {
        return layerTaskPrefix(layerIndex) + task;
    }

    // @formatter:off
    /**
     * The layer graph before the one holding {@code layerIndex}, or {@code null} for the first.
     *
     * <p>Every import names its producer. TornadoVM resolves a graph's unnamed imports against the
     * graph that ran before it only when the graph has no named import at all, so one named import
     * — a RoPE table from the first layer of its kind, a weight from a batch-prefill graph — would
     * silently leave the activation and the scratch buffers unlinked.
     */
    // @formatter:on
    private String previousGraphName(int layerIndex) {
        int offset = layerIndex - firstLayer;
        int first = firstLayer + offset - offset % LAYERS_PER_GRAPH;
        return first == firstLayer ? null : layerGraphName(first - LAYERS_PER_GRAPH);
    }

    private boolean firstLayerOfGraph(int layerIndex) {
        return (layerIndex - firstLayer) % LAYERS_PER_GRAPH == 0;
    }

    private boolean lastLayerOfGraph(int layerIndex) {
        return (layerIndex - firstLayer) % LAYERS_PER_GRAPH == LAYERS_PER_GRAPH - 1
                || layerIndex == endLayer(config.numberOfLayers()) - 1;
    }

    /**
     * One graph per group of layers, in order, with a smaller graph for any remainder.
     *
     * <p>Overrides the one-graph-per-layer construction rather than generalising it: the grouping
     * is this family's, and every other family keeps the loop it had.
     */
    @Override
    protected void setupFFNLayers() {
        int layers = endLayer(config.numberOfLayers());
        java.util.List<uk.ac.manchester.tornado.api.ImmutableTaskGraph> graphs =
                new java.util.ArrayList<>();
        for (int first = firstLayer; first < layers; first += LAYERS_PER_GRAPH) {
            TaskGraph graph = new TaskGraph(layerGraphName(first));
            java.util.Arrays.fill(ropeBound, false);
            int last = Math.min(first + LAYERS_PER_GRAPH, layers) - 1;
            for (int layer = first; layer <= last; layer++) {
                appendLayer(graph, layer);
            }
            lastFFNLayerTaskGraphID = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
        ffnLayerITGs = java.util.List.copyOf(graphs);
    }

    /** Kept for the base class's contract; the grouped construction above is what runs. */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph graph = new TaskGraph(layerGraphName(layerIndex));
        appendLayer(graph, layerIndex);
        return graph;
    }

    protected void appendLayer(TaskGraph unifiedLayer, int layerIndex) {
        // Key/value heads by layer: the 31B has 16 on its sliding-window layers and 4 on the
        // global ones, E2B one count everywhere.
        final int nHeadKv = config.keyValueHeads(layerIndex);
        final int kvMul = config.kvMul(layerIndex);
        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int qDim = nHead * headDim;
        final int kvDim = nHeadKv * headDim;
        final int ffnLen = config.feedForwardLength(layerIndex);
        final int cacheBaseOffset = gemma4State.cacheLayerBaseOffset[layerIndex];
        final int windowSize = isSwa ? config.slidingWindowSize() : config.contextLength();
        final var freqCisReal =
                (isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull).asFloatArray();
        final var freqCisImag =
                (isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull).asFloatArray();
        final int peOffset = layerIndex * nEmbdPerLayer;

        if (firstLayerOfGraph(layerIndex)) {
            String producer =
                    layerIndex == 0
                            ? activationGraphName()
                            : layerIndex == firstLayer ? stageInput : null;
            if (producer == null) {
                producer = previousGraphName(layerIndex);
            }
            if (producer == null) {
                unifiedLayer.consumeFromDevice(gemma4State.workspace.wrapX);
            } else {
                unifiedLayer.consumeFromDevice(producer, gemma4State.workspace.wrapX);
            }
        }
        bindLayerWeights(unifiedLayer, layerIndex);
        bindRopeTables(unifiedLayer, layerIndex);
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        if (layerIndex == 0) {
            appendPLESetupTasks(unifiedLayer);
        }

        // ═══════════════════════════════════ ATTENTION ═══════════════════════════════════
        addRmsNorm(
                unifiedLayer,
                layerIndex,
                "attn_norm",
                gemma4State.workspace.temp,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapXb,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                false);

        boolean packed = packedFor(layerIndex);
        if (packed) {
            unifiedLayer.task(
                    tn(layerIndex, "attn_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums);
        }
        addProjection(
                unifiedLayer,
                tn(layerIndex, "q_proj"),
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapQ,
                weights.wqLayered[layerIndex],
                dim,
                qDim,
                packed);
        unifiedLayer.task(
                tn(layerIndex, "q_norm"),
                Gemma4Kernels::rmsNormPerHead,
                context,
                gemma4State.workspace.wrapQ,
                weights.attnQNorm[layerIndex].asFloatArray(),
                nHead,
                headDim,
                HEAD_NORM_LOCAL_SIZE,
                config.rmsNormEps());

        if (hasOwnKv) {
            addProjection(
                    unifiedLayer,
                    tn(layerIndex, "k_proj"),
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapK,
                    weights.wkLayered[layerIndex],
                    dim,
                    kvDim,
                    packed);
            boolean valuesAreKeys = weights.wvLayered[layerIndex] == null;
            if (valuesAreKeys) {
                // No value projection (the 31B's global layers): the values are the keys as
                // projected, before the key norm and RoPE, as llama.cpp takes them.
                unifiedLayer.task(
                        tn(layerIndex, "v_from_k"),
                        Gemma4Kernels::copyValues,
                        context,
                        gemma4State.workspace.wrapK,
                        gemma4State.workspace.wrapV,
                        kvDim);
            }
            unifiedLayer.task(
                    tn(layerIndex, "k_norm"),
                    Gemma4Kernels::rmsNormPerHead,
                    context,
                    gemma4State.workspace.wrapK,
                    weights.attnKNorm[layerIndex].asFloatArray(),
                    nHeadKv,
                    headDim,
                    HEAD_NORM_LOCAL_SIZE,
                    config.rmsNormEps());
            if (!valuesAreKeys) {
                addProjection(
                        unifiedLayer,
                        tn(layerIndex, "v_proj"),
                        gemma4State.workspace.wrapXb,
                        gemma4State.workspace.wrapV,
                        weights.wvLayered[layerIndex],
                        dim,
                        kvDim,
                        packed);
            }
            unifiedLayer.task(
                    tn(layerIndex, "v_norm"),
                    Gemma4Kernels::rmsNormPerHeadNoWeight,
                    context,
                    gemma4State.workspace.wrapV,
                    nHeadKv,
                    headDim,
                    HEAD_NORM_LOCAL_SIZE,
                    config.rmsNormEps());
            if (fp16KeyValue) {
                unifiedLayer.task(
                        tn(layerIndex, "rope_and_cache"),
                        Gemma4AttentionKernels::ropeNeoxRotateAndCacheCopyFP16,
                        context,
                        gemma4State.workspace.positionHolder,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapK,
                        gemma4State.workspace.wrapV,
                        gemma4State.workspace.wrapKeyCacheFP16,
                        gemma4State.workspace.wrapValueCacheFP16,
                        freqCisReal,
                        freqCisImag,
                        nHeadKv,
                        headDim,
                        kvDim,
                        cacheBaseOffset);
            } else {
                unifiedLayer.task(
                        tn(layerIndex, "rope_and_cache"),
                        Gemma4Kernels::ropeNeoxRotateAndCacheCopy,
                        context,
                        gemma4State.workspace.positionHolder,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapK,
                        gemma4State.workspace.wrapV,
                        gemma4State.workspace.wrapKeyCache,
                        gemma4State.workspace.wrapValueCache,
                        freqCisReal,
                        freqCisImag,
                        nHeadKv,
                        headDim,
                        kvDim,
                        cacheBaseOffset);
            }
        } else {
            unifiedLayer.task(
                    tn(layerIndex, "rope_q_only"),
                    Gemma4Kernels::ropeNeoxRotateQOnly,
                    context,
                    gemma4State.workspace.positionHolder,
                    gemma4State.workspace.wrapQ,
                    freqCisReal,
                    freqCisImag,
                    headDim);
        }

        int splits = attentionSplits();
        if (fp16KeyValue && !groupDecode(layerIndex)) {
            unifiedLayer.task(
                    tn(layerIndex, "attention_split"),
                    Gemma4Kernels::attentionWithSlidingWindowSplitFP16,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCacheFP16,
                    gemma4State.workspace.wrapValueCacheFP16,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapAttSplit,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    splits,
                    ATTENTION_LOCAL_SIZE);
            unifiedLayer.task(
                    tn(layerIndex, "attention_combine"),
                    Gemma4Kernels::combineSplitKVAttentionPerElement,
                    context,
                    gemma4State.workspace.wrapAttSplit,
                    gemma4State.workspace.wrapXb,
                    nHead,
                    headDim,
                    splits);
        } else if (fp16KeyValue) {
            int slices = decodeSlices(isSwa);
            // The shuffle kernel where the backend lowers simdShuffleDown (CUDA, and Metal's
            // 32-wide SIMD-group shuffles); elsewhere its shared-memory twin, on the same
            // workgroup, grid and partial layout.
            if (SchedulerDetectionService.isShuffleReducedFp16GemvSupported()
                    || SchedulerDetectionService.isSubgroupShuffle32Supported()) {
                unifiedLayer.task(
                        tn(layerIndex, "attention_group"),
                        Gemma4AttentionKernels::attentionDecodeGroupFP16,
                        context,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapKeyCacheFP16,
                        gemma4State.workspace.wrapValueCacheFP16,
                        gemma4State.workspace.wrapAttSplit,
                        gemma4State.workspace.positionHolder,
                        headDim,
                        kvDim,
                        cacheBaseOffset,
                        windowSize,
                        slices);
            } else {
                unifiedLayer.task(
                        tn(layerIndex, "attention_group"),
                        Gemma4AttentionKernels::attentionDecodeGroupFP16Shared,
                        context,
                        gemma4State.workspace.wrapQ,
                        gemma4State.workspace.wrapKeyCacheFP16,
                        gemma4State.workspace.wrapValueCacheFP16,
                        gemma4State.workspace.wrapAttSplit,
                        gemma4State.workspace.positionHolder,
                        headDim,
                        kvDim,
                        cacheBaseOffset,
                        windowSize,
                        slices);
            }
            unifiedLayer.task(
                    tn(layerIndex, "attention_combine"),
                    Gemma4AttentionKernels::combineDecodeGroup,
                    context,
                    gemma4State.workspace.wrapAttSplit,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.positionHolder,
                    nHead,
                    headDim,
                    windowSize,
                    slices);
        } else if (splits > 1) {
            unifiedLayer.task(
                    tn(layerIndex, "attention_split"),
                    Gemma4Kernels::attentionWithSlidingWindowSplit,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCache,
                    gemma4State.workspace.wrapValueCache,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapAttSplit,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    splits,
                    ATTENTION_LOCAL_SIZE);
            unifiedLayer.task(
                    tn(layerIndex, "attention_combine"),
                    Gemma4Kernels::combineSplitKVAttentionPerElement,
                    context,
                    gemma4State.workspace.wrapAttSplit,
                    gemma4State.workspace.wrapXb,
                    nHead,
                    headDim,
                    splits);
        } else {
            unifiedLayer.task(
                    tn(layerIndex, "attention"),
                    Gemma4Kernels::attentionWithSlidingWindowParallel,
                    context,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapKeyCache,
                    gemma4State.workspace.wrapValueCache,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapAtt,
                    nHead,
                    headDim,
                    kvDim,
                    kvMul,
                    gemma4State.workspace.positionHolder,
                    cacheBaseOffset,
                    windowSize,
                    config.contextLength(),
                    ATTENTION_LOCAL_SIZE);
        }

        if (packed) {
            // The attention output, not the normalized activation the projections above read: the
            // triple holds one activation at a time and attention has overwritten wrapXb since.
            unifiedLayer.task(
                    tn(layerIndex, "attn_out_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums);
        }
        addProjection(
                unifiedLayer,
                tn(layerIndex, "wo_proj"),
                gemma4State.workspace.wrapXb,
                gemma4State.workspace.wrapXb2,
                weights.woLayered[layerIndex],
                qDim,
                dim,
                packed);

        addRmsNorm(
                unifiedLayer,
                layerIndex,
                "post_attn",
                gemma4State.workspace.tempPostAttn,
                gemma4State.workspace.wrapXb2,
                gemma4State.workspace.wrapX,
                weights.attnPostNorm[layerIndex].asFloatArray(),
                true);

        // ═══════════════════════════════════════ FFN ═════════════════════════════════════
        addRmsNorm(
                unifiedLayer,
                layerIndex,
                "ffn_norm",
                gemma4State.workspace.tempFFN,
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapXb,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                false);

        // The feed-forward's activation is a different vector from the attention branch's, so it
        // needs its own quantization: the triple holds one activation at a time.
        if (packed) {
            unifiedLayer.task(
                    tn(layerIndex, "ffn_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums);
        }
        // Gate and up share one pass over one local array and one tree reduction, so the pair has
        // to be dispatched together rather than tensor by tensor where a fused kernel reads their
        // one representation; a layer that holds anything else projects them apart (splitGateUp).
        if (prepackedQ4(weights.w1Layered[layerIndex])
                || prepackedQ4(weights.w3Layered[layerIndex])) {
            if (!packed
                    || !prepackedQ4(weights.w1Layered[layerIndex])
                    || !prepackedQ4(weights.w3Layered[layerIndex])) {
                throw WeightPrepacking.noPrepackedKernel(
                        "gemma4 layer " + layerIndex + " fused gate/up");
            }
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    TransformerComputeKernelsQ4_0Packed::fusedFFNGateUpGeGLUQ4_0Packed,
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    ffnLen);
        } else if (splitGateUp(layerIndex)) {
            // Gate into wrapHb and up into wrapHb2, each by its own type, then GeGLU in place.
            addProjection(
                    unifiedLayer,
                    tn(layerIndex, "ffn_gate"),
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex],
                    dim,
                    ffnLen,
                    packed);
            addProjection(
                    unifiedLayer,
                    tn(layerIndex, "ffn_up"),
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapHb2,
                    weights.w3Layered[layerIndex],
                    dim,
                    ffnLen,
                    packed);
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    Gemma4Kernels::pleGateGeluMul,
                    context,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapHb2,
                    0,
                    ffnLen);
        } else if (packed && weights.w1Layered[layerIndex].dataType() == DataType.Q8_0) {
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpGeGLUQ8_0DP4A,
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    ffnLen,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else if (packed && weights.w1Layered[layerIndex].dataType() == DataType.Q4_0) {
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpGeGLUQ4_0DP4A,
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    ffnLen,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else if (weights.w1Layered[layerIndex].dataType() == DataType.Q4_0) {
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpGeGLUQ4_0,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    ffnLen,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            unifiedLayer.task(
                    tn(layerIndex, "ffn_gate_up"),
                    Gemma4Kernels::fusedGateUpGeGLUQ8,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    ffnLen,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }
        if (packed) {
            // The feed-forward's hidden vector, which is wider than anything quantized above --
            // up to 12288 here against the embedding's 1536.
            unifiedLayer.task(
                    tn(layerIndex, "ffn_hidden_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums);
        }
        addProjection(
                unifiedLayer,
                tn(layerIndex, "ffn_down_proj"),
                gemma4State.workspace.wrapHb,
                gemma4State.workspace.wrapXb2,
                weights.w2Layered[layerIndex],
                ffnLen,
                dim,
                packed);

        addRmsNorm(
                unifiedLayer,
                layerIndex,
                "post_ffn",
                gemma4State.workspace.tempPostFfn,
                gemma4State.workspace.wrapXb2,
                gemma4State.workspace.wrapX,
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                true);

        // ═══════════════════════════ PER-LAYER EMBEDDING (PLE) ═══════════════════════════
        if (config.hasPerLayerEmbeddings()) {
            addProjection(
                    unifiedLayer,
                    tn(layerIndex, "ple_gate_proj"),
                    gemma4State.workspace.wrapX,
                    gemma4State.workspace.wrapPerLayerGate,
                    weights.perLayerInpGate[layerIndex],
                    dim,
                    nEmbdPerLayer);
            unifiedLayer.task(
                    tn(layerIndex, "ple_gate_gelu_mul"),
                    Gemma4Kernels::pleGateGeluMul,
                    context,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerInputs,
                    peOffset,
                    nEmbdPerLayer);
            addProjection(
                    unifiedLayer,
                    tn(layerIndex, "ple_proj"),
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut,
                    weights.perLayerProj[layerIndex],
                    nEmbdPerLayer,
                    dim);

            addRmsNorm(
                    unifiedLayer,
                    layerIndex,
                    "ple_post",
                    gemma4State.workspace.tempPostPle,
                    gemma4State.workspace.wrapPerLayerOut,
                    gemma4State.workspace.wrapX,
                    weights.perLayerPostNorm[layerIndex].asFloatArray(),
                    true);
        }

        if (weights.layerOutputScale[layerIndex] != null) {
            unifiedLayer.task(
                    tn(layerIndex, "layer_output_scale"),
                    Gemma4Kernels::scaleInPlaceFromTensor,
                    context,
                    gemma4State.workspace.wrapX,
                    weights.layerOutputScale[layerIndex].asFloatArray(),
                    dim);
        }

        if (lastLayerOfGraph(layerIndex)) {
            unifiedLayer.persistOnDevice(gemma4State.workspace.wrapX);
        }
    }

    /**
     * One-time-per-token setup tasks, prepended to layer 0's graph: scales the token embedding by
     * {@code sqrt(dim)} (Gemma4 scales embeddings on input -- the generic {@link Activation} task
     * graph that produced {@code wrapX} doesn't know about this), then computes the per-layer
     * embedding inputs from the per-layer model projection and the (host-gathered) per-token
     * per-layer-token-embedding row. Mirrors steps 1-2 of {@link InferenceCore#forwardJavaGemma4}.
     */
    private void appendPLESetupTasks(TaskGraph unifiedLayer) {
        unifiedLayer.task(
                "scale_embedding",
                TransformerComputeKernels::scaleInPlace,
                context,
                gemma4State.workspace.wrapX,
                embedScale,
                dim);
        if (!config.hasPerLayerEmbeddings()) {
            return;
        }

        addProjection(
                unifiedLayer,
                "ple_model_proj",
                gemma4State.workspace.wrapX,
                gemma4State.workspace.wrapPerLayerProjScratch,
                weights.perLayerModelProj,
                dim,
                perLayerTotal);
        unifiedLayer.task(
                "ple_proj_scale_norm",
                Gemma4Kernels::pleProjScaleAndNormalize,
                context,
                gemma4State.workspace.wrapPerLayerProjScratch,
                weights.perLayerProjNorm.asFloatArray(),
                nEmbdPerLayer,
                HEAD_NORM_LOCAL_SIZE,
                perLayerProjScale,
                config.rmsNormEps());
        unifiedLayer.task(
                "ple_merge",
                Gemma4Kernels::addAndScale,
                context,
                gemma4State.workspace.wrapPerLayerInputs,
                gemma4State.workspace.wrapPerLayerProjScratch,
                gemma4State.workspace.wrapPerLayerTokenEmbedRow,
                perLayerInputScale,
                perLayerTotal);
    }

    // @formatter:off
    /**
     * This layer's weights: uploaded here in the single-token plan, and in the batched plan bound
     * from the batch-prefill graph of the same layer, which uploaded them already.
     *
     * <p>A weight bound with {@code transferToDevice} in two graphs of one execution plan gets a
     * device buffer in each, so the batched plan used to hold every projection twice. Only what
     * that prefill graph hands to one of its own tasks can be bound from it — a weight it declares
     * and no task reads is never allocated, and a consumer of it reads zeros — so the two
     * per-layer-embedding projections, which the prefill reads as FP16 copies, stay uploaded here.
     */
    // @formatter:on
    private void bindLayerWeights(TaskGraph unifiedLayer, int layerIndex) {
        java.util.List<Object> shared = new java.util.ArrayList<>();
        java.util.List<Object> own = new java.util.ArrayList<>();
        shared.add(weights.rms_att_weightLayered[layerIndex].asFloatArray());
        shared.add(weights.attnQNorm[layerIndex].asFloatArray());
        shared.add(weights.attnPostNorm[layerIndex].asFloatArray());
        shared.add(weights.rms_ffn_weightLayered[layerIndex].asFloatArray());
        shared.add(weights.ffnPostNorm[layerIndex].asFloatArray());
        if (config.hasPerLayerEmbeddings()) {
            shared.add(weights.perLayerPostNorm[layerIndex].asFloatArray());
        }
        if (config.hasOwnKv(layerIndex)) {
            shared.add(weights.attnKNorm[layerIndex].asFloatArray());
        }
        if (weights.layerOutputScale[layerIndex] != null) {
            shared.add(weights.layerOutputScale[layerIndex].asFloatArray());
        }
        // The projections: the batch-prefill graph reads them itself, whichever JIT GEMMs it
        // runs. On MLX it reads its own form of them instead, so they are uploaded here.
        java.util.List<Object> projections = batchedPlan && mlxPrefill() ? own : shared;
        projections.add(weights.wqLayered[layerIndex]);
        if (config.hasOwnKv(layerIndex)) {
            projections.add(weights.wkLayered[layerIndex]);
            if (weights.wvLayered[layerIndex] != null) {
                projections.add(weights.wvLayered[layerIndex]);
            }
        }
        projections.add(weights.woLayered[layerIndex]);
        projections.add(weights.w1Layered[layerIndex]);
        projections.add(weights.w3Layered[layerIndex]);
        projections.add(weights.w2Layered[layerIndex]);
        if (batchedPlan) {
            unifiedLayer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex),
                    WeightPrepacker.deviceArrays(shared.toArray()));
        } else {
            own.addAll(shared);
        }
        if (config.hasPerLayerEmbeddings()) {
            own.add(weights.perLayerInpGate[layerIndex]);
            own.add(weights.perLayerProj[layerIndex]);
        }
        gemma4State.workspace.prepacker.configureGraphInputs(
                unifiedLayer, state.workspace.prepacking, own.toArray());
    }

    /** Whether the batched prefill multiplies the projections on MLX, in its own form of them. */
    private boolean mlxPrefill() {
        return Gemma4BatchPrefillLayers.mlxPrefill(
                        gemma4State.executionPolicy(), weights.dataType(), config)
                .isPresent();
    }

    /** Which RoPE pairs the graph being built has bound already: sliding, full. */
    private final boolean[] ropeBound = new boolean[2];

    // @formatter:off
    /**
     * This layer's RoPE pair, bound once per graph and uploaded once per plan.
     *
     * <p>The tables are the same arrays for every layer of a kind, and a graph that binds an array
     * without declaring where it comes from gets its own device copy. The first layer of a kind
     * uploads its pair — or, in the batched plan, the first batch-prefill layer of that kind
     * already did — and every other graph binds it from there. From that graph and not another: a
     * graph that declares an array none of its tasks reads never allocates it.
     */
    // @formatter:on
    private void bindRopeTables(TaskGraph unifiedLayer, int layerIndex) {
        boolean isSwa = config.isSwa(layerIndex);
        int kind = isSwa ? 0 : 1;
        if (ropeBound[kind]) {
            return;
        }
        ropeBound[kind] = true;
        Object[] pair =
                isSwa
                        ? new Object[] {
                            weights.freqCisRealSwa.asFloatArray(),
                            weights.freqCisImagSwa.asFloatArray()
                        }
                        : new Object[] {
                            weights.freqCisRealFull.asFloatArray(),
                            weights.freqCisImagFull.asFloatArray()
                        };
        int firstUser = firstLayerOfKind(isSwa);
        if (batchedPlan) {
            unifiedLayer.consumeFromDevice(GraphNames.batchPrefillLayer(firstUser), pair);
        } else if (layerGraphName(firstUser).equals(layerGraphName(layerIndex))) {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, pair);
        } else {
            unifiedLayer.consumeFromDevice(layerGraphName(firstUser), pair);
        }
    }

    /** The first layer that attends with a sliding window, or the first that attends fully. */
    private int firstLayerOfKind(boolean isSwa) {
        for (int l = firstLayer; l < endLayer(config.numberOfLayers()); l++) {
            if (config.isSwa(l) == isSwa) {
                return l;
            }
        }
        throw new IllegalStateException("no layer attends with isSwa=" + isSwa);
    }

    // @formatter:off
    /**
     * One RMSNorm of {@code input}: into {@code target} ({@code residual == false}), or added into
     * it ({@code residual == true}).
     *
     * <p>The reduction into {@code temp}, the finalizing step where the scheduler needs one, then
     * the apply.
     */
    // @formatter:on
    private void addRmsNorm(
            TaskGraph graph,
            int layerIndex,
            String name,
            FloatArray temp,
            FloatArray input,
            FloatArray target,
            FloatArray weight,
            boolean residual) {
        graph.task(
                tn(layerIndex, name + "_reduce"),
                rmsReduceKernel(),
                context,
                temp,
                input,
                dim,
                config.rmsNormEps(),
                gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            graph.task(
                    tn(layerIndex, name + "_finalize"),
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    temp,
                    dim,
                    config.rmsNormEps());
        }
        if (residual) {
            graph.task(
                    tn(layerIndex, name + "_apply"),
                    Gemma4Kernels::rmsNormApplyWithResidual,
                    context,
                    target,
                    input,
                    weight,
                    temp,
                    dim);
        } else {
            graph.task(
                    tn(layerIndex, name + "_apply"),
                    Gemma4Kernels::applyRmsNorm,
                    context,
                    target,
                    input,
                    weight,
                    temp,
                    dim);
        }
    }

    /** Configure data transfers for first and subsequent layers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // Per graph, not per layer. A graph that both transfers a buffer in and consumes it from
        // the previous graph is declaring two contradictory things about the same edge, which is
        // what grouping made possible: layers 1..3 of a group used to be separate graphs whose
        // consume was the real edge, and inside one graph they are simply the layers above them.
        if (!firstLayerOfGraph(layerIndex)) {
            return unifiedLayer;
        }
        if (layerIndex == firstLayer) {
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    gemma4State.workspace.positionHolder,
                    gemma4State.workspace.wrapPerLayerTokenEmbedRow,
                    gemma4State.workspace.temp,
                    gemma4State.workspace.tempFFN,
                    gemma4State.workspace.tempPostAttn,
                    gemma4State.workspace.tempPostFfn,
                    gemma4State.workspace.tempPostPle);
            if (config.hasPerLayerEmbeddings()) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, weightArray(weights.perLayerModelProj));
                if (batchedPlan) {
                    // Read by the first batch-prefill graph's own setup, which uploaded it.
                    unifiedLayer.consumeFromDevice(
                            "batchPrefillLayer_0", weights.perLayerProjNorm.asFloatArray());
                } else {
                    unifiedLayer.transferToDevice(
                            DataTransferMode.FIRST_EXECUTION,
                            weights.perLayerProjNorm.asFloatArray());
                }
            }
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXb2,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapK,
                    gemma4State.workspace.wrapV,
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapPerLayerInputs,
                    gemma4State.workspace.wrapPerLayerProjScratch,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut);
            if (anySplitGateUp()) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, gemma4State.workspace.wrapHb2);
            }
            bindKeyValueCache(unifiedLayer);
        } else {
            unifiedLayer.consumeFromDevice(
                    previousGraphName(layerIndex),
                    context,
                    gemma4State.workspace.wrapXb,
                    gemma4State.workspace.wrapXb2,
                    gemma4State.workspace.wrapQ,
                    gemma4State.workspace.wrapK,
                    gemma4State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    gemma4State.workspace.wrapAtt,
                    gemma4State.workspace.wrapHb,
                    gemma4State.workspace.wrapPerLayerInputs,
                    gemma4State.workspace.wrapPerLayerGate,
                    gemma4State.workspace.wrapPerLayerOut,
                    gemma4State.workspace.positionHolder);
            if (anySplitGateUp()) {
                unifiedLayer.consumeFromDevice(
                        previousGraphName(layerIndex), gemma4State.workspace.wrapHb2);
            }
        }
        return unifiedLayer;
    }

    // @formatter:off
    /**
     * The graph layer 0 takes its activation from: always named.
     *
     * <p>In the batched plan the graph list holds the batch-prefill layers between the activation
     * and layer 0, and layer 0 that does not name its producer imports a buffer nobody wrote —
     * measured as an activation of exactly zero out of every decode layer, with the model still
     * emitting fluent tokens off the resulting logits. In the single-token plan the activation
     * graph runs just before, but layer 0 also consumes its prepacked weights by name, and a graph
     * that consumes anything by name takes nothing through an unnamed consume.
     */
    // @formatter:on
    private String activationGraphName() {
        return batchedPlan ? GraphNames.DECODE_ACTIVATION : GraphNames.ACTIVATION_UPDATE;
    }

    // @formatter:off
    /**
     * Where layer 0 gets the key and value caches from.
     *
     * <p>Allocating them here is right for every plan that prefills one token at a time, because
     * nothing ran before this graph. It is wrong for the batched plan: there the batch-prefill
     * graphs already filled a cache, and a decode that allocated its own would attend an empty one
     * and answer as though the prompt had never been read — fluent output, silently wrong, and not
     * something a throughput number would show.
     */
    // @formatter:on
    private void bindKeyValueCache(TaskGraph unifiedLayer) {
        if (batchedPlan) {
            // A later stage's decode-activation graph only receives the activation; the cache
            // its prefill filled is in that prefill's last graph on the same device.
            String source =
                    firstLayer == 0
                            ? GraphNames.DECODE_ACTIVATION
                            : GraphNames.batchPrefillLayer(endLayer(config.numberOfLayers()) - 1);
            unifiedLayer.consumeFromDevice(source, keyCache(), valueCache());
            return;
        }
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, keyCache(), valueCache());
    }

    /** The key cache in the representation the state allocated. */
    private Object keyCache() {
        return fp16KeyValue
                ? gemma4State.workspace.wrapKeyCacheFP16
                : gemma4State.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state allocated. */
    private Object valueCache() {
        return fp16KeyValue
                ? gemma4State.workspace.wrapValueCacheFP16
                : gemma4State.workspace.wrapValueCache;
    }

    /**
     * Slices the grouped decode attention is launched with for a layer: the whole window for a
     * sliding layer, the whole context for a full one. Idle slices return at once.
     */
    private int decodeSlices(boolean isSwa) {
        int span =
                isSwa
                        ? Math.min(config.slidingWindowSize(), config.contextLength())
                        : config.contextLength();
        return Gemma4AttentionKernels.decodeSlices(span);
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                          MIXED-PRECISION PROJECTION HELPERS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * Adds a matrix-vector projection task, selecting the matmul kernel from the weight tensor's
     * execution type. The per-layer-embedding projections are not uniformly Q8_0 in a Q8_0 GGUF
     * (they may be F32 or F16), so the kernel/accessor must be chosen per tensor.
     *
     * <p>Dispatches on {@link TornadoTensor#dataType()} rather than the file's {@code GGMLType}: a
     * backend kernel is chosen by what it computes with, and naming a format type here would put
     * GGUF's vocabulary in the backend (Rule 4).
     */
    private void addProjection(
            TaskGraph tg,
            String taskName,
            FloatArray in,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        addProjection(tg, taskName, in, out, w, n, d, false);
    }

    /**
     * @param packedActivation whether {@code in} has already been quantized into the {@code
     *     wrapXbQuants/Scales/Sums} triple by a task in this graph. It is carried explicitly rather
     *     than inferred from the buffer, because one buffer holds several different activations
     *     over a layer and its identity says nothing about which one is in it.
     */
    private void addProjection(
            TaskGraph tg,
            String taskName,
            FloatArray in,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean packedActivation) {
        if (prepackedQ4(w)) {
            if (packedActivation) {
                tg.task(
                        taskName,
                        org.beehive.jitllm.backend.tornado.kernels
                                        .TransformerComputeKernelsQ4_0Packed
                                ::matrixVectorQ4_0Packed,
                        context,
                        gemma4State.workspace.wrapXbQuants,
                        gemma4State.workspace.wrapXbScales,
                        out,
                        w.asByteArray(),
                        n,
                        d,
                        0);
            } else {
                tg.task(
                        taskName,
                        org.beehive.jitllm.backend.tornado.kernels
                                        .TransformerComputeKernelsQ4_0Packed
                                ::matrixVectorQ4_0PackedF32,
                        context,
                        in,
                        out,
                        w.asByteArray(),
                        n,
                        d,
                        0);
            }
            return;
        }
        if (packedActivation && w.dataType() == DataType.Q8_0) {
            tg.task(
                    taskName,
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    projectionLocalSize(d));
            return;
        }
        if (packedActivation
                && (w.dataType() == DataType.Q4_1
                        || w.dataType() == DataType.Q4_K
                        || w.dataType() == DataType.Q5_K
                        || w.dataType() == DataType.Q6_K)) {
            tg.task(
                    taskName,
                    switch (w.dataType()) {
                        case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KDP4A;
                        case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_KDP4A;
                        case Q6_K -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A;
                        default -> TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1DP4A;
                    },
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    projectionLocalSize(d));
            return;
        }
        if (packedActivation && w.dataType() == DataType.Q4_0) {
            tg.task(
                    taskName,
                    TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                    context,
                    gemma4State.workspace.wrapXbQuants,
                    gemma4State.workspace.wrapXbScales,
                    gemma4State.workspace.wrapXbSums,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    projectionLocalSize(d));
            return;
        }
        if (warpProjection(w)) {
            tg.task(
                    taskName,
                    Gemma4Kernels::matrixVectorQ8_0Warp,
                    context,
                    in,
                    out,
                    w.asByteArray(),
                    n,
                    d);
            return;
        }
        switch (w.dataType()) {
            case Q8_0 ->
                    tg.task(
                            taskName,
                            TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                            context,
                            in,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            projectionLocalSize(d));
            case F16 ->
                    tg.task(
                            taskName,
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            in,
                            out,
                            w.asHalfFloatArray(),
                            n,
                            d,
                            projectionLocalSize(d));
            case Q4_0 ->
                    tg.task(
                            taskName,
                            TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                            context,
                            in,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            projectionLocalSize(d));
                // ffn_down is Q4_1 on this file's first blocks and Q4_0 on the rest, so the kernel
                // comes from the tensor rather than from the model's representation.
            case Q4_1 ->
                    tg.task(
                            taskName,
                            TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1,
                            context,
                            in,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            projectionLocalSize(d));
            case Q4_K, Q5_K, Q6_K ->
                    tg.task(
                            taskName,
                            switch (w.dataType()) {
                                case Q4_K -> TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K;
                                case Q5_K -> TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K;
                                default -> TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K;
                            },
                            context,
                            in,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            projectionLocalSize(d));
            case F32 ->
                    tg.task(
                            taskName,
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            in,
                            out,
                            w.asFloatArray(),
                            n,
                            d,
                            projectionLocalSize(d));
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported projection weight type: " + w.dataType());
        }
    }

    /** Whether the plan prepacks {@code w} as Q4_0. */
    private boolean prepackedQ4(TornadoTensor w) {
        return state.workspace.prepacking.prepackedQ4(w);
    }

    // @formatter:off
    /**
     * Whether a projection takes the warp-per-row Q8_0 kernel: Q8_0, on the NVIDIA path, on a
     * backend whose {@code simdShuffleDown} is correct.
     *
     * <p>The scheduler type alone is not enough: an NVIDIA device reached through OpenCL is on the
     * NVIDIA path too, and TornadoVM's OpenCL backend refuses the shuffle at compile time ("Unable
     * to build sketch for method: matrixVectorQ8_0Warp"), which left the first generation failing
     * and the session unusable. There the generic shared-memory Q8_0 kernel runs, as it does on
     * every non-NVIDIA device. CUDA holds the capability, so its selection does not change.
     */
    // @formatter:on
    private boolean warpProjection(TornadoTensor w) {
        // A Q8_0 projection takes the packed-integer kernel instead wherever that one runs.
        return w.dataType() == DataType.Q8_0
                && !packedIntegerDot()
                && !shouldUseFinalNormalization()
                && SchedulerDetectionService.isShuffleReducedFp16GemvSupported();
    }

    /** The worker grid of a projection of {@code rows} outputs, matching its kernel. */
    private WorkerGrid projectionGrid(TornadoTensor w, int rows) {
        if (prepackedQ4(w)) {
            return WorkerGridFactory.genericWorker(
                    rows
                            / 16
                            * org.beehive.jitllm.backend.tornado.kernels
                                    .TransformerComputeKernelsQ4_0Packed.LOCAL,
                    TransformerComputeKernelsQ4_0Packed.LOCAL);
        }
        if (warpProjection(w)) {
            int groups =
                    (rows + Gemma4Kernels.WARP_ROWS_PER_GROUP - 1)
                            / Gemma4Kernels.WARP_ROWS_PER_GROUP;
            return WorkerGridFactory.genericWorker(groups * 256, 256);
        }
        return WorkerGridFactory.genericWorker(
                rows * projectionLocalSize(rows), projectionLocalSize(rows));
    }

    /**
     * Returns the device-resident native array backing a weight tensor (for {@code
     * transferToDevice}), matching {@link #addProjection}'s dispatch.
     */
    private static Object weightArray(TornadoTensor w) {
        return switch (w.dataType()) {
            case Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K -> w.asByteArray();
            case F16 -> w.asHalfFloatArray();
            case F32 -> w.asFloatArray();
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported projection weight type: " + w.dataType());
        };
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                 GRID SCHEDULER
    // ═══════════════════════════════════════════════════════════════════════════════════

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker =
                WorkerGridFactory.createRmsNormWorker(dim, gemma4State.localSize);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);
        WorkerGrid dimElementWiseWorker =
                WorkerGridFactory.genericWorker(dim, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid normApplyWorker = dimElementWiseWorker;
        WorkerGrid woProjWorker =
                WorkerGridFactory.genericWorker(
                        dim * projectionLocalSize(dim), projectionLocalSize(dim));
        // Never empty, so a model without per-layer embeddings still gets valid (unused) grids.
        int pleWidth = Math.max(32, nEmbdPerLayer);
        int pleTotal = Math.max(32, perLayerTotal);
        WorkerGrid pleGateProjWorker =
                WorkerGridFactory.genericWorker(
                        pleWidth * projectionLocalSize(pleWidth), projectionLocalSize(pleWidth));
        WorkerGrid pleGateGeluWorker =
                WorkerGridFactory.genericWorker(pleWidth, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // === Layer-0 PLE setup ===
        gridScheduler.addWorkerGrid("layer_0.scale_embedding", dimElementWiseWorker);
        gridScheduler.addWorkerGrid(
                "layer_0.ple_model_proj",
                WorkerGridFactory.genericWorker(
                        pleTotal * projectionLocalSize(pleTotal), projectionLocalSize(pleTotal)));
        gridScheduler.addWorkerGrid(
                "layer_0.ple_proj_scale_norm",
                WorkerGridFactory.genericWorker(
                        config.numberOfLayers() * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE));
        gridScheduler.addWorkerGrid(
                "layer_0.ple_merge",
                WorkerGridFactory.genericWorker(pleTotal, LOCAL_WORK_GROUP_SIZE_ALLOC));

        for (int i = firstLayer; i < endLayer(config.numberOfLayers()); i++) {
            String prefix = layerGraphName(i) + "." + layerTaskPrefix(i);
            int nHeadKv = config.keyValueHeads(i);
            int headDim = config.headDim(i);
            boolean hasOwnKv = config.hasOwnKv(i);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(i);

            WorkerGrid headNormWorker =
                    WorkerGridFactory.genericWorker(
                            nHead * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid kvHeadNormWorker =
                    WorkerGridFactory.genericWorker(
                            nHeadKv * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(nHead, headDim);
            WorkerGrid attentionWorker = WorkerGridFactory.createAttentionWorker(nHead, headDim);
            WorkerGrid ffnGateUpWorker =
                    prepackedQ4(weights.w1Layered[i])
                            ? WorkerGridFactory.genericWorker(
                                    ffnLen
                                            / 16
                                            * org.beehive.jitllm.backend.tornado.kernels
                                                    .TransformerComputeKernelsQ4_0Packed
                                                    .FUSED_LOCAL,
                                    org.beehive.jitllm.backend.tornado.kernels
                                            .TransformerComputeKernelsQ4_0Packed.FUSED_LOCAL)
                            : WorkerGridFactory.genericWorker(
                                    ffnLen * LOCAL_WORK_GROUP_SIZE_ALLOC,
                                    LOCAL_WORK_GROUP_SIZE_ALLOC);

            gridScheduler.addWorkerGrid(prefix + "attn_norm_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "attn_norm_apply", normApplyWorker);
            if (packedFor(i)) {
                WorkerGrid quantizeWorker = WorkerGridFactory.genericWorker(dim, 32);
                gridScheduler.addWorkerGrid(prefix + "attn_quantize", quantizeWorker);
                gridScheduler.addWorkerGrid(prefix + "ffn_quantize", quantizeWorker);
                gridScheduler.addWorkerGrid(
                        prefix + "attn_out_quantize", WorkerGridFactory.genericWorker(qDim, 32));
                gridScheduler.addWorkerGrid(
                        prefix + "ffn_hidden_quantize",
                        WorkerGridFactory.genericWorker(ffnLen, 32));
            }
            gridScheduler.addWorkerGrid(
                    prefix + "q_proj", projectionGrid(weights.wqLayered[i], qDim));
            gridScheduler.addWorkerGrid(prefix + "q_norm", headNormWorker);
            if (hasOwnKv) {
                gridScheduler.addWorkerGrid(
                        prefix + "k_proj", projectionGrid(weights.wkLayered[i], kvDim));
                gridScheduler.addWorkerGrid(prefix + "k_norm", kvHeadNormWorker);
                if (weights.wvLayered[i] != null) {
                    gridScheduler.addWorkerGrid(
                            prefix + "v_proj", projectionGrid(weights.wvLayered[i], kvDim));
                } else {
                    gridScheduler.addWorkerGrid(
                            prefix + "v_from_k",
                            WorkerGridFactory.genericWorker(kvDim, LOCAL_WORK_GROUP_SIZE_ALLOC));
                }
                gridScheduler.addWorkerGrid(prefix + "v_norm", kvHeadNormWorker);
                gridScheduler.addWorkerGrid(prefix + "rope_and_cache", ropeWorker);
            } else {
                gridScheduler.addWorkerGrid(prefix + "rope_q_only", ropeWorker);
            }
            if (groupDecode(i)) {
                gridScheduler.addWorkerGrid(
                        prefix + "attention_group",
                        WorkerGridFactory.genericWorker(
                                nHeadKv
                                        * decodeSlices(config.isSwa(i))
                                        * Gemma4AttentionKernels.DECODE_LANES,
                                Gemma4AttentionKernels.DECODE_LANES));
                gridScheduler.addWorkerGrid(
                        prefix + "attention_combine",
                        WorkerGridFactory.genericWorker(nHead * headDim, 128));
            } else if (fp16KeyValue || attentionSplits() > 1) {
                gridScheduler.addWorkerGrid(
                        prefix + "attention_split",
                        WorkerGridFactory.genericWorker(
                                nHead * attentionSplits() * ATTENTION_LOCAL_SIZE,
                                ATTENTION_LOCAL_SIZE));
                gridScheduler.addWorkerGrid(
                        prefix + "attention_combine",
                        WorkerGridFactory.genericWorker(nHead * headDim, 128));
            } else {
                gridScheduler.addWorkerGrid(prefix + "attention", attentionWorker);
            }
            gridScheduler.addWorkerGrid(
                    prefix + "wo_proj", projectionGrid(weights.woLayered[i], dim));
            gridScheduler.addWorkerGrid(prefix + "post_attn_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "post_attn_apply", normApplyWorker);

            gridScheduler.addWorkerGrid(prefix + "ffn_norm_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_norm_apply", normApplyWorker);
            if (splitGateUp(i)) {
                gridScheduler.addWorkerGrid(
                        prefix + "ffn_gate", projectionGrid(weights.w1Layered[i], ffnLen));
                gridScheduler.addWorkerGrid(
                        prefix + "ffn_up", projectionGrid(weights.w3Layered[i], ffnLen));
                gridScheduler.addWorkerGrid(
                        prefix + "ffn_gate_up",
                        WorkerGridFactory.genericWorker((ffnLen + 255) / 256 * 256, 256));
            } else {
                gridScheduler.addWorkerGrid(prefix + "ffn_gate_up", ffnGateUpWorker);
            }
            gridScheduler.addWorkerGrid(
                    prefix + "ffn_down_proj", projectionGrid(weights.w2Layered[i], dim));
            gridScheduler.addWorkerGrid(prefix + "post_ffn_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "post_ffn_apply", normApplyWorker);

            gridScheduler.addWorkerGrid(prefix + "ple_gate_proj", pleGateProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_gate_gelu_mul", pleGateGeluWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_apply", normApplyWorker);

            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid(prefix + "attn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_attn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ffn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_ffn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ple_post_finalize", rmsNormWorker);
            }
            if (weights.layerOutputScale[i] != null) {
                gridScheduler.addWorkerGrid(prefix + "layer_output_scale", dimElementWiseWorker);
            }
        }
        return gridScheduler;
    }
}
