package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.BatchMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4AttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillSimdgroupAttentionFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillSimdgroupAttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TensorCoreAttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
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
    public String describeNativeLibraries() {
        return mlx != null ? "MLX" : nativeProjections ? "cuBLAS" : "none";
    }

    @Override
    public String describeProjections() {
        if (mlx != null) {
            return mlx.projections() == MlxPrefill.Projections.AFFINE
                    ? "MLX quantized GEMMs (Q8_0 weights repacked to MLX's affine format)"
                    : "MLX GEMMs (Q8_0 weights decoded once to resident FP16 copies)";
        }
        return simdgroup
                ? "Metal SIMD-group GEMMs (Q8_0 weights dequantized as they are staged)"
                : "FP16 tensor-core MMA (quantized weights dequantized where needed)";
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

    // @formatter:off
    /**
     * With native libraries ({@code --with-native-libraries}): each layer's projections decoded
     * once to FP16 row-major matrices for cuBLAS, several tensors stacked into one matrix (query,
     * key and value; gate and up) so one GEMM computes them side by side; otherwise {@code null}
     * and the JIT kernels run.
     */
    // @formatter:on
    private final boolean nativeProjections;

    // @formatter:off
    /**
     * Whether the projections run on the Metal SIMD-group GEMMs ({@code simdgroup_matrix}) instead
     * of the tensor-core ones; see {@link BatchPrefillGemmPolicy#simdgroup()}.
     */
    // @formatter:on
    private final boolean simdgroup = BatchPrefillGemmPolicy.simdgroup();

    // @formatter:off
    /**
     * With native libraries ({@code --with-native-libraries}) on Metal: the query/key/value,
     * output, gate/up and down projections as MLX GEMMs over each layer's weights in MLX's form
     * ({@link MlxPrefill#projections()}), query, key and value stacked into one matrix and gate and
     * up into another; otherwise {@code null}. Attention, the norms and the per-layer-embedding
     * GEMMs stay on the Metal kernels.
     */
    // @formatter:on
    private final MlxPrefill mlx;

    private MlxPrefillProjections.Matrix[] mlxQkv;
    private MlxPrefillProjections.Matrix[] mlxWo;
    private MlxPrefillProjections.Matrix[] mlxGateUp;
    private MlxPrefillProjections.Matrix[] mlxDown;

    // @formatter:off
    /**
     * The MLX GEMMs' FP16 operands and products, and the down projection's row factors.
     *
     * <p>MLX takes an array only at its exact extent, and this family's layers differ in head and
     * feed-forward width, so a buffer exists per width ({@code paddedBatch} rows of it) rather than
     * one at the widest: the packed {@code [q|k|v]} product, the attention output the output
     * projection reads, the gate/up product and the GeGLU the down projection reads. Each is
     * uploaded by the first layer graph that uses it and bound from that graph by the others.
     */
    // @formatter:on
    private final java.util.Map<Integer, HalfFloatArray> mlxQkvOut = new java.util.TreeMap<>();

    private final java.util.Map<Integer, HalfFloatArray> mlxAttnIn = new java.util.TreeMap<>();
    private final java.util.Map<Integer, HalfFloatArray> mlxGateUpOut = new java.util.TreeMap<>();
    private final java.util.Map<Integer, HalfFloatArray> mlxHbIn = new java.util.TreeMap<>();
    private HalfFloatArray mlxOutHalf; // paddedBatch × dim
    private FloatArray mlxDownRowScale; // paddedBatch

    /** For each MLX buffer, the first layer whose graph uses it, and so uploads it. */
    private final java.util.Map<Object, Integer> mlxFirstUser = new java.util.IdentityHashMap<>();

    /** Threads per row of the down projection's row-scale reduction. */
    private static final int ROW_THREADS = 256;

    /** The SIMD-group projections with their output counts, for their grids. */
    private final java.util.Map<String, Integer> simdgroupTasks = new java.util.LinkedHashMap<>();

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    /** Whether the model has per-layer input embeddings; the 31B has none. */
    private final boolean ple;

    /** The layers {@code [firstLayer, endLayer)} this stack builds: all of them, or one stage's. */
    private final int firstLayer;

    private final int endLayer;

    /** By layer: whether its query/key/value, output, gate/up and down GEMMs are int8. */
    private final boolean[] int8Qkv;

    private final boolean[] int8Wo;

    private final boolean[] int8GateUp;

    private final boolean[] int8Down;

    /** The int8 tasks' worker grids, by qualified task name, recorded as the graphs are built. */
    private final java.util.Map<String, WorkerGrid> int8Grids = new java.util.LinkedHashMap<>();

    /**
     * Whether the attention runs as FP16 tensor-core GEMMs over the cache gathered per key/value
     * head: scores, a causal sliding-window softmax and the weighted values, a pass of query heads
     * at a time. Needs the FP16 cache.
     */
    private boolean gemmAttention;

    private int paddedPositions;
    private int headsPerPass;
    private HalfFloatArray queriesF16;
    private HalfFloatArray keysF16;
    private HalfFloatArray valuesTF16;
    private FloatArray scores;
    private HalfFloatArray probs;
    private FloatArray attnF32;

    public Gemma4BatchPrefillLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            int batchSize) {
        this(state, weights, config, batchSize, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices. A
     * stage after the first takes its activation from the graph named {@code prefillActivation}.
     */
    public Gemma4BatchPrefillLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        if (firstLayer < 0 || endLayer > config.numberOfLayers() || firstLayer >= endLayer) {
            throw new IllegalArgumentException(
                    "layer range [" + firstLayer + ", " + endLayer + ") outside the model");
        }
        if (firstLayer > 0 && config.hasPerLayerEmbeddings()) {
            throw new UnsupportedOperationException(
                    "a gemma4 model with per-layer embeddings cannot be split across devices");
        }
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.ple = config.hasPerLayerEmbeddings();
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.fp16KeyValue = state.usesFp16KeyValueCache();
        this.paddedBatch = (batchSize + 127) & ~127;
        this.nHead = config.numberOfHeads();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));

        int layers = config.numberOfLayers();
        this.pleGateF16 = new HalfFloatArray[layers];
        this.pleProjF16 = new HalfFloatArray[layers];
        for (int l = firstLayer; ple && l < endLayer; l++) {
            int pleElements = dim * nEmbdPerLayer;
            pleGateF16[l] =
                    narrowToF16(weights.perLayerInpGate[l], pleElements, "blk." + l + ".inp_gate");
            pleProjF16[l] = narrowToF16(weights.perLayerProj[l], pleElements, "blk." + l + ".proj");
        }
        this.pleModelProjF16 =
                ple
                        ? narrowToF16(
                                weights.perLayerModelProj,
                                perLayerTotal * dim,
                                "per_layer_model_proj")
                        : null;

        this.nativeProjections = nativeProjections(state);
        this.mlx = mlxPrefill(state.executionPolicy(), weights.dataType(), config).orElse(null);
        if (mlx != null) {
            buildMlxProjections();
        }

        this.gemmAttention = fp16KeyValue && TensorCoreSupport.isTensorCoreCapableBackend();
        if (gemmAttention) {
            paddedPositions = (config.contextLength() + 127) / 128 * 128;
            headsPerPass = headsPerPass(nHead, paddedBatch, paddedPositions);
            int qWidth = nHead * config.maxHeadDim();
            queriesF16 = TornadoWorkspaces.halfFloats(paddedBatch * qWidth);
            keysF16 = TornadoWorkspaces.halfFloats(config.maxKeyValueDim() * paddedPositions);
            valuesTF16 = TornadoWorkspaces.halfFloats(config.maxKeyValueDim() * paddedPositions);
            scores = TornadoWorkspaces.floats(paddedBatch * headsPerPass * paddedPositions);
            probs = TornadoWorkspaces.halfFloats(paddedBatch * headsPerPass * paddedPositions);
            attnF32 = TornadoWorkspaces.floats(paddedBatch * qWidth);
        }
        this.int8Qkv = new boolean[layers];
        this.int8Wo = new boolean[layers];
        this.int8GateUp = new boolean[layers];
        this.int8Down = new boolean[layers];
        for (int l = firstLayer; l < endLayer; l++) {
            int8Qkv[l] =
                    config.hasOwnKv(l)
                            ? int8(weights.wqLayered[l], weights.wkLayered[l], valueWeights(l))
                            : int8(weights.wqLayered[l]);
            int8Wo[l] = int8(weights.woLayered[l]);
            int8GateUp[l] = int8(weights.w1Layered[l], weights.w3Layered[l]);
            int8Down[l] = int8(weights.w2Layered[l]);
        }

        List<ImmutableTaskGraph> graphs = new ArrayList<>(endLayer - firstLayer);
        for (int l = firstLayer; l < endLayer; l++) {
            graphs.add(createBatchPrefillLayerTaskGraph(l).snapshot());
        }
        this.layerITGs = List.copyOf(graphs);
    }

    /**
     * Whether these projections take the int8 GEMM: each Q8_0 or Q4_0, JIT kernels, tensor cores.
     */
    private boolean int8(TornadoTensor... tensors) {
        for (TornadoTensor t : tensors) {
            if (t.dataType() != DataType.Q8_0 && t.dataType() != DataType.Q4_0) {
                return false;
            }
        }
        return !nativeProjections && mlx == null && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /** The most query heads per attention pass whose scores fit in 64 MB, a divisor of all. */
    private static int headsPerPass(int heads, int rows, int positions) {
        long budget = 64L << 20;
        for (int h = heads; h >= 1; h--) {
            if (heads % h == 0 && (long) rows * h * positions * 4 <= budget) {
                return h;
            }
        }
        return 1;
    }

    // @formatter:off
    /**
     * The chunk's attention as FP16 tensor-core GEMMs: the layer's cache rows up to the chunk's
     * last position gathered per key/value head, the queries in FP16, then per pass of {@link
     * #headsPerPass} query heads the scores (each head against its key/value head), the causal
     * sliding-window softmax (this family's attention scale is one), and the weighted values; the
     * output in FP16 for the output projection.
     */
    // @formatter:on
    private void gemmAttention(
            TaskGraph layer,
            int hd,
            int kvMul,
            int qDim,
            int kvDim,
            int stride,
            int cacheBaseOffset,
            int window) {
        var ws = state.workspace;
        int npad = paddedPositions;
        String graph = layer.getTaskGraphName() + ".";
        layer.task(
                "attn_gather",
                TensorCoreAttentionKernels::gatherKeyValues,
                context,
                ws.batchStartPosHolder,
                ws.wrapKeyCacheFP16,
                ws.wrapValueCacheFP16,
                keysF16,
                valuesTF16,
                cacheBaseOffset,
                kvDim,
                hd,
                npad);
        int8Grids.put(graph + "attn_gather", elementwise(npad * kvDim, 256));
        layer.task(
                "attn_q16",
                TensorCoreAttentionKernels::queriesToHalf,
                context,
                ws.qkvResultBatch,
                queriesF16,
                qDim,
                stride,
                paddedBatch * qDim);
        int8Grids.put(graph + "attn_q16", elementwise(paddedBatch * qDim, 256));
        int hp = headsPerPass;
        for (int pass = 0; pass * hp < nHead; pass++) {
            int z0 = pass * hp;
            String sTask = "attn_scores_" + pass;
            layer.task(
                    sTask,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    queriesF16,
                    keysF16,
                    scores,
                    paddedBatch,
                    npad,
                    hd,
                    qDim,
                    hd,
                    hp * npad,
                    hd,
                    npad * hd,
                    npad,
                    ws.batchStartPosHolder,
                    1,
                    0,
                    z0,
                    z0,
                    kvMul,
                    0);
            int8Grids.put(graph + sTask, mma3(paddedBatch, npad, hp));
            String pTask = "attn_softmax_" + pass;
            layer.task(
                    pTask,
                    TensorCoreAttentionKernels::windowedSoftmax,
                    context,
                    ws.batchStartPosHolder,
                    scores,
                    probs,
                    hp,
                    npad,
                    1.0f,
                    window);
            WorkerGrid1D softmax =
                    new WorkerGrid1D(paddedBatch * hp * TensorCoreAttentionKernels.SOFTMAX_LANES);
            softmax.setLocalWork(TensorCoreAttentionKernels.SOFTMAX_LANES, 1, 1);
            int8Grids.put(graph + pTask, softmax);
            String aTask = "attn_values_" + pass;
            layer.task(
                    aTask,
                    BatchMmaKernels::gemmMMAStrided,
                    context,
                    probs,
                    valuesTF16,
                    attnF32,
                    paddedBatch,
                    hd,
                    npad,
                    hp * npad,
                    npad,
                    qDim,
                    npad,
                    hd * npad,
                    hd,
                    ws.batchStartPosHolder,
                    0,
                    1,
                    0,
                    z0,
                    kvMul,
                    z0);
            int8Grids.put(graph + aTask, mma3(paddedBatch, hd, hp));
        }
        layer.task(
                "attn_out16",
                BatchMmaKernels::toHalf,
                context,
                attnF32,
                ws.attnOutFP16,
                paddedBatch * qDim);
        int8Grids.put(graph + "attn_out16", elementwise(paddedBatch * qDim, 256));
    }

    private static WorkerGrid mma3(int m, int n, int batches) {
        WorkerGrid3D grid = new WorkerGrid3D((m / 128) * 256, n / 128, batches);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    /**
     * Quantizes {@code k}-wide FP16 rows of the padded chunk to int8 blocks for the GEMMs after.
     */
    private void int8Quantize(TaskGraph layer, String task, HalfFloatArray x, int k) {
        int total = paddedBatch * k;
        layer.task(
                task,
                Int8GemmKernels::quantizeActivationsQ8WarpFP16,
                context,
                x,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                total);
        WorkerGrid1D g = new WorkerGrid1D(total);
        g.setLocalWork(256, 1, 1);
        int8Grids.put(layer.getTaskGraphName() + "." + task, g);
    }

    // @formatter:off
    /**
     * {@code out[r * ldo + colOffset + c] = x8[r] . w[c]} over the quantized chunk, for {@code n}
     * outputs over {@code k} inputs. With {@code split} the GEMM may cut K into parts, so its tiles
     * fill the device, and a pass adds the parts into a plain {@code out}.
     */
    // @formatter:on
    private void int8Gemm(
            TaskGraph layer,
            String task,
            TornadoTensor w,
            FloatArray out,
            int n,
            int k,
            int ldo,
            int colOffset,
            boolean split) {
        int splits = split ? gemmSplits(n, k) : 1;
        FloatArray partial = splits > 1 ? state.workspace.splitKPartialBatch : out;
        String graph = layer.getTaskGraphName() + ".";
        layer.task(
                task,
                state.workspace.prepacking.prepackedQ4(w)
                        ? Int8GemmKernels::gemmInt8Q4_0Packed
                        : w.dataType() == DataType.Q4_0
                                ? Int8GemmKernels::gemmInt8Q4_0
                                : Int8GemmKernels::gemmInt8Q8_0,
                context,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                w.asByteArray(),
                out,
                out,
                paddedBatch,
                n,
                k,
                Int8GemmKernels.EPILOGUE_STORE,
                partial,
                splits,
                state.workspace.batchStartPosHolder,
                ldo,
                colOffset);
        WorkerGrid2D g =
                new WorkerGrid2D(
                        (paddedBatch / Int8GemmKernels.I8_BM) * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN * splits);
        g.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        int8Grids.put(graph + task, g);
        if (splits > 1) {
            int total = paddedBatch * n;
            layer.task(
                    task + "_reduce",
                    Int8GemmKernels::reduceSplitsQ8_0,
                    context,
                    state.workspace.splitKPartialBatch,
                    out,
                    total,
                    splits,
                    Int8GemmKernels.EPILOGUE_STORE,
                    state.workspace.batchStartPosHolder,
                    n);
            int8Grids.put(graph + task + "_reduce", elementwise(total, 256));
        }
    }

    /** The streaming multiprocessors of the device this plan is built for; 0 when unknown. */
    private static int smCount() {
        return TornadoDevices.target().computeUnits();
    }

    // @formatter:off
    /**
     * How many parts an int8 GEMM of {@code outputs} over {@code k} cuts K into: the count with the
     * fewest (waves x rounds) on this device, a block to a multiprocessor, taken only where it
     * saves a tenth, and no more than the partial-sum buffer holds.
     */
    // @formatter:on
    private int gemmSplits(int outputs, int k) {
        int sms = smCount();
        if (sms <= 0) {
            return 1;
        }
        int tiles = (paddedBatch / Int8GemmKernels.I8_BM) * (outputs / Int8GemmKernels.I8_BN);
        int rounds = k / Int8GemmKernels.I8_BK;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= SPLIT_K_SLICES && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return bestCost * 10 <= unsplit * 9 ? best : 1;
    }

    // @formatter:off
    /**
     * The value projection of a layer. A layer without one takes its value from the raw key, so
     * projecting the key a second time into the value's slot gives the packed {@code [q|k|v]} row
     * the head norms and the cache writes already read.
     */
    // @formatter:on
    private TornadoTensor valueWeights(int l) {
        return weights.wvLayered[l] != null ? weights.wvLayered[l] : weights.wkLayered[l];
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
        if (type != DataType.Q8_0
                && type != DataType.Q4_0
                && type != DataType.Q4_1
                && type != DataType.F16) {
            throw new UnsupportedOperationException(
                    "gemma4 batched prefill has no decoder for "
                            + type
                            + " ("
                            + name
                            + "); the trunk must be Q8_0, Q4_0, Q4_1 or F16");
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
        if (state.workspace.prepacking.prepackedQ4(w)) {
            // A prepacked weight has only the int8 GEMM (-Djitllm.gemma4.noInt8Prefill and native
            // projections are not open to it).
            throw WeightPrepacking.noPrepackedKernel(
                    "gemma4 batch-prefill task '" + taskName + "'");
        }
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
            case F16 ->
                    layer.task(
                            taskName,
                            Gemma4BatchPrefillKernels::copyF16,
                            context,
                            w.asHalfFloatArray(),
                            state.workspace.weightsF16Scratch,
                            destOffset,
                            w.asHalfFloatArray().getSize());
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
        if (simdgroupAttention(layerIndex)) {
            return false;
        }
        return fp16KeyValue
                && config.headDim(layerIndex) % Gemma4AttentionKernels.TC_HEAD == 0
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /** Whether this state's prefill projections are cuBLAS GEMMs: requested, and available. */
    static boolean nativeProjections(Gemma4State state) {
        return NativePrefillSupport.nativeProjections(state.executionPolicy());
    }

    // ── MLX projections ──────────────────────────────────────────────────────

    /**
     * The MLX prefill a session running {@code policy} builds over this model, or empty when its
     * batched prefill keeps the Metal kernels. The one answer the prefill layers, the decode layers
     * that bind their weights and the memory preflight all read.
     */
    public static Optional<MlxPrefill> mlxPrefill(
            ExecutionPolicy policy, DataType weights, Gemma4Configuration config) {
        return MlxPrefill.resolve(policy, weights, config);
    }

    /**
     * What the MLX projections allocate besides the model's own weights: the projections in MLX's
     * form (resident FP16 copies, or the affine repack), and the FP16 operands and products of
     * {@code paddedBatch} rows, one per distinct width, with the row factors.
     */
    public static BatchPrefillWorkspace mlxWorkspace(
            DataType weights, Gemma4Configuration config, int batchSize) {
        long rows = (batchSize + 127) & ~127;
        java.util.Set<Integer> qkv = new java.util.HashSet<>();
        java.util.Set<Integer> attn = new java.util.HashSet<>();
        java.util.Set<Integer> ffn = new java.util.HashSet<>();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int qDim = config.numberOfHeads() * config.headDim(l);
            qkv.add(config.hasOwnKv(l) ? qDim + 2 * config.keyValueDim(l) : qDim);
            attn.add(qDim);
            ffn.add(config.feedForwardLength(l));
        }
        long widths = config.dim();
        for (int w : qkv) {
            widths += w;
        }
        for (int w : attn) {
            widths += w;
        }
        for (int w : ffn) {
            widths += 3L * w;
        }
        long staging = Short.BYTES * rows * widths + Float.BYTES * rows;
        long projections = config.projectionWeights();
        return new BatchPrefillWorkspace(
                projections,
                staging + FP16Projections.residentBytes(projections),
                staging + MlxPrefill.affineBytes(weights, projections));
    }

    /** Each layer's projections in MLX's form, and the FP16 buffers the GEMMs share. */
    private void buildMlxProjections() {
        int layers = config.numberOfLayers();
        mlxQkv = new MlxPrefillProjections.Matrix[layers];
        mlxWo = new MlxPrefillProjections.Matrix[layers];
        mlxGateUp = new MlxPrefillProjections.Matrix[layers];
        mlxDown = new MlxPrefillProjections.Matrix[layers];
        MlxPrefill.Projections form = mlx.projections();
        for (int l = firstLayer; l < endLayer; l++) {
            TornadoTensor[] qkv =
                    config.hasOwnKv(l)
                            ? new TornadoTensor[] {
                                weights.wqLayered[l], weights.wkLayered[l], valueWeights(l)
                            }
                            : new TornadoTensor[] {weights.wqLayered[l]};
            String name = "blk." + l;
            mlxQkv[l] = MlxPrefillProjections.matrix(form, requireMlx(name + ".attn_qkv", qkv));
            mlxWo[l] =
                    MlxPrefillProjections.matrix(
                            form, requireMlx(name + ".attn_output", weights.woLayered[l]));
            mlxGateUp[l] =
                    MlxPrefillProjections.matrix(
                            form,
                            requireMlx(
                                    name + ".ffn_gate_up",
                                    weights.w1Layered[l],
                                    weights.w3Layered[l]));
            mlxDown[l] =
                    MlxPrefillProjections.matrix(
                            form, requireMlx(name + ".ffn_down", weights.w2Layered[l]));
        }
        mlxOutHalf = new HalfFloatArray(paddedBatch * dim);
        mlxDownRowScale = new FloatArray(paddedBatch);
        for (int l = firstLayer; l < endLayer; l++) {
            int qDim = nHead * config.headDim(l);
            int ffnLen = config.feedForwardLength(l);
            for (Object buffer :
                    new Object[] {
                        mlxBuffer(mlxQkvOut, qkvStride(l)),
                        mlxBuffer(mlxAttnIn, qDim),
                        mlxBuffer(mlxGateUpOut, 2 * ffnLen),
                        mlxBuffer(mlxHbIn, ffnLen),
                        mlxOutHalf,
                        mlxDownRowScale
                    }) {
                mlxFirstUser.putIfAbsent(buffer, l);
            }
        }
    }

    /** The buffer of {@code paddedBatch} rows of {@code width} in {@code byWidth}, made once. */
    private HalfFloatArray mlxBuffer(java.util.Map<Integer, HalfFloatArray> byWidth, int width) {
        return byWidth.computeIfAbsent(width, w -> new HalfFloatArray(paddedBatch * w));
    }

    /**
     * Refuses, by name, a projection the MLX prefill was not resolved for: another type, or a
     * prepacked weight. Never handed back to the Metal kernels, so the decode layers that ask the
     * same {@link #mlxPrefill} which weights these graphs bind are always right.
     */
    private TornadoTensor[] requireMlx(String name, TornadoTensor... tensors) {
        for (TornadoTensor t : tensors) {
            if (state.workspace.prepacking.prepacked(t)) {
                throw WeightPrepacking.noPrepackedKernel(
                        "The gemma4 MLX batched prefill (" + name + ")");
            }
            if (t.dataType() != mlx.weights()) {
                throw new IllegalStateException(
                        "the gemma4 MLX batched prefill was resolved for "
                                + mlx.weights()
                                + " weights, but "
                                + name
                                + " is "
                                + t.dataType());
            }
        }
        return tensors;
    }

    /**
     * The stacked query/key/value GEMM on MLX into the packed {@code [q|k|v]} rows the head norms
     * and the rotation read, in FP32.
     */
    private void mlxQkvProjection(TaskGraph layer, int layerIndex, int stride) {
        MlxPrefillProjections.gemm(
                layer,
                "qkvProj",
                state.workspace.wrapXbFP16Batch,
                mlxQkv[layerIndex],
                mlxQkvOut.get(stride),
                paddedBatch,
                dim,
                stride);
        layer.task(
                "mlx_qkv_float",
                MlxPrefillKernels::toFloat,
                context,
                mlxQkvOut.get(stride),
                state.workspace.qkvResultBatch,
                batchSize * stride);
    }

    /**
     * Declares the MLX buffers layer {@code l}'s graph uses: uploaded by the first graph that uses
     * each, bound from that graph by every later one. From that graph and not the previous one: a
     * graph that declares an array none of its tasks reads never allocates it.
     */
    private void declareMlxBuffers(TaskGraph layer, int l) {
        int qDim = nHead * config.headDim(l);
        int ffnLen = config.feedForwardLength(l);
        for (Object buffer :
                new Object[] {
                    mlxQkvOut.get(qkvStride(l)),
                    mlxAttnIn.get(qDim),
                    mlxGateUpOut.get(2 * ffnLen),
                    mlxHbIn.get(ffnLen),
                    mlxOutHalf,
                    mlxDownRowScale
                }) {
            int firstUser = mlxFirstUser.get(buffer);
            if (firstUser == l) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffer);
            } else {
                layer.consumeFromDevice(GraphNames.batchPrefillLayer(firstUser), buffer);
            }
        }
    }

    /** Layer {@code l}'s projections in MLX's form, which its graph uploads. */
    private Object[] mlxWeights(int l) {
        List<Object> arrays = new ArrayList<>();
        for (MlxPrefillProjections.Matrix m :
                new MlxPrefillProjections.Matrix[] {
                    mlxQkv[l], mlxWo[l], mlxGateUp[l], mlxDown[l]
                }) {
            arrays.addAll(List.of(m.arrays()));
        }
        return arrays.toArray();
    }

    // @formatter:off
    /**
     * {@code out[m][n] = x[m][k] . w[n][k]} as one cuBLAS GEMM: FP16 operands, FP32 accumulation
     * and output. Row-major on both sides, which is {@code out^T = w . x^T} in cuBLAS's
     * column-major terms: the weight transposed, the activation as is.
     */
    // @formatter:on
    private void nativeGemm(
            TaskGraph layer,
            String task,
            HalfFloatArray w,
            HalfFloatArray x,
            FloatArray out,
            int n,
            int k) {
        layer.libraryTask(
                task,
                uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16FP32,
                1,
                0,
                n,
                paddedBatch,
                k,
                1.0f,
                w,
                k,
                x,
                k,
                0.0f,
                out,
                n);
    }

    // @formatter:off
    /**
     * {@code out[m * ldo + off + r] = x[m] . w[r]} on the Metal SIMD-group GEMMs, the FP16
     * activations read as they are staged: one projection written into its columns of a packed row.
     */
    // @formatter:on
    private void simdgroupGemm(
            TaskGraph layer,
            String task,
            HalfFloatArray x,
            Object w,
            FloatArray out,
            int n,
            int d,
            int ldo,
            int off) {
        simdgroupTasks.put(layer.getTaskGraphName() + "." + task, d);
        if (w instanceof TornadoTensor t) {
            if (t.dataType() != DataType.Q8_0) {
                throw new UnsupportedOperationException(
                        "gemma4 batched prefill on Metal reads Q8_0 projections only, not "
                                + t.dataType());
            }
            layer.task(
                    task,
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmHalfQ8Strided,
                    context,
                    x,
                    out,
                    t.asByteArray(),
                    n,
                    d,
                    batchSize,
                    ldo,
                    off);
        } else {
            layer.task(
                    task,
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmHalfFP16Strided,
                    context,
                    x,
                    out,
                    (HalfFloatArray) w,
                    n,
                    d,
                    batchSize,
                    ldo,
                    off);
        }
    }

    // @formatter:off
    /**
     * Whether a layer's prefill attention runs on the Metal SIMD-group kernels: over either cache,
     * for the 256- and 512-wide heads they are written for. On with the SIMD-group projections;
     * {@code -Djitllm.metal.simdgroupAttention=false} keeps the scalar kernel.
     */
    // @formatter:on
    private boolean simdgroupAttention(int layerIndex) {
        int headDim = config.headDim(layerIndex);
        return simdgroup
                && (headDim == 256 || headDim == 512)
                && !"false".equals(System.getProperty("jitllm.metal.simdgroupAttention"));
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
        for (int l = firstLayer; l < endLayer; l++) {
            if (config.isSwa(l) == isSwa) {
                firstUser = l;
                break;
            }
        }
        String graphName = layer.getTaskGraphName();
        if (graphName.equals(GraphNames.batchPrefillLayer(firstUser))) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, freqCisReal, freqCisImag);
        } else {
            layer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(firstUser), freqCisReal, freqCisImag);
        }
    }

    /** The packed [q|k|v] row stride this layer's projection writes. */
    private int qkvStride(int layerIndex) {
        int headDim = config.headDim(layerIndex);
        int qDim = nHead * headDim;
        return config.hasOwnKv(layerIndex)
                ? qDim + 2 * config.keyValueHeads(layerIndex) * headDim
                : qDim;
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = GraphNames.batchPrefillLayer(layerIndex);
        if (layerIndex == endLayer - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);

        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int nHeadKv = config.keyValueHeads(layerIndex);
        final int kvMul = config.kvMul(layerIndex);
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
        if (layerIndex == firstLayer) {
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
                    state.workspace.w2Out,
                    state.workspace.wrapQ8ActBatch,
                    state.workspace.wrapQ8ActScales);
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
            if (gemmAttention) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        queriesF16,
                        keysF16,
                        valuesTF16,
                        scores,
                        probs,
                        attnF32);
            }
            if (ple) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        pleModelProjF16,
                        weights.perLayerProjNorm.asFloatArray());
            }
            layer.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
        } else {
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
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
                    state.workspace.w2Out,
                    state.workspace.wrapQ8ActBatch,
                    state.workspace.wrapQ8ActScales);
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
                        GraphNames.batchPrefillLayer(firstLayer),
                        state.workspace.attnOutF32Batch,
                        state.workspace.attnProbStageBatch);
            }
            if (gemmAttention) {
                layer.consumeFromDevice(
                        pred, queriesF16, keysF16, valuesTF16, scores, probs, attnF32);
            }
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.attnQNorm[layerIndex].asFloatArray(),
                weights.attnPostNorm[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.ffnPostNorm[layerIndex].asFloatArray());
        if (ple) {
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    pleGateF16[layerIndex],
                    pleProjF16[layerIndex],
                    weights.perLayerPostNorm[layerIndex].asFloatArray());
        }
        // Each projection in the file's representation: the int8 and FP16 GEMMs read it, and
        // cuBLAS reads it decoded into the shared FP16 scratch. MLX reads its own form instead, so
        // the decode layers upload the file's projections themselves.
        Object[] projections = {
            weights.wqLayered[layerIndex],
            weights.woLayered[layerIndex],
            weights.w1Layered[layerIndex],
            weights.w3Layered[layerIndex],
            weights.w2Layered[layerIndex]
        };
        if (mlx != null) {
            declareMlxBuffers(layer, layerIndex);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, mlxWeights(layerIndex));
        } else {
            state.workspace.prepacker.configureGraphInputs(
                    layer, state.workspace.prepacking, projections);
        }
        if (hasOwnKv) {
            requireDecodable(weights.wkLayered[layerIndex], "blk." + layerIndex + ".attn_k");
            requireDecodable(valueWeights(layerIndex), "blk." + layerIndex + ".attn_v");
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, weights.attnKNorm[layerIndex].asFloatArray());
            if (mlx == null) {
                state.workspace.prepacker.configureGraphInputs(
                        layer, state.workspace.prepacking, weights.wkLayered[layerIndex]);
                if (weights.wvLayered[layerIndex] != null) {
                    state.workspace.prepacker.configureGraphInputs(
                            layer, state.workspace.prepacking, weights.wvLayered[layerIndex]);
                }
            }
        }
        if (weights.layerOutputScale[layerIndex] != null) {
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }

        bindRopeTables(layer, isSwa, freqCisReal, freqCisImag);
        if (layerIndex == 0) {
            appendScaleEmbedding(layer);
            if (ple) {
                appendPleSetup(layer);
            }
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
                                    valueWeights(layerIndex));
            if (mlx != null) {
                mlxQkvProjection(layer, layerIndex, stride);
            } else if (simdgroup) {
                simdgroupGemm(
                        layer,
                        "qkvProj",
                        state.workspace.wrapXbFP16Batch,
                        weights.wqLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        dim,
                        qDim,
                        stride,
                        0);
                simdgroupGemm(
                        layer,
                        "kProj",
                        state.workspace.wrapXbFP16Batch,
                        weights.wkLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        dim,
                        kvDim,
                        stride,
                        qDim);
                simdgroupGemm(
                        layer,
                        "vProj",
                        state.workspace.wrapXbFP16Batch,
                        weights.wvLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        dim,
                        kvDim,
                        stride,
                        qDim + kvDim);
            } else if (int8Qkv[layerIndex]) {
                // Three GEMMs over one quantized activation, each into its slot of the packed
                // [q|k|v] row; a layer without a value projection projects the key again.
                int8Quantize(layer, "qkv_q8", state.workspace.wrapXbFP16Batch, dim);
                int8Gemm(
                        layer,
                        "q_i8",
                        weights.wqLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        qDim,
                        dim,
                        stride,
                        0,
                        false);
                int8Gemm(
                        layer,
                        "k_i8",
                        weights.wkLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        kvDim,
                        dim,
                        stride,
                        qDim,
                        false);
                int8Gemm(
                        layer,
                        "v_i8",
                        valueWeights(layerIndex),
                        state.workspace.qkvResultBatch,
                        kvDim,
                        dim,
                        stride,
                        qDim + kvDim,
                        false);
            } else if (nativeProjections) {
                addDequant(layer, "qDequant", weights.wqLayered[layerIndex], 0);
                addDequant(layer, "kDequant", weights.wkLayered[layerIndex], qDim * dim);
                addDequant(layer, "vDequant", valueWeights(layerIndex), (qDim + kvDim) * dim);
                nativeGemm(
                        layer,
                        "qkvProj",
                        state.workspace.weightsF16Scratch,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.qkvResultBatch,
                        qDim + 2 * kvDim,
                        dim);
            } else if (!qkvDirect) {
                // Query, key and value laid end to end make one [qDim + 2*kvDim, dim] matrix, so a
                // single plain GEMM writes exactly the packed [q|k|v] row the head norms and the
                // rotation already read.
                addDequant(layer, "qDequant", weights.wqLayered[layerIndex], 0);
                addDequant(layer, "kDequant", weights.wkLayered[layerIndex], qDim * dim);
                addDequant(layer, "vDequant", valueWeights(layerIndex), (qDim + kvDim) * dim);
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
                        valueWeights(layerIndex).asByteArray(),
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
            if (mlx != null) {
                mlxQkvProjection(layer, layerIndex, stride);
            } else if (simdgroup) {
                simdgroupGemm(
                        layer,
                        "qkvProj",
                        state.workspace.wrapXbFP16Batch,
                        weights.wqLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        dim,
                        qDim,
                        stride,
                        0);
            } else if (int8Qkv[layerIndex]) {
                int8Quantize(layer, "qkv_q8", state.workspace.wrapXbFP16Batch, dim);
                int8Gemm(
                        layer,
                        "q_i8",
                        weights.wqLayered[layerIndex],
                        state.workspace.qkvResultBatch,
                        qDim,
                        dim,
                        stride,
                        0,
                        false);
            } else if (nativeProjections) {
                addDequant(layer, "qDequant", weights.wqLayered[layerIndex], 0);
                nativeGemm(
                        layer,
                        "qkvProj",
                        state.workspace.weightsF16Scratch,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.qkvResultBatch,
                        qDim,
                        dim);
            } else if (!DEQUANT_PROJECTIONS && allQ8(weights.wqLayered[layerIndex])) {
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

        // On MLX the attention writes straight into the output projection's exact-width operand.
        final HalfFloatArray attnOut =
                mlx != null ? mlxAttnIn.get(qDim) : state.workspace.attnOutFP16;
        if (simdgroupAttention(layerIndex)) {
            if (fp16KeyValue) {
                layer.task(
                        "batch_attention",
                        headDim == 256
                                ? Gemma4BatchPrefillSimdgroupAttentionFP16Kernels
                                        ::batchedSlidingWindowAttentionKVFP16Head256
                                : Gemma4BatchPrefillSimdgroupAttentionFP16Kernels
                                        ::batchedSlidingWindowAttentionKVFP16Head512,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        attnOut,
                        nHead,
                        headDim,
                        kvDim,
                        kvMul,
                        stride,
                        cacheBaseOffset,
                        windowSize);
            } else {
                layer.task(
                        "batch_attention",
                        headDim == 256
                                ? Gemma4BatchPrefillSimdgroupAttentionKernels
                                        ::batchedSlidingWindowAttentionHead256
                                : Gemma4BatchPrefillSimdgroupAttentionKernels
                                        ::batchedSlidingWindowAttentionHead512,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        attnOut,
                        nHead,
                        headDim,
                        kvDim,
                        kvMul,
                        stride,
                        cacheBaseOffset,
                        windowSize);
            }
        } else if (gemmAttention) {
            gemmAttention(layer, headDim, kvMul, qDim, kvDim, stride, cacheBaseOffset, windowSize);
        } else if (tensorCoreAttention(layerIndex)) {
            layer.task(
                    "batch_attention",
                    Gemma4AttentionKernels::attentionPrefillTensorCoreFP16,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutF32Batch,
                    attnOut,
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
                    attnOut,
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
                    attnOut,
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

        if (mlx != null) {
            MlxPrefillProjections.gemm(
                    layer,
                    "woProj",
                    attnOut,
                    mlxWo[layerIndex],
                    mlxOutHalf,
                    paddedBatch,
                    qDim,
                    dim);
            layer.task(
                    "mlx_wo_float",
                    MlxPrefillKernels::toFloat,
                    context,
                    mlxOutHalf,
                    state.workspace.woOut,
                    batchSize * dim);
        } else if (simdgroup) {
            simdgroupGemm(
                    layer,
                    "woProj",
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex],
                    state.workspace.woOut,
                    qDim,
                    dim,
                    dim,
                    0);
        } else if (int8Wo[layerIndex]) {
            int8Quantize(layer, "wo_q8", state.workspace.attnOutFP16, qDim);
            int8Gemm(
                    layer,
                    "wo_i8",
                    weights.woLayered[layerIndex],
                    state.workspace.woOut,
                    dim,
                    qDim,
                    dim,
                    0,
                    true);
        } else if (nativeProjections) {
            addDequant(layer, "woDequant", weights.woLayered[layerIndex], 0);
            nativeGemm(
                    layer,
                    "woProj",
                    state.workspace.weightsF16Scratch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    dim,
                    qDim);
        } else if (SPLIT_K && (DEQUANT_SPLIT_K || !allQ8(weights.woLayered[layerIndex]))) {
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
        if (mlx != null) {
            // The GeGLU product of a token can pass FP16's range; its row is scaled into range
            // before the down GEMM, and the scale undone in FP32 after it.
            // Over the padded rows too: the down GEMM reads every row of its operand.
            MlxPrefillProjections.gemm(
                    layer,
                    "gateUpProj",
                    state.workspace.normedXFFNFP16,
                    mlxGateUp[layerIndex],
                    mlxGateUpOut.get(2 * ffnLen),
                    paddedBatch,
                    dim,
                    2 * ffnLen);
            layer.task(
                    "mlx_ffn_row_scale",
                    MlxPrefillKernels::gegluRowScale,
                    context,
                    mlxGateUpOut.get(2 * ffnLen),
                    mlxDownRowScale,
                    ffnLen,
                    paddedBatch,
                    ROW_THREADS);
            layer.task(
                    "batch_geglu",
                    MlxPrefillKernels::gegluScaled,
                    context,
                    mlxGateUpOut.get(2 * ffnLen),
                    mlxHbIn.get(ffnLen),
                    mlxDownRowScale,
                    ffnLen,
                    paddedBatch * ffnLen);
        } else if (simdgroup) {
            simdgroupGemm(
                    layer,
                    "gateProj",
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex],
                    state.workspace.gateUpResultBatch,
                    dim,
                    ffnLen,
                    2 * ffnLen,
                    0);
            simdgroupGemm(
                    layer,
                    "upProj",
                    state.workspace.normedXFFNFP16,
                    weights.w3Layered[layerIndex],
                    state.workspace.gateUpResultBatch,
                    dim,
                    ffnLen,
                    2 * ffnLen,
                    ffnLen);
        } else if (int8GateUp[layerIndex]) {
            // Gate and up into the two halves of the packed [gate|up] row the activation reads.
            int8Quantize(layer, "ffn_q8", state.workspace.normedXFFNFP16, dim);
            int8Gemm(
                    layer,
                    "gate_i8",
                    weights.w1Layered[layerIndex],
                    state.workspace.gateUpResultBatch,
                    ffnLen,
                    dim,
                    2 * ffnLen,
                    0,
                    false);
            int8Gemm(
                    layer,
                    "up_i8",
                    weights.w3Layered[layerIndex],
                    state.workspace.gateUpResultBatch,
                    ffnLen,
                    dim,
                    2 * ffnLen,
                    ffnLen,
                    false);
        } else if (nativeProjections) {
            addDequant(layer, "gateDequant", weights.w1Layered[layerIndex], 0);
            addDequant(layer, "upDequant", weights.w3Layered[layerIndex], ffnLen * dim);
            nativeGemm(
                    layer,
                    "gateUpProj",
                    state.workspace.weightsF16Scratch,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    2 * ffnLen,
                    dim);
        } else if (DEQUANT_GATE_UP
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
        if (mlx == null) {
            layer.task(
                    "batch_geglu",
                    Gemma4BatchPrefillKernels::batchedGeGLUFP16Packed,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.gateUpResultBatch,
                    ffnLen);
        }
        if (mlx != null) {
            MlxPrefillProjections.gemm(
                    layer,
                    "w2Proj",
                    mlxHbIn.get(ffnLen),
                    mlxDown[layerIndex],
                    mlxOutHalf,
                    paddedBatch,
                    ffnLen,
                    dim);
            layer.task(
                    "mlx_w2_float",
                    MlxPrefillKernels::toFloatRowScaled,
                    context,
                    mlxOutHalf,
                    state.workspace.w2Out,
                    mlxDownRowScale,
                    dim,
                    batchSize * dim);
        } else if (simdgroup) {
            simdgroupGemm(
                    layer,
                    "w2Proj",
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex],
                    state.workspace.w2Out,
                    ffnLen,
                    dim,
                    dim,
                    0);
        } else if (int8Down[layerIndex]) {
            int8Quantize(layer, "down_q8", state.workspace.wrapHbFP16Batch, ffnLen);
            int8Gemm(
                    layer,
                    "down_i8",
                    weights.w2Layered[layerIndex],
                    state.workspace.w2Out,
                    dim,
                    ffnLen,
                    dim,
                    0,
                    true);
        } else if (nativeProjections) {
            addDequant(layer, "w2Dequant", weights.w2Layered[layerIndex], 0);
            nativeGemm(
                    layer,
                    "w2Proj",
                    state.workspace.weightsF16Scratch,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out,
                    dim,
                    ffnLen);
        } else if (SPLIT_K && (DEQUANT_SPLIT_K || !allQ8(weights.w2Layered[layerIndex]))) {
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
        if (ple) {
            layer.task(
                    "batch_x_cast",
                    TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXFP16Batch);
            if (simdgroup) {
                simdgroupGemm(
                        layer,
                        "pleGateProj",
                        state.workspace.wrapXFP16Batch,
                        pleGateF16[layerIndex],
                        state.workspace.wrapPerLayerGateBatch,
                        dim,
                        nEmbdPerLayer,
                        nEmbdPerLayer,
                        0);
            } else if (nativeProjections) {
                nativeGemm(
                        layer,
                        "pleGateProj",
                        pleGateF16[layerIndex],
                        state.workspace.wrapXFP16Batch,
                        state.workspace.wrapPerLayerGateBatch,
                        nEmbdPerLayer,
                        dim);
            } else {
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
            }
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
            if (simdgroup) {
                simdgroupGemm(
                        layer,
                        "pleProj",
                        state.workspace.wrapPerLayerGateFP16Batch,
                        pleProjF16[layerIndex],
                        state.workspace.wrapPerLayerOutBatch,
                        nEmbdPerLayer,
                        dim,
                        dim,
                        0);
            } else if (nativeProjections) {
                nativeGemm(
                        layer,
                        "pleProj",
                        pleProjF16[layerIndex],
                        state.workspace.wrapPerLayerGateFP16Batch,
                        state.workspace.wrapPerLayerOutBatch,
                        dim,
                        nEmbdPerLayer);
            } else {
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
            }
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
        }

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
                "batch_embed_cast",
                TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapXFP16Batch);
        if (simdgroup) {
            simdgroupGemm(
                    layer,
                    "pleModelProj",
                    state.workspace.wrapXFP16Batch,
                    pleModelProjF16,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    dim,
                    perLayerTotal,
                    perLayerTotal,
                    0);
        } else if (nativeProjections) {
            nativeGemm(
                    layer,
                    "pleModelProj",
                    pleModelProjF16,
                    state.workspace.wrapXFP16Batch,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    perLayerTotal,
                    dim);
        } else {
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
        }
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

    /** Layer 0 scales the embeddings by the square root of the model width. */
    private void appendScaleEmbedding(TaskGraph layer) {
        layer.task(
                "batch_scale_embedding",
                Gemma4BatchPrefillKernels::batchedScaleInPlace,
                context,
                state.workspace.wrapXBatch,
                embedScale);
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
        WorkerGrid pleGateWorker = ple ? mmaGrid(paddedBatch, nEmbdPerLayer) : null;
        WorkerGrid pleGateGeluWorker = ple ? elementwise(batchSize * nEmbdPerLayer, 256) : null;
        WorkerGrid reduceWorker = elementwise(paddedBatch * dim, 256);

        for (int l = firstLayer; l < endLayer; l++) {
            String p = GraphNames.batchPrefillLayer(l) + ".";
            int headDim = config.headDim(l);
            int nHeadKv = config.keyValueHeads(l);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(l);
            boolean hasOwnKv = config.hasOwnKv(l);

            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", dimApplyWorker);
            // Native projections are library calls, launched by cuBLAS: no grids to register.
            boolean qkvDirect =
                    !DEQUANT_PROJECTIONS
                            && (hasOwnKv
                                    ? allQ8(
                                            weights.wqLayered[l],
                                            weights.wkLayered[l],
                                            valueWeights(l))
                                    : allQ8(weights.wqLayered[l]));
            if (mlx != null) {
                // MLX library tasks have no grids; only the glue around them does.
                int stride = qkvStride(l);
                scheduler.addWorkerGrid(p + "mlx_qkv_float", elementwise(batchSize * stride, 256));
                scheduler.addWorkerGrid(p + "mlx_wo_float", dimApplyWorker);
                scheduler.addWorkerGrid(
                        p + "mlx_ffn_row_scale",
                        WorkerGridFactory.genericWorker(paddedBatch * ROW_THREADS, ROW_THREADS));
                scheduler.addWorkerGrid(p + "mlx_w2_float", dimApplyWorker);
            }
            if (mlx == null && !nativeProjections && !simdgroup && !int8Qkv[l]) {
                scheduler.addWorkerGrid(
                        p + "qkvProj", mmaGrid(paddedBatch, hasOwnKv ? qDim + 2 * kvDim : qDim));
            }
            if (mlx == null && !simdgroup && (nativeProjections || (!qkvDirect && !int8Qkv[l]))) {
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
            if (!gemmAttention)
                scheduler.addWorkerGrid(
                        p + "batch_attention",
                        simdgroupAttention(l)
                                ? WorkerGridFactory.genericWorker(
                                        ((batchSize
                                                                + Gemma4BatchPrefillSimdgroupAttentionKernels
                                                                        .QUERY_TILE
                                                                - 1)
                                                        / Gemma4BatchPrefillSimdgroupAttentionKernels
                                                                .QUERY_TILE)
                                                * nHead
                                                * Gemma4BatchPrefillSimdgroupAttentionKernels
                                                        .THREADS,
                                        Gemma4BatchPrefillSimdgroupAttentionKernels.THREADS)
                                : tensorCoreAttention(l)
                                        ? WorkerGridFactory.genericWorker(
                                                paddedBatch
                                                        / Gemma4AttentionKernels.TC_QUERIES
                                                        * nHead
                                                        * Gemma4AttentionKernels.TC_LANES,
                                                Gemma4AttentionKernels.TC_LANES)
                                        : WorkerGridFactory.genericWorker(
                                                paddedBatch * nHead * HEAD_LOCAL_SIZE,
                                                HEAD_LOCAL_SIZE));
            if (simdgroup || mlx != null) {
                // the SIMD-group grids below, or MLX's
            } else if (nativeProjections) {
                // cuBLAS, over the projection decoded into the scratch
                scheduler.addWorkerGrid(p + "woDequant", elementwise(dim * qDim, 256));
            } else if (int8Wo[l]) {
                // recorded with the int8 tasks
            } else if (SPLIT_K) {
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
            if (mlx == null && !nativeProjections && !simdgroup && !int8GateUp[l]) {
                scheduler.addWorkerGrid(p + "gateUpProj", mmaGrid(paddedBatch, 2 * ffnLen));
            }
            if (mlx == null
                    && !simdgroup
                    && (nativeProjections
                            || !int8GateUp[l]
                                    && (DEQUANT_GATE_UP
                                            || !allQ8(
                                                    weights.w1Layered[l], weights.w3Layered[l])))) {
                WorkerGrid dequantWorker = elementwise(ffnLen * dim, 256);
                scheduler.addWorkerGrid(p + "gateDequant", dequantWorker);
                scheduler.addWorkerGrid(p + "upDequant", dequantWorker);
            }
            scheduler.addWorkerGrid(
                    p + "batch_geglu",
                    elementwise((mlx != null ? paddedBatch : batchSize) * ffnLen, 256));
            if (simdgroup || mlx != null) {
                // the SIMD-group grids below, or MLX's
            } else if (nativeProjections) {
                // cuBLAS, over the projection decoded into the scratch
                scheduler.addWorkerGrid(p + "w2Dequant", elementwise(dim * ffnLen, 256));
            } else if (int8Down[l]) {
                // recorded with the int8 tasks
            } else if (SPLIT_K) {
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

            if (ple) {
                scheduler.addWorkerGrid(p + "batch_x_cast", dimApplyWorker);
                if (!nativeProjections && !simdgroup) {
                    scheduler.addWorkerGrid(p + "pleGateProj", pleGateWorker);
                }
                scheduler.addWorkerGrid(p + "batch_ple_gate_gelu", pleGateGeluWorker);
                if (!nativeProjections && !simdgroup) {
                    scheduler.addWorkerGrid(p + "pleProj", mmaDimWorker);
                }
                scheduler.addWorkerGrid(p + "batch_ple_post_rms", rmsWorker);
                scheduler.addWorkerGrid(p + "batch_ple_post_apply", dimApplyWorker);
            }
            if (weights.layerOutputScale[l] != null) {
                scheduler.addWorkerGrid(p + "batch_layer_output_scale", dimApplyWorker);
            }
        }

        int8Grids.forEach(scheduler::addWorkerGrid);
        if (firstLayer > 0) {
            return;
        }
        String p0 = "batchPrefillLayer_0.";
        scheduler.addWorkerGrid(p0 + "batch_scale_embedding", dimApplyWorker);
        if (!ple) {
            return;
        }
        scheduler.addWorkerGrid(p0 + "batch_embed_cast", dimApplyWorker);
        if (!nativeProjections && !simdgroup) {
            scheduler.addWorkerGrid(p0 + "pleModelProj", mmaGrid(paddedBatch, perLayerTotal));
        }
        scheduler.addWorkerGrid(
                p0 + "batch_ple_proj_scale_norm",
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfLayers() * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p0 + "batch_ple_merge", elementwise(batchSize * perLayerTotal, 256));
        int sgLocal = TransformerBatchPrefillSimdgroupKernels.THREADS;
        for (var task : simdgroupTasks.entrySet()) {
            scheduler.addWorkerGrid(
                    task.getKey(),
                    WorkerGridFactory.genericWorker(
                            ((task.getValue() + 63) / 64) * ((batchSize + 63) / 64) * sgLocal,
                            sgLocal));
        }
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
