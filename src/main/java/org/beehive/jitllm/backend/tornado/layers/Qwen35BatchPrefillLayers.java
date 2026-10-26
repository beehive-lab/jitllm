package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MMAKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.plan.FusedOperandSupport;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The {@code qwen35} transformer layers over a chunk of prompt tokens, one task graph per block.
 *
 * <p>The same computation the single-token graphs perform, with a row index — except in the two
 * places where a row depends on the row before it. There the kernel scans the chunk itself, in
 * token order, one lane per convolution channel or per delta-net value column. See {@code
 * Qwen35BatchKernels} for why that is exact and why it makes the result independent of the chunk
 * size.
 *
 * <p>Every weight-reading task is still selected by that tensor's own representation, and the fused
 * gate/up task still requires its two operands to agree. Nothing is materialized for the batched
 * path that is not materialized for the single-token one, which is to say nothing at all.
 *
 * <p>Graph names are {@code batchLayer_i}, distinct from the decode graphs' {@code layer_i}: the
 * batched plan holds both families at once, and the decode ones consume what these produced.
 */
// @formatter:on
public class Qwen35BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeNativeLibraries() {
        return org.beehive.jitllm.backend.tornado.NativePrefillSupport.nativeProjections(
                        state.executionPolicy())
                ? "cuBLAS"
                : "none";
    }

    @Override
    public String describeProjections() {
        if (q8DirectGemms.isEmpty()
                && dequantTasks.isEmpty()
                && mmaTasks.isEmpty()
                && q8DequantTasks.isEmpty()) {
            return "JIT kernels (no tensor-core MMA at width " + batchSize + ")";
        }
        return "tensor-core projections: "
                + q8DirectGemms.size()
                + " INT8 / "
                + dequantTasks.size()
                + " dequantized FP16 / "
                + mmaTasks.size()
                + " direct FP16"
                + (q8DequantTasks.isEmpty() ? "" : " / " + q8DequantTasks.size() + " cuBLAS TF32");
    }

    // @formatter:off
    /**
     * Whether this family's eligible batched projections run on the tensor cores.
     *
     * <p>On by default; {@code -Djitllm.qwen35.tensorCores=false} (a developer diagnostic, not a
     * launcher option) keeps the scalar batched kernels, whose multiplicands are not FP16 and whose
     * outputs are therefore not bit-identical to the tensor-core path. It applies only where the
     * backend has tensor cores ({@code TensorCoreSupport.isTensorCoreCapableBackend()}, CUDA);
     * elsewhere the scalar kernels run regardless. What it covers is every projection {@link
     * #mmaEligible} admits — the gate and up panels, {@code ffn_down} in both of this model's
     * representations, the Q5_K recurrent readout and the attention output.
     *
     * <p>Which kernels a plan is built to dispatch is a question about that plan, not about this
     * field: the {@code Qwen35Mma*} accel classes check their own plan's grid scheduler and task
     * graphs, and {@code theBatchedPlanSelectsTensorCoresPerWidth} pins the per-width selection and
     * the MMA grids against the scalar path at an ineligible width.
     */
    // @formatter:on
    private static final boolean TENSOR_CORES =
            Boolean.parseBoolean(System.getProperty("jitllm.qwen35.tensorCores", "true"));

    private static final int MATVEC_LOCAL = 128;
    private static final int ELEMENTWISE_LOCAL = 128;

    /**
     * Lanes per attention workgroup of the non-tensor-core kernels: the staged kernel's lane
     * mapping is written for this width, and the warp kernel for four warps of it.
     */
    private static final int ATTENTION_LOCAL = Qwen35BatchKernels.ATTENTION_STAGE_LANES;

    private final Qwen35State state;
    private final Qwen35TornadoWeights weights;
    private final Qwen35Configuration config;
    private final int batchSize;

    /** The feed-forward of a family built on this one, or {@code null} for the dense SwiGLU. */
    private final Qwen35BatchFeedForward feedForward;

    /** The layers this prefill builds, {@code [firstLayer, endLayer)}: one pipeline stage's. */
    private final int firstLayer;

    private final int endLayer;

    /** Q8_0 projections decoded for cuBLAS, as "graph.task" and element count, for their grids. */
    private final java.util.Map<String, Integer> q8DequantTasks = new java.util.LinkedHashMap<>();

    private final KernelContext context = new KernelContext();

    private final List<ImmutableTaskGraph> graphs;
    private String lastLayerTaskGraphID;

    /**
     * How many prompt rows each task's grid covers per workgroup — one, unless it is tiled.
     *
     * <p>Recorded while the graphs are built, from the kernel's own tile constant, because the
     * worker grid has to match the kernel the dispatch chose — and the choice is per tensor, not
     * per task name. {@code ffn_down} is Q4_1 on this model's first eight blocks and Q4_0 on the
     * rest, so the same name can be tiled differently in different layers; keying on the name alone
     * gave the untiled kernel a tiled grid, and every row past the first read the wrong activation.
     * Taking the number from the kernel that was selected means a tile constant can only ever be
     * changed in one place.
     */
    private final java.util.Map<String, Integer> rowTiles = new java.util.LinkedHashMap<>();

    /**
     * How many <b>output</b> rows each task's grid covers per workgroup — one, unless the kernel
     * also tiles that axis. Recorded the same way and for the same reason as {@link #rowTiles}.
     */
    private final java.util.Map<String, Integer> colTiles = new java.util.LinkedHashMap<>();

    /** Tasks that run on the tensor cores, whose grid is a warp per (16 x 8) output tile. */
    private final java.util.Map<String, Integer> mmaTasks = new java.util.LinkedHashMap<>();

    /**
     * Projections — Q4_0, and the Q5_K ssm_out — that run as a dequantization into the FP16 scratch
     * followed by the tiled FP16 GEMM: the projection task to its output width, and its
     * dequantization task to the matrix's element count. Each pair's grids come from here.
     */
    private final java.util.Map<String, Integer> gemmTasks = new java.util.LinkedHashMap<>();

    private final java.util.Map<String, Integer> dequantTasks = new java.util.LinkedHashMap<>();

    /** The layer prefixes whose up projection writes SwiGLU's FP16 output from its epilogue. */
    private final java.util.Set<String> fusedSwigluLayers = new java.util.HashSet<>();

    /** The layer prefixes that convert SwiGLU's FP32 output to FP16 in a task of their own. */
    private final java.util.Set<String> hbConvertLayers = new java.util.HashSet<>();

    /**
     * The residual passes built after a projection that stores rather than adds, by qualified task
     * name. Not every tensor-core projection fuses its residual: a packed int8 GEMM below the FP16
     * pair's output width stores, so the grid follows the tasks built, not the path taken.
     */
    private final java.util.Set<String> residualPasses = new java.util.LinkedHashSet<>();

    /** The int8 pair's activation quantizations, with their lane counts. */
    private final java.util.Map<String, Integer> quantizeTasks = new java.util.LinkedHashMap<>();

    /** The int8 GEMMs that read their Q8_0 weights directly, with no decode before them. */
    private final java.util.Set<String> q8DirectGemms = new java.util.LinkedHashSet<>();

    /**
     * The F32 projections placed on the warp kernels, with their output counts: positive for a warp
     * per output, negative for a warp per 4 x 4 tile.
     */
    private final java.util.Map<String, Integer> warpMatVecTasks = new java.util.LinkedHashMap<>();

    /** The Q4_0 and Q8_0 projections placed on the Metal SIMD-group GEMMs, with their outputs. */
    private final java.util.Map<String, Integer> simdgroupTasks = new java.util.LinkedHashMap<>();

    /**
     * The smallest output width measured to gain from dequantize-then-GEMM. At 1,024 outputs the
     * GEMM launches too few tiles and the pair lost at both eligible widths; at 5,120 and above it
     * won. Everything below stays on the direct quantized kernel.
     */
    private static final int DEQUANT_GEMM_MIN_OUTPUTS = 5120;

    /** Rows of one FP16 GEMM tile. */
    private static final int GEMM_TILE = 128;

    private static final int GEMM_LOCAL = 256;

    /**
     * Whether a quantized projection of {@code n} outputs over {@code k} inputs takes the
     * dequantize-then-GEMM pair: the state allocated the scratch (which is what says the width
     * fills whole GEMM tiles), the shape divides the GEMM's tiles, the matrix fits the scratch, and
     * the output width is one the pair was measured to gain at. Asked for the Q4_0 projections, the
     * Q4_1 ffn_down and the Q5_K ssm_out alike; the scratch is one buffer that every pair of every
     * layer graph writes and reads in turn, in graph order.
     */
    private boolean dequantGemmEligible(int n, int k) {
        return state.workspace.wrapDequantScratchFP16 != null
                && Qwen35Configuration.dequantGemmWidth(batchSize)
                && n % GEMM_TILE == 0
                && k % 16 == 0
                && n >= DEQUANT_GEMM_MIN_OUTPUTS
                && (long) n * k <= state.workspace.wrapDequantScratchFP16.getSize();
    }

    /**
     * Width from which the Q4_0 key/value projections (kvDim outputs, below the shared output
     * threshold) take the pair: measured on the 1024 x 5120 shape, the complete pair is 44% faster
     * than the direct kernel at 512, 67% at 1024 and 77% at 2048, level at 256 and twice as slow at
     * 128.
     */
    private static final int DEQUANT_GEMM_KV_MIN_WIDTH = 512;

    /**
     * Whether a Q4_0 projection of exactly the key/value width takes the pair: the scratch, the
     * tile divisibility and the fit as {@link #dequantGemmEligible}, but the width rule above in
     * place of the output threshold. Only Q4_0 asks this; the Q4_1 and Q5_K pairs keep the shared
     * rule.
     */
    private boolean kvPairEligible(int n, int k) {
        return n == config.kvDim()
                && batchSize >= DEQUANT_GEMM_KV_MIN_WIDTH
                && state.workspace.wrapDequantScratchFP16 != null
                && Qwen35Configuration.dequantGemmWidth(batchSize)
                && n % GEMM_TILE == 0
                && k % 16 == 0
                && (long) n * k <= state.workspace.wrapDequantScratchFP16.getSize();
    }

    /** What a pair's GEMM does with its accumulators. */
    private enum Epilogue {
        /** Store the product to {@code out}. */
        STORE,
        /** Add the product to {@code out} in place: {@code out += A x B}. */
        RESIDUAL,
        /** {@code hb16 = fp16(silu(gate) * (A x B))}, {@code gate} the FP32 gate projection. */
        SWIGLU
    }

    /**
     * Whether a Q4_0 projection of this shape runs as the dequantize-then-GEMM pair. Callers that
     * want the pair's fused epilogues ask this to know whether a separate residual or SwiGLU task
     * is still needed.
     */
    private boolean q40Pair(int n, int k) {
        return dequantGemmEligible(n, k) || kvPairEligible(n, k);
    }

    /**
     * Test seam: which projections (by task name) may take the int8 path. Production accepts all;
     * the topology tests narrow it to build the producer/consumer mixes that a shape-driven
     * eligibility never produces on the real model, and check the buffers still line up.
     */
    public static volatile java.util.function.Predicate<String> int8TaskFilterForTests =
            task -> true;

    /**
     * Diagnostic seam: the FP16 pair fed activations that were quantized and dequantized the way
     * the int8 path quantizes them (normed chunk and attention output). Never set in production.
     */
    public static volatile boolean fakeQuantizeForTests = false;

    /**
     * Whether a Q4_0 projection of this shape takes the int8 GEMM rather than the FP16 pair: the
     * int8 activations were allocated (the width fills whole tiles), the FP16 pair's own
     * eligibility holds, K is a whole number of 64-k rounds, and an activation the enclosing layer
     * quantized for this K precedes it in task order (the caller's responsibility). Only Q4_0 asks;
     * Q4_1 and Q5_K keep the FP16 pair, and the direct kernels keep every width the pairs do not
     * take.
     */
    private boolean int8Eligible(String task, int n, int k) {
        return int8TaskFilterForTests.test(task) && int8Eligible(n, k);
    }

    private boolean int8Eligible(int n, int k) {
        return TensorCoreSupport.isInt8MmaCapable()
                && state.workspace.wrapQ8ActBatch != null
                && q40Pair(n, k)
                && k % Int8GemmKernels.I8_BK == 0;
    }

    /**
     * Whether a packed Q4_0 projection of {@code n} outputs over {@code k} inputs runs as the
     * packed int8 GEMM: the activation of this K is quantized, and the outputs fill whole tiles. A
     * packed weight has no other prefill kernel, so every packed projection must answer yes here.
     */
    private boolean packedQ4Eligible(String task, int n, int k) {
        return int8TaskFilterForTests.test(task) && int8Quantizes(k) && n % GEMM_TILE == 0;
    }

    /**
     * Whether the Q4_0 projection {@code w} of {@code n} outputs over {@code k} inputs runs as an
     * int8 GEMM, so its input must be quantized rather than converted to FP16. A packed weight
     * takes it below the FP16 pair's output threshold too, since it has no other kernel.
     */
    private boolean q40Int8(String task, TornadoTensor w, int n, int k) {
        return int8Eligible(task, n, k) || (w.isPackedQ4() && packedQ4Eligible(task, n, k));
    }

    /** {@code x += out}: the residual of a projection whose kernel stored rather than added. */
    private void residualPass(TaskGraph layer, int layerIndex, String task) {
        residualPasses.add("batchLayer_" + layerIndex + "." + task);
        layer.task(
                task,
                Qwen35MMAKernels::residualAdd,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapFFNDownBatch);
    }

    /** Whether the int8 pair quantizes activations of {@code k} columns at all. */
    private boolean int8Quantizes(int k) {
        return TENSOR_CORES
                && TensorCoreSupport.isInt8MmaCapable()
                && state.workspace.wrapQ8ActBatch != null
                && Qwen35Configuration.dequantGemmWidth(batchSize)
                && k % Int8GemmKernels.I8_BK == 0;
    }

    /**
     * The task quantizing an FP32 activation of {@code k} columns into the int8 scratch, one lane
     * per element in 32-lane blocks, {@code batchSize * k} lanes. The scratch is one buffer: a
     * quantization is read by the projections between it and the next one, in graph order.
     */
    private void quantizeActivation(TaskGraph graph, int layer, String task, FloatArray x, int k) {
        quantizedInput = x;
        quantizeTasks.put("batchLayer_" + layer + "." + task, batchSize * k);
        graph.task(
                task,
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                x,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                k);
    }

    /**
     * The activation the int8 scratch holds at this point of the layer graph being built: set by
     * every quantization, cleared at the start of each layer. Every buffer quantized is rewritten
     * only by the task that precedes its own quantization (a norm, SwiGLU, the attention or the
     * delta-net readout), so a projection reading it can rely on the scratch until the next one.
     */
    private FloatArray quantizedInput;

    /** The int8 GEMM over {@code w}'s blocks where they lie, the activation quantized per block. */
    private void int8Projection(
            TaskGraph graph,
            String qualified,
            String task,
            TornadoTensor w,
            FloatArray out,
            FloatArray gate,
            FloatArray hb,
            int n,
            int k,
            Epilogue epilogue) {
        int8Projection(graph, qualified, task, w, out, gate, hb, n, k, epilogue, false);
    }

    /** {@link #int8Projection}, the weight Q8_0 rather than Q4_0 when {@code q8} is set. */
    private void int8Projection(
            TaskGraph graph,
            String qualified,
            String task,
            TornadoTensor w,
            FloatArray out,
            FloatArray gate,
            FloatArray hb,
            int n,
            int k,
            Epilogue epilogue,
            boolean q8) {
        // The GEMM reads the Q8_0 or Q4_0 blocks itself, or their packed tiles: no decode task,
        // no weight scratch.
        q8DirectGemms.add(qualified);
        gemmTasks.put(qualified, n);
        int mode =
                switch (epilogue) {
                    case STORE -> Int8GemmKernels.EPILOGUE_STORE;
                    case RESIDUAL -> Int8GemmKernels.EPILOGUE_RESIDUAL;
                    case SWIGLU -> Int8GemmKernels.EPILOGUE_SWIGLU;
                };
        int splits = epilogue == Epilogue.SWIGLU ? 1 : gemmSplits(n, k);
        FloatArray target = epilogue == Epilogue.SWIGLU ? hb : out;
        // Unsplit, the partial-sum parameter is never read; it is bound to the output.
        FloatArray partial = splits > 1 ? state.workspace.wrapQ8SplitPartial : target;
        if (splits > 1) {
            gemmSplitCounts.put(qualified, splits);
        }
        boolean packed = q8 ? w.isPackedQ8() : w.isPackedQ4();
        graph.task(
                task,
                packed
                        ? (q8
                                ? Int8GemmKernels::gemmInt8Q8_0Packed
                                : Int8GemmKernels::gemmInt8Q4_0Packed)
                        : (q8 ? Int8GemmKernels::gemmInt8Q8_0 : Int8GemmKernels::gemmInt8Q4_0),
                context,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                w.asByteArray(),
                target,
                epilogue == Epilogue.SWIGLU ? gate : out,
                batchSize,
                n,
                k,
                mode,
                partial,
                splits,
                state.workspace.batchStartPosHolder,
                n,
                0);
        if (splits > 1) {
            // The splits' partial sums added in order, then stored or added to the residual.
            splitReduceTasks.put(qualified + "_reduce", batchSize * n);
            graph.task(
                    task + "_reduce",
                    Int8GemmKernels::reduceSplitsQ8_0,
                    context,
                    partial,
                    out,
                    batchSize * n,
                    splits,
                    mode,
                    state.workspace.batchStartPosHolder,
                    n);
        }
    }

    /**
     * A Q4_0 projection on the tensor cores: {@code out[batch][n] = a[batch][k] x w[n][k]}. The
     * dequantize-then-GEMM pair where {@link #dequantGemmEligible} says so, otherwise the direct
     * quantized kernel; either way the task named {@code task} is the one that writes {@code out}.
     *
     * <p>Every pair is the tiled one: the decoder writes the scratch in the GEMM's B-tile order,
     * both nibbles of a packed byte from one lane, and the GEMM copies each tile global-to-shared
     * as contiguous words. The Q4_1 ffn_down and the Q5_K ssm_out take the same GEMM with their own
     * paired-nibble decoders; the scratch carries no layout of its own, each pair's two tasks agree
     * between themselves, and no task reads what another family's decoder wrote.
     */
    private void q40Projection(
            TaskGraph graph,
            String qualified,
            String task,
            HalfFloatArray aFP16,
            TornadoTensor w,
            FloatArray out,
            int n,
            int k) {
        q40Projection(graph, qualified, task, aFP16, w, out, null, null, n, k, Epilogue.STORE);
    }

    /**
     * A Q4_0 projection on the tensor cores with the pair's epilogue chosen: {@code STORE} writes
     * {@code out}; {@code RESIDUAL} adds into {@code out}; {@code SWIGLU} writes {@code hb16} from
     * {@code gate} and the product. The direct kernel only stores, so a caller asking for a fused
     * epilogue on a shape the pair does not take gets a store and adds its own task (see {@link
     * #q40Pair}).
     */
    private void q40Projection(
            TaskGraph graph,
            String qualified,
            String task,
            HalfFloatArray aFP16,
            TornadoTensor w,
            FloatArray out,
            FloatArray gate,
            HalfFloatArray hb16,
            int n,
            int k,
            Epilogue epilogue) {
        if (w.isPackedQ4()) {
            if (!packedQ4Eligible(task, n, k) || epilogue == Epilogue.SWIGLU) {
                throw PackedWeights.noPackedKernel("qwen35 batch-prefill task '" + qualified + "'");
            }
            int8Projection(graph, qualified, task, w, out, null, null, n, k, epilogue);
            return;
        }
        if (int8Eligible(task, n, k) && epilogue != Epilogue.SWIGLU) {
            int8Projection(graph, qualified, task, w, out, null, null, n, k, epilogue);
            return;
        }
        if (q40Pair(n, k)) {
            // One lane per packed byte: both nibbles decoded, two halves written.
            dequantTasks.put(qualified + "_dequant", n * k / 2);
            gemmTasks.put(qualified, n);
            graph.task(
                    task + "_dequant",
                    Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                    context,
                    w.asByteArray(),
                    state.workspace.wrapDequantScratchFP16,
                    n,
                    k);
            pairGemm(graph, task, aFP16, out, gate, hb16, n, k, epilogue);
            return;
        }
        mmaTasks.put(qualified, n);
        graph.task(
                task,
                Qwen35MMAKernels::projectionMMAQ4_0Prefetch,
                context,
                aFP16,
                w.asByteArray(),
                out,
                batchSize,
                n,
                k);
    }

    /** The pair's GEMM over the shared scratch, with the chosen epilogue. */
    private void pairGemm(
            TaskGraph graph,
            String task,
            HalfFloatArray aFP16,
            FloatArray out,
            FloatArray gate,
            HalfFloatArray hb16,
            int n,
            int k,
            Epilogue epilogue) {
        switch (epilogue) {
            case STORE ->
                    graph.task(
                            task,
                            Qwen35MMAKernels::gemmMMATiledB,
                            context,
                            aFP16,
                            state.workspace.wrapDequantScratchFP16,
                            out,
                            batchSize,
                            n,
                            k);
            case RESIDUAL ->
                    graph.task(
                            task,
                            Qwen35MMAKernels::gemmMMATiledBResidual,
                            context,
                            aFP16,
                            state.workspace.wrapDequantScratchFP16,
                            out,
                            batchSize,
                            n,
                            k);
            case SWIGLU ->
                    graph.task(
                            task,
                            Qwen35MMAKernels::gemmMMATiledBSwiGLU,
                            context,
                            aFP16,
                            state.workspace.wrapDequantScratchFP16,
                            gate,
                            hb16,
                            batchSize,
                            n,
                            k);
        }
    }

    /**
     * Whether a Q4_0 projection over this shape can run on the tensor cores.
     *
     * <p>The chunk has to fill whole 16-row MMA tiles, because the store writes a whole tile and
     * the output buffer is not padded; the reduction dimension has to be a whole number of Q4_0
     * blocks; and the output has to be a whole number of the 8-column tile.
     */
    private boolean mmaEligible(int n, int d) {
        return TENSOR_CORES
                && TensorCoreSupport.isTensorCoreCapableBackend()
                && batchSize % Qwen35MMAKernels.BM == 0
                && n % 32 == 0
                && d % Qwen35MMAKernels.BN == 0;
    }

    public Qwen35BatchPrefillLayers(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            int batchSize) {
        this(state, weights, config, batchSize, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * The first layer of the range takes the role layer 0 has otherwise; the key/value and
     * recurrent state keep their whole-model layout and absolute indices.
     */
    public Qwen35BatchPrefillLayers(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        this(state, weights, config, batchSize, firstLayer, endLayer, null);
    }

    /**
     * The layers {@code [firstLayer, endLayer)}, their feed-forward {@code feedForward} in place of
     * the dense SwiGLU when it is not null: a family built on this one.
     */
    public Qwen35BatchPrefillLayers(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            int batchSize,
            int firstLayer,
            int endLayer,
            Qwen35BatchFeedForward feedForward) {
        if (firstLayer < 0 || endLayer > config.numberOfLayers() || firstLayer >= endLayer) {
            throw new IllegalArgumentException(
                    "layer range ["
                            + firstLayer
                            + ", "
                            + endLayer
                            + ") outside the model's "
                            + config.numberOfLayers()
                            + " layers");
        }
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        allocateQ8Scratch();
        if (feedForward != null && !int8Quantizes(config.dim())) {
            throw new UnsupportedOperationException(
                    "this feed-forward runs on the int8 tensor cores, at widths that are a"
                            + " multiple of 128; this width is "
                            + batchSize);
        }
        this.feedForward = feedForward;
        this.graphs =
                IntStream.range(firstLayer, endLayer)
                        .mapToObj(this::buildLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
        this.lastLayerTaskGraphID = "batchLayer_" + (endLayer - 1);
    }

    // @formatter:off
    /**
     * The scratch buffers this range's Q8_0 projections need: the partial sums of the int8 GEMMs
     * split along K ({@link #gemmSplits}), sized for the largest; and, when native libraries are
     * on, the FP32 scratch for the projections that do not tile for the int8 GEMM, sized for the
     * largest of them. Every projection uses each in turn, so one of each is enough. Without native
     * libraries the untiled projections take the warp kernel and need no scratch.
     */
    // @formatter:on
    private void allocateQ8Scratch() {
        boolean nativeLibraries =
                org.beehive.jitllm.backend.tornado.NativePrefillSupport.nativeProjections(
                        state.executionPolicy());
        int dim = config.dim();
        // Each projection's tensors with its (outputs, inputs).
        Object[][] projections = {
            {weights.wqLayered, config.queryGateDim(), dim},
            {weights.wkLayered, config.kvDim(), dim},
            {weights.wvLayered, config.kvDim(), dim},
            {weights.woLayered, dim, config.attentionOutputInputDim()},
            {weights.w1Layered, config.hiddenDim(), dim},
            {weights.w3Layered, config.hiddenDim(), dim},
            {weights.w2Layered, dim, config.hiddenDim()},
            {weights.ssmQkv, config.deltaNetConvDim(), dim},
            {weights.ssmGate, config.deltaNetValueDim(), dim},
            {weights.ssmAlpha, config.numberOfValueHeads(), dim},
            {weights.ssmBeta, config.numberOfValueHeads(), dim},
            {weights.ssmOut, dim, config.deltaNetValueDim()}
        };
        long widest = 0;
        long partials = 0;
        for (Object[] projection : projections) {
            TornadoTensor[] tensors = (TornadoTensor[]) projection[0];
            int outputs = (Integer) projection[1];
            int inputs = (Integer) projection[2];
            for (int layer = firstLayer; layer < endLayer; layer++) {
                if (tensors != null
                        && layer < tensors.length
                        && tensors[layer] != null
                        && (tensors[layer].dataType() == DataType.Q8_0
                                || tensors[layer].dataType() == DataType.Q4_0)) {
                    boolean eligible =
                            tensors[layer].dataType() == DataType.Q8_0
                                    ? q8Int8Eligible("", outputs, inputs)
                                    : int8Eligible(outputs, inputs)
                                            || (tensors[layer].isPackedQ4()
                                                    && packedQ4Eligible("", outputs, inputs));
                    if (eligible) {
                        int splits = gemmSplits(outputs, inputs);
                        if (splits > 1) {
                            partials = Math.max(partials, (long) splits * batchSize * outputs);
                        }
                    } else if (nativeLibraries) {
                        widest = Math.max(widest, (long) outputs * inputs);
                    }
                }
            }
        }
        if (widest > 0) {
            FloatArray current = state.workspace.wrapDequantScratchF32;
            if (current == null || current.getSize() < widest) {
                state.workspace.wrapDequantScratchF32 =
                        org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces.floats(
                                Math.toIntExact(widest));
            }
        }
        if (partials > 0) {
            FloatArray current = state.workspace.wrapQ8SplitPartial;
            if (current == null || current.getSize() < partials) {
                state.workspace.wrapQ8SplitPartial =
                        org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces.floats(
                                Math.toIntExact(partials));
            }
        }
    }

    /** The streaming multiprocessors of the device this plan is built for; 0 when unknown. */
    private static int smCount() {
        return org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
    }

    private static final int MAX_GEMM_SPLITS = 8;

    // @formatter:off
    /**
     * How many parts the int8 GEMM of a Q8_0 projection of {@code outputs} over {@code k} inputs
     * splits K into, so that its 128 x 128 tiles fill the device's multiprocessors.
     *
     * <p>A tile is one block, and a block takes a multiprocessor to itself (its registers allow no
     * second one), so a GEMM runs in waves of {@link #smCount} tiles and its time is the number of
     * waves times the rounds each tile runs. On the A10's 72 multiprocessors the 5,120-wide
     * projections have 80 tiles at a 256-row chunk: two waves, the second eight tiles long. Split
     * in {@code s}, a GEMM has {@code s} times the tiles, each running {@code 1/s} of the rounds.
     * The split chosen is the one with the fewest (waves x rounds), taken only where it saves at
     * least a tenth, since every split adds a pass over its partial sums.
     */
    // @formatter:on
    private int gemmSplits(int outputs, int k) {
        int sms = smCount();
        if (sms <= 0) {
            return 1;
        }
        int tiles = (batchSize / GEMM_TILE) * (outputs / GEMM_TILE);
        int rounds = k / Int8GemmKernels.I8_BK;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= MAX_GEMM_SPLITS && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return bestCost * 10 <= unsplit * 9 ? best : 1;
    }

    /** The K-split int8 GEMMs, by qualified task name, with their split counts. */
    private final java.util.Map<String, Integer> gemmSplitCounts = new java.util.LinkedHashMap<>();

    /** The passes that add a split GEMM's partial sums, with their lane counts. */
    private final java.util.Map<String, Integer> splitReduceTasks = new java.util.LinkedHashMap<>();

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    // ── validation and dispatch ───────────────────────────────────────────────

    /**
     * Whether a Q4_0 or Q8_0 projection over {@code n} inputs runs on the Metal SIMD-group GEMMs,
     * which stage the contraction in 64-wide slices; see {@link
     * BatchPrefillGemmPolicy#simdgroup()}.
     */
    private static boolean simdgroupGemm(int n) {
        return BatchPrefillGemmPolicy.simdgroup() && n % 64 == 0;
    }

    /** Whether this state's key/value store is half precision. */
    private boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    /** The key store the graphs bind, whichever precision it is in. */
    private Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    private TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "qwen35 block "
                            + layer
                            + " is a "
                            + (config.isRecurrentLayer(layer) ? "recurrent" : "attention")
                            + " layer and must carry "
                            + role
                            + ", which this model does not hold for it");
        }
        return tensor;
    }

    /**
     * {@code out[b] = w · x[b]}, or {@code +=}, by the representation {@code w} is in.
     *
     * <p>The batched counterpart of the single-token dispatch, and refused the same way: a
     * representation with no batched kernel is named rather than converted.
     */
    private void matVecBatch(
            TaskGraph graph,
            int layer,
            String task,
            String role,
            TornadoTensor w,
            FloatArray xBatch,
            FloatArray outBatch,
            int n,
            int d,
            boolean residual) {
        requireWholeBlocks(layer, task, role, w.dataType(), n);
        if (w.dataType() == DataType.Q8_0) {
            q8Projection(graph, layer, task, w, xBatch, outBatch, n, d, residual);
            return;
        }
        switch (w.dataType()) {
            case F32 -> {
                if (residual) {
                    throw unsupported(layer, task, role, w.dataType(), "an accumulating");
                }
                // One warp per output where the warp shuffle is verified (CUDA, the tensor-core
                // guard): the same partial sums and the same reduction tree as the 128-lane
                // workgroup kernel, so the outputs are bit-identical; elsewhere the workgroup one.
                if (TensorCoreSupport.isTensorCoreCapableBackend()) {
                    // A warp per 4 x 4 tile of (row, output) pairs where the outputs divide into
                    // tiles — the same per-pair arithmetic, each load serving four products.
                    boolean tiled = d % TransformerBatchPrefillKernels.MATVEC_TILE == 0;
                    warpMatVecTasks.put("batchLayer_" + layer + "." + task, tiled ? -d : d);
                    graph.task(
                            task,
                            tiled
                                    ? TransformerBatchPrefillKernels::batchedMatVecF32WarpTile
                                    : TransformerBatchPrefillKernels::batchedMatVecF32Warp,
                            context,
                            xBatch,
                            outBatch,
                            w.asFloatArray(),
                            n,
                            d,
                            batchSize);
                } else {
                    graph.task(
                            task,
                            TransformerBatchPrefillKernels::batchedMatVecF32,
                            context,
                            xBatch,
                            outBatch,
                            w.asFloatArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            case Q4_0 -> {
                if (w.isPackedQ4()) {
                    // A packed weight has only the packed int8 GEMM: the activation quantized here.
                    String qualified = "batchLayer_" + layer + "." + task;
                    if (!packedQ4Eligible(task, d, n)) {
                        throw PackedWeights.noPackedKernel(
                                "qwen35 batch-prefill task '" + qualified + "'");
                    }
                    quantizeInput(graph, layer, task, xBatch, n);
                    int8Projection(
                            graph,
                            qualified,
                            task,
                            w,
                            outBatch,
                            null,
                            null,
                            d,
                            n,
                            residual ? Epilogue.RESIDUAL : Epilogue.STORE);
                    return;
                }
                if (simdgroupGemm(n)) {
                    // Metal: 64-token x 64-output tiles on SIMD-group matrices, the weights
                    // dequantized to float as they are staged.
                    simdgroupTasks.put("batchLayer_" + layer + "." + task, d);
                    graph.task(
                            task,
                            residual
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmQ4_0WithResidual
                                    : TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize);
                    return;
                }
                // The normed chunk is also staged as FP16 right after the norm, so a projection
                // reading it can run on the tensor cores rather than as a scalar matrix-vector.
                // Anything else — a residual form, another input, a shape the MMA tiles do not
                // divide — takes the tiled scalar kernel below.
                if (!residual && xBatch == state.workspace.wrapNormedBatch && mmaEligible(n, d)) {
                    q40Projection(
                            graph,
                            "batchLayer_" + layer + "." + task,
                            task,
                            state.workspace.wrapNormedFP16Batch,
                            w,
                            outBatch,
                            d,
                            n);
                    return;
                }
                // Tiled: one workgroup per (tile of rows, output row), decoding each weight once
                // for the tile. A quantized projection is memory-bound, and a chunk is only worth
                // scheduling if it reuses the weights it reads.
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_0.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_0.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorTiledBatchWithResidualQ4_0,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorTiledBatchQ4_0,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            case Q4_1 -> {
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_1.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_1.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorTiledBatchWithResidualQ4_1,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorTiledBatchQ4_1,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            case Q5_K -> {
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ5_K.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ5_K.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorTiledBatchWithResidualQ5_K,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorTiledBatchQ5_K,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            default -> throw unsupported(layer, task, role, w.dataType(), "a batched");
        }
    }

    /** Layers whose SwiGLU follows two separate Q8_0 projections, for its grid. */
    private final java.util.Set<String> q8SwigluLayers = new java.util.LinkedHashSet<>();

    /**
     * Whether a Q8_0 projection of {@code outputs} over {@code k} inputs runs on the int8 GEMM: the
     * input can be quantized at this width and K, and the output divides into 128-column tiles.
     */
    private boolean q8Int8Eligible(String task, int outputs, int k) {
        return int8TaskFilterForTests.test(task) && int8Quantizes(k) && outputs % GEMM_TILE == 0;
    }

    /** Quantizes {@code x} into the int8 scratch unless the scratch already holds it. */
    private void quantizeInput(TaskGraph graph, int layer, String task, FloatArray x, int k) {
        if (quantizedInput != x) {
            quantizeActivation(graph, layer, task.replace("_proj", "") + "_q8", x, k);
        }
    }

    // @formatter:off
    /**
     * A Q8_0 projection of the whole chunk, {@code out[rows][d] (+)= x[rows][n] · Wᵀ} with W the
     * {@code [d][n]} row-major weight. Three ways, in order of preference:
     *
     * <ul>
     *   <li>the int8 GEMM: the activation quantized per 32-block, and the block-scaled int8
     *       tensor-core GEMM reading the Q8_0 blocks where they lie — the arithmetic of llama.cpp's
     *       MMQ, with the residual added from the accumulators;
     *   <li>on Metal, the SIMD-group GEMM: 64-token x 64-output tiles on {@code simdgroup_float8x8}
     *       matrices, the Q8_0 blocks dequantized to float as they are staged — the residual form
     *       adds into {@code out}, and the stored form is the Q/K/V kernel with its key/value width
     *       zero, so every tile lies in its first output (which needs {@code d % 64 == 0});
     *   <li>where the shape does not tile, and native libraries are on: the weight decoded to FP32
     *       and one cuBLAS TF32 GEMM ({@code C = op_T(W) · X} in cuBLAS's column-major terms);
     *   <li>otherwise a warp per output.
     * </ul>
     */
    // @formatter:on
    private void q8Projection(
            TaskGraph graph,
            int layer,
            String task,
            TornadoTensor w,
            FloatArray xBatch,
            FloatArray outBatch,
            int n,
            int d,
            boolean residual) {
        String qualified = "batchLayer_" + layer + "." + task;
        if (q8Int8Eligible(task, d, n)) {
            quantizeInput(graph, layer, task, xBatch, n);
            int8Projection(
                    graph,
                    qualified,
                    task,
                    w,
                    outBatch,
                    null,
                    null,
                    d,
                    n,
                    residual ? Epilogue.RESIDUAL : Epilogue.STORE,
                    true);
            return;
        }
        if (w.isPackedQ8()) {
            throw PackedWeights.noPackedKernel(
                    "qwen35 batch-prefill layer "
                            + layer
                            + " task '"
                            + task
                            + "' (shape not tiled by the int8 GEMM)");
        }
        if (simdgroupGemm(n) && (residual || d % 64 == 0)) {
            simdgroupTasks.put(qualified, d);
            ByteArray weight = w.asByteArray();
            if (residual) {
                graph.task(
                        task,
                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                        context,
                        xBatch,
                        outBatch,
                        weight,
                        n,
                        d,
                        batchSize);
            } else {
                graph.task(
                        task,
                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
                        context,
                        xBatch,
                        outBatch,
                        outBatch,
                        outBatch,
                        weight,
                        weight,
                        weight,
                        n,
                        d,
                        0,
                        batchSize);
            }
            return;
        }
        FloatArray scratch = state.workspace.wrapDequantScratchF32;
        if (scratch == null) {
            warpMatVecTasks.put(qualified, d);
            graph.task(
                    task,
                    TransformerBatchPrefillKernels::batchedMatVecQ8_0Warp,
                    context,
                    xBatch,
                    outBatch,
                    w.asByteArray(),
                    n,
                    d,
                    batchSize,
                    residual ? 1 : 0);
            return;
        }
        q8DequantTasks.put(qualified + "_dequant", n * d);
        graph.task(
                task + "_dequant",
                org.beehive.jitllm.backend.tornado.kernels.NativeProjectionKernels
                        ::dequantizeQ8_0ToFP32,
                context,
                w.asByteArray(),
                scratch,
                n * d);
        graph.libraryTask(
                task,
                uk.ac.manchester.tornado.cublas.CuBlas::cublasSgemmTF32,
                1,
                0,
                d,
                batchSize,
                n,
                1.0f,
                scratch,
                n,
                xBatch,
                n,
                residual ? 1.0f : 0.0f,
                outBatch,
                d);
    }

    private UnsupportedOperationException unsupported(
            int layer, String task, String role, DataType type, String article) {
        return new UnsupportedOperationException(
                "qwen35 batch-prefill layer "
                        + layer
                        + " task '"
                        + task
                        + "' reads "
                        + role
                        + " as "
                        + type
                        + ", for which this backend has no "
                        + article
                        + " matrix-vector kernel. It is not converted to another representation to"
                        + " get one.");
    }

    private void requireWholeBlocks(int layer, String task, String role, DataType type, int n) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1, Q8_0 -> 32;
                    case Q4_K, Q5_K, Q6_K -> 256;
                    default -> 1;
                };
        if (n % blockSize != 0) {
            throw new UnsupportedOperationException(
                    "qwen35 batch-prefill layer "
                            + layer
                            + " task '"
                            + task
                            + "' reads "
                            + role
                            + " as "
                            + type
                            + " with a row of "
                            + n
                            + " weights, which is not a whole number of "
                            + blockSize
                            + "-weight blocks.");
        }
    }

    /**
     * @param hbFP16 whether the down projection reads the SwiGLU output as FP16 (it is on the
     *     tensor cores): then SwiGLU writes the FP16 buffer directly and no conversion follows
     */
    private void fusedGateUpBatch(
            TaskGraph graph,
            int layer,
            TornadoTensor gate,
            TornadoTensor up,
            FloatArray xBatch,
            boolean hbFP16) {
        FusedOperandSupport.requireUniform(
                "qwen35 batch-prefill layer " + layer + " fused gate/up feed-forward",
                List.of("ffn_gate", "ffn_up"),
                gate,
                up);
        if (gate.dataType() == DataType.Q8_0) {
            if (q8Int8Eligible("ffn_up_proj", config.hiddenDim(), config.dim())) {
                // The int8 GEMM, as for Q4_0: the gate stored, then the up GEMM writing
                // silu(gate) * up from its accumulators, no SwiGLU task.
                quantizeInput(graph, layer, "ffn_gate_up", xBatch, config.dim());
                int8Projection(
                        graph,
                        "batchLayer_" + layer + ".ffn_gate_proj",
                        "ffn_gate_proj",
                        gate,
                        state.workspace.wrapGateBatch,
                        null,
                        null,
                        config.hiddenDim(),
                        config.dim(),
                        Epilogue.STORE,
                        true);
                int8Projection(
                        graph,
                        "batchLayer_" + layer + ".ffn_up_proj",
                        "ffn_up_proj",
                        up,
                        null,
                        state.workspace.wrapGateBatch,
                        state.workspace.wrapHbBatch,
                        config.hiddenDim(),
                        config.dim(),
                        Epilogue.SWIGLU,
                        true);
                fusedSwigluLayers.add("batchLayer_" + layer + ".");
                return;
            }
            // Gate and up as two projections into their own buffers, then SwiGLU.
            q8Projection(
                    graph,
                    layer,
                    "ffn_gate_proj",
                    gate,
                    xBatch,
                    state.workspace.wrapGateBatch,
                    config.dim(),
                    config.hiddenDim(),
                    false);
            q8Projection(
                    graph,
                    layer,
                    "ffn_up_proj",
                    up,
                    xBatch,
                    state.workspace.wrapUpBatch,
                    config.dim(),
                    config.hiddenDim(),
                    false);
            q8SwigluLayers.add("batchLayer_" + layer + ".");
            graph.task(
                    "ffn_swiglu",
                    Qwen35MMAKernels::swiGLUBatch,
                    context,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbBatch);
            return;
        }
        if (gate.dataType() != DataType.Q4_0) {
            throw unsupported(
                    layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), "a batched");
        }
        if (mmaEligible(config.dim(), config.hiddenDim())) {
            // Two single-panel projections rather than one two-panel kernel. Same M, N and K, the
            // same FP16 chunk, the same two destination buffers and the same SwiGLU task after
            // them; each call stages the activation tile for its own panel, which the fused form
            // staged once. The kernel is the one every other projection already uses.
            q40Projection(
                    graph,
                    "batchLayer_" + layer + ".ffn_gate_proj",
                    "ffn_gate_proj",
                    state.workspace.wrapNormedFP16Batch,
                    gate,
                    state.workspace.wrapGateBatch,
                    config.hiddenDim(),
                    config.dim());
            if (q40Int8("ffn_up_proj", up, config.hiddenDim(), config.dim())) {
                // The int8 pair: the up GEMM writes silu(gate) * up in FP32 from its
                // accumulators; the down projection quantizes that next, or converts it to FP16
                // where it stays on the FP16 pair (the Q4_1 layers).
                int8Projection(
                        graph,
                        "batchLayer_" + layer + ".ffn_up_proj",
                        "ffn_up_proj",
                        up,
                        null,
                        state.workspace.wrapGateBatch,
                        state.workspace.wrapHbBatch,
                        config.hiddenDim(),
                        config.dim(),
                        Epilogue.SWIGLU);
                fusedSwigluLayers.add("batchLayer_" + layer + ".");
                return;
            }
            if (hbFP16 && q40Pair(config.hiddenDim(), config.dim())) {
                // The up GEMM writes silu(gate) * up as FP16 from its accumulators: no up matrix
                // stored, no SwiGLU task.
                q40Projection(
                        graph,
                        "batchLayer_" + layer + ".ffn_up_proj",
                        "ffn_up_proj",
                        state.workspace.wrapNormedFP16Batch,
                        up,
                        null,
                        state.workspace.wrapGateBatch,
                        state.workspace.wrapHbFP16BatchMMA,
                        config.hiddenDim(),
                        config.dim(),
                        Epilogue.SWIGLU);
                fusedSwigluLayers.add("batchLayer_" + layer + ".");
                return;
            }
            q40Projection(
                    graph,
                    "batchLayer_" + layer + ".ffn_up_proj",
                    "ffn_up_proj",
                    state.workspace.wrapNormedFP16Batch,
                    up,
                    state.workspace.wrapUpBatch,
                    config.hiddenDim(),
                    config.dim());
            if (hbFP16) {
                graph.task(
                        "ffn_swiglu",
                        Qwen35MMAKernels::swiGLUBatchFP16,
                        context,
                        state.workspace.wrapGateBatch,
                        state.workspace.wrapUpBatch,
                        state.workspace.wrapHbFP16BatchMMA);
            } else {
                graph.task(
                        "ffn_swiglu",
                        Qwen35MMAKernels::swiGLUBatch,
                        context,
                        state.workspace.wrapGateBatch,
                        state.workspace.wrapUpBatch,
                        state.workspace.wrapHbBatch);
            }
            return;
        }
        if (gate.isPackedQ4() || up.isPackedQ4()) {
            throw PackedWeights.noPackedKernel(
                    "qwen35 batch-prefill layer " + layer + " fused gate/up");
        }
        if (simdgroupGemm(config.dim())) {
            // Metal: gate and up as two SIMD-group GEMMs, then SwiGLU, as on the tensor cores.
            matVecBatch(
                    graph,
                    layer,
                    "ffn_gate_proj",
                    "ffn_gate",
                    gate,
                    xBatch,
                    state.workspace.wrapGateBatch,
                    config.dim(),
                    config.hiddenDim(),
                    false);
            matVecBatch(
                    graph,
                    layer,
                    "ffn_up_proj",
                    "ffn_up",
                    up,
                    xBatch,
                    state.workspace.wrapUpBatch,
                    config.dim(),
                    config.hiddenDim(),
                    false);
            graph.task(
                    "ffn_swiglu",
                    Qwen35MMAKernels::swiGLUBatch,
                    context,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbBatch);
            return;
        }
        rowTiles.put(
                "batchLayer_" + layer + ".ffn_gate_up", TransformerComputeKernelsQ4_0.ffnRowTile());
        colTiles.put(
                "batchLayer_" + layer + ".ffn_gate_up", TransformerComputeKernelsQ4_0.ffnColTile());
        graph.task(
                "ffn_gate_up",
                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUTiledBatchQ4_0,
                context,
                xBatch,
                state.workspace.wrapHbBatch,
                gate.asByteArray(),
                up.asByteArray(),
                config.dim(),
                config.hiddenDim(),
                batchSize,
                MATVEC_LOCAL);
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    private TaskGraph buildLayer(int layerIndex) {
        TaskGraph layer = new TaskGraph("batchLayer_" + layerIndex);
        quantizedInput = null;

        String predecessor =
                layerIndex == firstLayer ? "prefillActivation" : "batchLayer_" + (layerIndex - 1);
        layer.consumeFromDevice(predecessor, state.workspace.wrapXBatch);
        configureTransfers(layer, layerIndex, predecessor);
        transferLayerWeights(layer, layerIndex);

        normalize(
                layer,
                layerIndex,
                "attn_rms_reduce",
                "attn_rms_apply",
                state.workspace.attnScaleBatch,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));

        if (config.isRecurrentLayer(layerIndex)) {
            deltaNetBranch(layer, layerIndex);
        } else {
            attentionBranch(layer, layerIndex);
        }

        normalize(
                layer,
                layerIndex,
                "ffn_rms_reduce",
                "ffn_rms_apply",
                state.workspace.ffnScaleBatch,
                require(weights.rms_ffn_weightLayered, layerIndex, "post_attention_norm"));

        if (feedForward != null) {
            feedForward.appendTasks(layer, context, layerIndex);
        } else {
            denseFeedForward(layer, layerIndex);
        }

        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                keyStore(),
                valueStore(),
                state.workspace.wrapBlockTable,
                state.workspace.wrapConvState,
                state.workspace.wrapDeltaState);
        return layer;
    }

    /** The dense feed-forward of one layer, from the normalized chunk into {@code wrapXBatch}. */
    private void denseFeedForward(TaskGraph layer, int layerIndex) {
        TornadoTensor down = require(weights.w2Layered, layerIndex, "ffn_down");
        // Both representations this family's ffn_down comes in. The Q4_1 kernel below was written
        // and tested with the Q4_0 one, and then never reached: this condition asked for Q4_0 and
        // the choice of kernel underneath it asked whether the tensor was Q4_1, so the first eight
        // blocks -- the Q4_1 ones -- fell through to the scalar path, which is what the profile
        // showed still running.
        boolean downOnTensorCores =
                (down.dataType() == DataType.Q4_0 || down.dataType() == DataType.Q4_1)
                        && mmaEligible(config.hiddenDim(), config.dim());
        // With the gate/up projections on the tensor cores and the down projection reading FP16,
        // SwiGLU writes the FP16 buffer itself and the conversion task below is not built.
        TornadoTensor up = require(weights.w3Layered, layerIndex, "ffn_up");
        boolean gateUpInt8 = q40Int8("ffn_up_proj", up, config.hiddenDim(), config.dim());
        boolean downInt8 =
                downOnTensorCores
                        && down.dataType() == DataType.Q4_0
                        && q40Int8("ffn_down_proj", down, config.dim(), config.hiddenDim());
        // The int8 down projection quantizes SwiGLU's FP32 output, so SwiGLU must write FP32
        // whenever the down projection is int8, whatever the gate/up pair did.
        boolean swigluWritesFP16 =
                downOnTensorCores
                        && mmaEligible(config.dim(), config.hiddenDim())
                        && !gateUpInt8
                        && !downInt8;
        fusedGateUpBatch(
                layer,
                layerIndex,
                require(weights.w1Layered, layerIndex, "ffn_gate"),
                up,
                state.workspace.wrapNormedBatch,
                swigluWritesFP16);
        if (downOnTensorCores) {
            // The pair's GEMM adds the product into the residual stream from its accumulators;
            // the direct kernel's store overwrites, so there the residual is a pass of its own.
            // The input is SwiGLU's output rather than a normed chunk, so that is converted here
            // too, unless SwiGLU wrote it as FP16 already.
            boolean downResidualFused = false;
            if (downInt8) {
                quantizeActivation(
                        layer,
                        layerIndex,
                        "ffn_down_q8",
                        state.workspace.wrapHbBatch,
                        config.hiddenDim());
            } else if (!swigluWritesFP16) {
                hbConvertLayers.add("batchLayer_" + layerIndex + ".");
                layer.task(
                        "ffn_down_fp16",
                        Qwen35MMAKernels::convertToFP16,
                        context,
                        state.workspace.wrapHbBatch,
                        state.workspace.wrapHbFP16BatchMMA);
            }
            if (down.dataType() == DataType.Q4_1) {
                if (dequantGemmEligible(config.dim(), config.hiddenDim())) {
                    // The same tiled pair as the Q4_0 projections, through the same scratch,
                    // with the Q4_1 paired-nibble decoder; this matrix is the scratch's full
                    // size. One lane per packed byte.
                    String qualified = "batchLayer_" + layerIndex + ".ffn_down_proj";
                    dequantTasks.put(qualified + "_dequant", config.dim() * config.hiddenDim() / 2);
                    gemmTasks.put(qualified, config.dim());
                    layer.task(
                            "ffn_down_proj_dequant",
                            Qwen35MMAKernels::dequantizeQ4_1ToFP16TiledPairs,
                            context,
                            down.asByteArray(),
                            state.workspace.wrapDequantScratchFP16,
                            config.dim(),
                            config.hiddenDim());
                    pairGemm(
                            layer,
                            "ffn_down_proj",
                            state.workspace.wrapHbFP16BatchMMA,
                            state.workspace.wrapXBatch,
                            null,
                            null,
                            config.dim(),
                            config.hiddenDim(),
                            Epilogue.RESIDUAL);
                    downResidualFused = true;
                } else {
                    mmaTasks.put("batchLayer_" + layerIndex + ".ffn_down_proj", config.dim());
                    layer.task(
                            "ffn_down_proj",
                            Qwen35MMAKernels::projectionMMAQ4_1,
                            context,
                            state.workspace.wrapHbFP16BatchMMA,
                            down.asByteArray(),
                            state.workspace.wrapFFNDownBatch,
                            batchSize,
                            config.dim(),
                            config.hiddenDim());
                }
            } else {
                downResidualFused = q40Pair(config.dim(), config.hiddenDim());
                q40Projection(
                        layer,
                        "batchLayer_" + layerIndex + ".ffn_down_proj",
                        "ffn_down_proj",
                        state.workspace.wrapHbFP16BatchMMA,
                        down,
                        downResidualFused
                                ? state.workspace.wrapXBatch
                                : state.workspace.wrapFFNDownBatch,
                        null,
                        null,
                        config.dim(),
                        config.hiddenDim(),
                        downResidualFused ? Epilogue.RESIDUAL : Epilogue.STORE);
            }
            if (!downResidualFused) {
                // The direct kernel's store overwrites, so the residual is a pass of its own.
                residualPass(layer, layerIndex, "ffn_down_residual");
            }
        } else {
            matVecBatch(
                    layer,
                    layerIndex,
                    "ffn_down_proj",
                    "ffn_down",
                    down,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    config.hiddenDim(),
                    config.dim(),
                    true);
        }
    }

    /** The largest workgroup a CUDA device schedules; a head wider than it keeps one lane. */
    private static final int MAX_GROUP = 1024;

    /**
     * Whether a per-head norm over {@code headDim} runs as a workgroup per (row, head) with the
     * exact one-lane arithmetic ({@code Qwen35BatchKernels.*BatchGroup}): the head has to be a
     * workgroup's width or less. The one-lane kernels stay for anything wider.
     */
    private static boolean groupNormEligible(int headDim) {
        return headDim >= 1 && headDim <= MAX_GROUP;
    }

    /**
     * Whether the row RMS reduction runs as a workgroup per row with the exact one-lane fold: the
     * row has to be whole strides of the group and fit the staged shared row.
     */
    private static boolean groupRmsEligible(int dim) {
        return dim % Qwen35BatchKernels.RMS_GROUP_LOCAL == 0 && dim <= 12288;
    }

    /**
     * Whether the causal convolution runs as a lane per (row, channel) plus a window update ({@code
     * causalConv1dSiluSplitBatch} + {@code causalConv1dWindowUpdate}) rather than the per-channel
     * scan, the SiLU and the split: the window update is written for the four-tap kernel this
     * family has.
     */
    private boolean parallelConv() {
        return config.ssmConvKernel() == 4;
    }

    /** {@code normed[b] = weight ⊙ rms(x[b])} — a scale per row, then the apply. */
    private void normalize(
            TaskGraph layer,
            int layerIndex,
            String reduce,
            String apply,
            FloatArray scaleBatch,
            TornadoTensor weight) {
        if (groupRmsEligible(config.dim())) {
            layer.task(
                    reduce,
                    Qwen35BatchKernels::rmsReduceBatchGroup,
                    context,
                    state.workspace.wrapXBatch,
                    scaleBatch,
                    config.dim(),
                    config.rmsNormEps());
        } else {
            layer.task(
                    reduce,
                    TransformerBatchPrefillKernels::batchedRmsReduce,
                    context,
                    state.workspace.wrapXBatch,
                    scaleBatch,
                    config.dim(),
                    config.rmsNormEps());
        }
        layer.task(
                apply,
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapXBatch,
                weight.asFloatArray(),
                scaleBatch,
                config.dim());
        if (TENSOR_CORES && TensorCoreSupport.isTensorCoreCapableBackend()) {
            if (fakeQuantizeForTests) {
                layer.task(
                        apply + "_fp16",
                        Int8GemmKernels::fakeQuantizeNormedToFP16,
                        context,
                        state.workspace.wrapNormedBatch,
                        state.workspace.wrapNormedFP16Batch,
                        config.dim(),
                        state.workspace.batchStartPosHolder);
            } else {
                layer.task(
                        apply + "_fp16",
                        Qwen35MMAKernels::convertNormedToFP16,
                        context,
                        state.workspace.wrapNormedBatch,
                        state.workspace.wrapNormedFP16Batch,
                        config.dim(),
                        state.workspace.batchStartPosHolder);
            }
        }
        if (int8Quantizes(config.dim())) {
            // The normed chunk quantized once for every Q4_0 projection that reads it.
            quantizeActivation(
                    layer,
                    layerIndex,
                    apply + "_q8",
                    state.workspace.wrapNormedBatch,
                    config.dim());
        }
    }

    private void attentionBranch(TaskGraph layer, int layerIndex) {
        final int headDim = config.numberOfHeadsKey();
        final int kvDim = config.kvDim();
        final int attnDim = config.attentionOutputInputDim();
        final int kvLayer = config.keyValueLayerIndex(layerIndex);

        matVecBatch(
                layer,
                layerIndex,
                "attn_q_proj",
                "attn_q",
                require(weights.wqLayered, layerIndex, "attn_q"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapQGateBatch,
                config.dim(),
                config.queryGateDim(),
                false);
        matVecBatch(
                layer,
                layerIndex,
                "attn_k_proj",
                "attn_k",
                require(weights.wkLayered, layerIndex, "attn_k"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapKBatch,
                config.dim(),
                kvDim,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "attn_v_proj",
                "attn_v",
                require(weights.wvLayered, layerIndex, "attn_v"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapVBatch,
                config.dim(),
                kvDim,
                false);

        layer.task(
                "attn_split_query_gate",
                Qwen35BatchKernels::splitQueryGateBatch,
                context,
                state.workspace.wrapQGateBatch,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapAttnGateBatch,
                config.numberOfHeads(),
                headDim,
                state.workspace.batchStartPosHolder);

        layer.task(
                "attn_qk_norm",
                groupNormEligible(headDim)
                        ? Qwen35BatchKernels::fusedQKRmsNormBatchGroup
                        : Qwen35BatchKernels::fusedQKRmsNormBatch,
                context,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapKBatch,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);

        layer.task(
                "attn_rope",
                Qwen35BatchKernels::ropeNeoxPartialBatch,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapKBatch,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.ropeDimensionCount());

        // Every row's key and value written before any row attends: a row may read an earlier
        // row's entry, and the append is what puts it there.
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }

        if (scoredAttention()) {
            // Each dot product once, kept in the state's score scratch for the two later passes.
            // The span stride is the context capacity the scratch was sized to. The warp form
            // computes each dot product with one warp (a 256-wide head over 128 lanes, on the
            // backend whose warp shuffle is verified — CUDA, the tensor-core guard; it
            // reassociates the FP32 sum); the staged form reads keys through a transposed shared
            // tile whose lane mapping is written for a 128-lane workgroup; any other width keeps
            // the per-lane form.
            if (tensorCoreAttention(headDim)) {
                // K Q^T and P V on the tensor cores, three passes through the score scratch (a
                // transposed, key-padded region per workgroup), one workgroup per (32-query tile,
                // head) with eight warps where the width divides into them, else per 16-query
                // tile with four; the FP16 staging is the state's.
                layer.task(
                        "attention",
                        wideTensorCoreAttention()
                                ? Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT32
                                : Qwen35BatchKernels::attentionBatchFP16PagedTensorCoreT,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapAttnQBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapXbBatch,
                        config.numberOfHeads(),
                        headDim,
                        kvDim,
                        config.kvMul(),
                        kvLayer,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        wideTensorCoreAttention() ? TC32_LOCAL : ATTENTION_LOCAL,
                        state.workspace.wrapAttnScoresBatch,
                        config.contextLength(),
                        state.workspace.wrapAttnStageFP16);
            } else {
                layer.task(
                        "attention",
                        warpAttention(headDim)
                                ? Qwen35BatchKernels::attentionBatchFP16PagedScoredWarp
                                : Qwen35BatchKernels::attentionBatchFP16PagedScoredStagedWide,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapAttnQBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapXbBatch,
                        config.numberOfHeads(),
                        headDim,
                        kvDim,
                        config.kvMul(),
                        kvLayer,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        ATTENTION_LOCAL,
                        state.workspace.wrapAttnScoresBatch,
                        config.contextLength());
            }
        } else if (fp16Kv()) {
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        } else {
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        }

        layer.task(
                "attn_output_gate",
                Qwen35BatchKernels::applyOutputGateBatch,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapAttnGateBatch,
                attnDim,
                state.workspace.batchStartPosHolder);

        TornadoTensor attnOutput = require(weights.woLayered, layerIndex, "attn_output");
        if (attnOutput.dataType() == DataType.Q4_0 && mmaEligible(attnDim, config.dim())) {
            // The same three-task shape ffn_down and ssm_out use: a tensor-core store overwrites,
            // so the residual is its own pass.
            //
            // Buffer lifetimes, from this graph's task order rather than from capacity alone.
            // wrapHbFP16BatchMMA holds ceil(batch/16)*16 * hiddenDim halves and is written by
            // ffn_down_fp16 *later in this same layer*; between here and there nothing reads it, so
            // the gated attention output can be staged in its first batchSize * attnDim elements
            // (6144 <= 17408). wrapFFNDownBatch holds batch * dim floats — exactly this
            // projection's output width — and is likewise written by ffn_down_proj later and read
            // only by the residual pass that immediately follows each write. wrapXBatch, the
            // residual destination, is read and written by the add below and by nothing in
            // between.
            if (q40Int8("attn_output_proj", attnOutput, config.dim(), attnDim)) {
                quantizeActivation(
                        layer, layerIndex, "attn_output_q8", state.workspace.wrapXbBatch, attnDim);
            } else if (fakeQuantizeForTests) {
                layer.task(
                        "attn_output_fp16",
                        Int8GemmKernels::fakeQuantizeToFP16,
                        context,
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapHbFP16BatchMMA);
            } else {
                layer.task(
                        "attn_output_fp16",
                        Qwen35MMAKernels::convertToFP16,
                        context,
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapHbFP16BatchMMA);
            }
            boolean fused = q40Pair(config.dim(), attnDim);
            q40Projection(
                    layer,
                    "batchLayer_" + layerIndex + ".attn_output_proj",
                    "attn_output_proj",
                    state.workspace.wrapHbFP16BatchMMA,
                    attnOutput,
                    fused ? state.workspace.wrapXBatch : state.workspace.wrapFFNDownBatch,
                    null,
                    null,
                    config.dim(),
                    attnDim,
                    fused ? Epilogue.RESIDUAL : Epilogue.STORE);
            if (!fused) {
                residualPass(layer, layerIndex, "attn_output_residual");
            }
            return;
        }
        matVecBatch(
                layer,
                layerIndex,
                "attn_output_proj",
                "attn_output",
                attnOutput,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                attnDim,
                config.dim(),
                true);
    }

    private void deltaNetBranch(TaskGraph layer, int layerIndex) {
        final int convDim = config.deltaNetConvDim();
        final int keyDim = config.deltaNetKeyDim();
        final int valueDim = config.deltaNetValueDim();
        final int valueHeads = config.numberOfValueHeads();
        final int headK = config.headKeyDim();
        final int headV = config.headValueDim();
        final int recurrent = config.recurrentLayerIndex(layerIndex);

        matVecBatch(
                layer,
                layerIndex,
                "ssm_qkv_proj",
                "attn_qkv",
                require(weights.ssmQkv, layerIndex, "attn_qkv"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmQkvBatch,
                config.dim(),
                convDim,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "ssm_gate_proj",
                "attn_gate",
                require(weights.ssmGate, layerIndex, "attn_gate"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmZBatch,
                config.dim(),
                valueDim,
                false);
        TornadoTensor ssmBeta = require(weights.ssmBeta, layerIndex, "ssm_beta");
        TornadoTensor ssmAlpha = require(weights.ssmAlpha, layerIndex, "ssm_alpha");
        if (ssmBeta.dataType() == DataType.Q8_0
                && ssmAlpha.dataType() == DataType.Q8_0
                && quantizedInput == state.workspace.wrapNormedBatch) {
            // Both narrow projections in one launch, against the chunk the norm quantized.
            alphaBetaFused.add("batchLayer_" + layerIndex + ".");
            layer.task(
                    "ssm_alpha_beta",
                    Qwen35BatchKernels::alphaBetaQ8_0,
                    context,
                    state.workspace.wrapQ8ActBatch,
                    state.workspace.wrapQ8ActScales,
                    ssmAlpha.asByteArray(),
                    ssmBeta.asByteArray(),
                    state.workspace.wrapSsmAlphaBatch,
                    state.workspace.wrapSsmBetaBatch,
                    state.workspace.batchStartPosHolder,
                    config.dim(),
                    valueHeads);
        } else {
            matVecBatch(
                    layer,
                    layerIndex,
                    "ssm_beta_proj",
                    "ssm_beta",
                    ssmBeta,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapSsmBetaBatch,
                    config.dim(),
                    valueHeads,
                    false);
            matVecBatch(
                    layer,
                    layerIndex,
                    "ssm_alpha_proj",
                    "ssm_alpha",
                    ssmAlpha,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapSsmAlphaBatch,
                    config.dim(),
                    valueHeads,
                    false);
        }

        layer.task(
                "ssm_decay_beta",
                Qwen35BatchKernels::decayAndBetaBatch,
                context,
                state.workspace.wrapSsmAlphaBatch,
                state.workspace.wrapSsmBetaBatch,
                require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray(),
                require(weights.ssmA, layerIndex, "ssm_a").asFloatArray(),
                valueHeads,
                state.workspace.batchStartPosHolder);

        if (parallelConv()) {
            // A lane per (row, channel), the rows being independent given the initial window;
            // then the chunk's final window as a task of its own, after every row has read the
            // initial one. Bit-equal to the scan.
            // The SiLU and the q/k/v split folded into the convolution's store.
            layer.task(
                    "ssm_conv",
                    Qwen35BatchKernels::causalConv1dSiluSplitBatch,
                    context,
                    state.workspace.wrapSsmQkvBatch,
                    require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray(),
                    state.workspace.wrapConvState,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    keyDim,
                    keyDim,
                    valueDim,
                    config.ssmConvKernel(),
                    recurrent * config.convStateSize(),
                    state.workspace.batchStartPosHolder);
            layer.task(
                    "ssm_conv_window",
                    Qwen35BatchKernels::causalConv1dWindowUpdate,
                    context,
                    state.workspace.wrapSsmQkvBatch,
                    state.workspace.wrapConvState,
                    convDim,
                    config.ssmConvKernel(),
                    recurrent * config.convStateSize(),
                    state.workspace.batchStartPosHolder);
        } else {
            // The scan: one lane per channel, walking the chunk in token order.
            layer.task(
                    "ssm_conv",
                    Qwen35BatchKernels::causalConv1dScan,
                    context,
                    state.workspace.wrapSsmQkvBatch,
                    require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray(),
                    state.workspace.wrapConvState,
                    state.workspace.wrapSsmConvOutBatch,
                    convDim,
                    config.ssmConvKernel(),
                    recurrent * config.convStateSize(),
                    state.workspace.batchStartPosHolder);
        }
        if (!parallelConv()) {
            layer.task(
                    "ssm_conv_silu",
                    Qwen35BatchKernels::siluInPlaceBatch,
                    context,
                    state.workspace.wrapSsmConvOutBatch,
                    convDim,
                    state.workspace.batchStartPosHolder);

            layer.task(
                    "ssm_split_qkv",
                    Qwen35BatchKernels::splitThreeWayBatch,
                    context,
                    state.workspace.wrapSsmConvOutBatch,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    keyDim,
                    keyDim,
                    valueDim,
                    state.workspace.batchStartPosHolder);
        }

        layer.task(
                "ssm_l2norm_q",
                groupNormEligible(headK)
                        ? Qwen35BatchKernels::l2NormPerHeadBatchGroup
                        : Qwen35BatchKernels::l2NormPerHeadBatch,
                context,
                state.workspace.wrapSsmQBatch,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);
        layer.task(
                "ssm_l2norm_k",
                groupNormEligible(headK)
                        ? Qwen35BatchKernels::l2NormPerHeadBatchGroup
                        : Qwen35BatchKernels::l2NormPerHeadBatch,
                context,
                state.workspace.wrapSsmKBatch,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);
        layer.task(
                "ssm_scale_q",
                Qwen35BatchKernels::scaleInPlaceBatch,
                context,
                state.workspace.wrapSsmQBatch,
                (float) (1.0 / Math.sqrt(headK)),
                keyDim,
                state.workspace.batchStartPosHolder);

        // The other scan: one lane per (value head, value column), same order.
        // The warp-per-column scan where the state is 128 wide and the backend's warp shuffle is
        // the one verified correct (CUDA: the same guard as the tensor-core path; OpenCL compiles
        // the shuffle and computes the wrong answer); else the shared-state scan for a 128-wide
        // state; else the per-lane scan over the persistent state.
        layer.task(
                "ssm_delta_rule",
                warpScan(headV)
                        ? Qwen35BatchKernels::deltaRuleScanWarp
                        : Qwen35BatchKernels.deltaSharedEligible(headV)
                                ? Qwen35BatchKernels::deltaRuleScanShared
                                : Qwen35BatchKernels::deltaRuleScan,
                context,
                state.workspace.wrapSsmQBatch,
                state.workspace.wrapSsmKBatch,
                state.workspace.wrapSsmVBatch,
                state.workspace.wrapSsmAlphaBatch,
                state.workspace.wrapSsmBetaBatch,
                state.workspace.wrapDeltaState,
                state.workspace.wrapSsmOutBatch,
                valueHeads,
                config.numberOfKeyHeads(),
                headV,
                recurrent * config.deltaNetStateSize(),
                state.workspace.batchStartPosHolder);

        layer.task(
                "ssm_gated_norm",
                groupNormEligible(headV)
                        ? Qwen35BatchKernels::gatedNormPerHeadBatchGroup
                        : Qwen35BatchKernels::gatedNormPerHeadBatch,
                context,
                state.workspace.wrapSsmOutBatch,
                state.workspace.wrapSsmZBatch,
                require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray(),
                valueHeads,
                headV,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);

        TornadoTensor ssmOut = require(weights.ssmOut, layerIndex, "ssm_out");
        if (ssmOut.dataType() == DataType.Q5_K && mmaEligible(valueDim, config.dim())) {
            // Same shape as ffn_down: convert the readout, project, then add the residual back,
            // because a tensor-core store overwrites.
            layer.task(
                    "ssm_out_fp16",
                    Qwen35MMAKernels::convertToFP16,
                    context,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapSsmOutFP16Batch);
            boolean ssmOutResidualFused = false;
            if (dequantGemmEligible(config.dim(), valueDim) && valueDim % 64 == 0) {
                // The same tiled pair the wide Q4_0 projections take, through the same scratch,
                // with the Q5_K paired-nibble decoder (a lane per qs byte, which asks for whole
                // fours of k-tiles per row block): its dequantization runs after the previous
                // projection's GEMM has read the scratch and before this GEMM, in this graph's
                // task order.
                String qualified = "batchLayer_" + layerIndex + ".ssm_out_proj";
                dequantTasks.put(qualified + "_dequant", config.dim() * valueDim / 2);
                gemmTasks.put(qualified, config.dim());
                layer.task(
                        "ssm_out_proj_dequant",
                        Qwen35MMAKernels::dequantizeQ5_KToFP16TiledPairs,
                        context,
                        ssmOut.asByteArray(),
                        state.workspace.wrapDequantScratchFP16,
                        config.dim(),
                        valueDim);
                pairGemm(
                        layer,
                        "ssm_out_proj",
                        state.workspace.wrapSsmOutFP16Batch,
                        state.workspace.wrapXBatch,
                        null,
                        null,
                        config.dim(),
                        valueDim,
                        Epilogue.RESIDUAL);
                ssmOutResidualFused = true;
            } else {
                mmaTasks.put("batchLayer_" + layerIndex + ".ssm_out_proj", config.dim());
                layer.task(
                        "ssm_out_proj",
                        Qwen35MMAKernels::projectionMMAQ5_KPaired,
                        context,
                        state.workspace.wrapSsmOutFP16Batch,
                        ssmOut.asByteArray(),
                        state.workspace.wrapFFNDownBatch,
                        batchSize,
                        config.dim(),
                        valueDim);
            }
            if (!ssmOutResidualFused) {
                residualPass(layer, layerIndex, "ssm_out_residual");
            }
        } else {
            matVecBatch(
                    layer,
                    layerIndex,
                    "ssm_out_proj",
                    "ssm_out",
                    ssmOut,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapXBatch,
                    valueDim,
                    config.dim(),
                    true);
        }
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    private void transferLayerWeights(TaskGraph layer, int layerIndex) {
        List<Object> tensors = new ArrayList<>();
        tensors.add(weights.rms_att_weightLayered[layerIndex].asFloatArray());
        tensors.add(weights.rms_ffn_weightLayered[layerIndex].asFloatArray());
        if (feedForward != null) {
            tensors.addAll(feedForward.layerWeights(layerIndex));
        } else {
            tensors.add(require(weights.w1Layered, layerIndex, "ffn_gate"));
            tensors.add(require(weights.w2Layered, layerIndex, "ffn_down"));
            tensors.add(require(weights.w3Layered, layerIndex, "ffn_up"));
        }
        if (config.isRecurrentLayer(layerIndex)) {
            tensors.add(require(weights.ssmQkv, layerIndex, "attn_qkv"));
            tensors.add(require(weights.ssmGate, layerIndex, "attn_gate"));
            tensors.add(require(weights.ssmAlpha, layerIndex, "ssm_alpha"));
            tensors.add(require(weights.ssmBeta, layerIndex, "ssm_beta"));
            tensors.add(require(weights.ssmOut, layerIndex, "ssm_out"));
            tensors.add(require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray());
            tensors.add(require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray());
            tensors.add(require(weights.ssmA, layerIndex, "ssm_a").asFloatArray());
            tensors.add(require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray());
        } else {
            tensors.add(require(weights.wqLayered, layerIndex, "attn_q"));
            tensors.add(require(weights.wkLayered, layerIndex, "attn_k"));
            tensors.add(require(weights.wvLayered, layerIndex, "attn_v"));
            tensors.add(require(weights.woLayered, layerIndex, "attn_output"));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray());
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray());
        }
        state.workspace.packedRepack.upload(layer, tensors.toArray());
    }

    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    // @formatter:off
    /**
     * The chunk's scratch and its persistent state.
     *
     * <p>The recurrent state is uploaded once and never read back: it is the session's own history,
     * advanced in place by these graphs and then by the decode graphs. Uploading it per chunk would
     * overwrite what the previous chunk wrote with the host's stale zeros, which is the one way a
     * chunk boundary could change the answer.
     */
    // @formatter:on
    private void configureTransfers(TaskGraph layer, int layerIndex, String predecessor) {
        if (layerIndex == firstLayer) {
            if (state.workspace.wrapDequantScratchF32 != null) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapDequantScratchF32);
            }
            if (state.workspace.wrapQ8SplitPartial != null) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapQ8SplitPartial);
            }
            if (feedForward != null) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, feedForward.scratch());
            }
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapNormedFP16Batch,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbFP16BatchMMA,
                    state.workspace.wrapFFNDownBatch,
                    state.workspace.wrapSsmOutFP16Batch,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQGateBatch,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapAttnGateBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyStore(),
                    valueStore());
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    state.workspace.wrapSsmQkvBatch,
                    state.workspace.wrapSsmConvOutBatch,
                    state.workspace.wrapSsmZBatch,
                    state.workspace.wrapSsmAlphaBatch,
                    state.workspace.wrapSsmBetaBatch,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapConvState,
                    state.workspace.wrapDeltaState);
            if (scoredAttention()) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttnScoresBatch);
            }
            if (state.workspace.wrapAttnStageFP16 != null) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttnStageFP16);
            }
            if (state.workspace.wrapDequantScratchFP16 != null) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapDequantScratchFP16);
            }
            if (state.workspace.wrapQ8ActBatch != null) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        state.workspace.wrapQ8ActBatch,
                        state.workspace.wrapQ8ActScales);
            }
        } else {
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapNormedFP16Batch,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbFP16BatchMMA,
                    state.workspace.wrapFFNDownBatch,
                    state.workspace.wrapSsmOutFP16Batch,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQGateBatch,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapAttnGateBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyStore(),
                    valueStore(),
                    state.workspace.batchStartPosHolder);
            layer.consumeFromDevice(predecessor, state.workspace.wrapBlockTable);
            if (feedForward != null) {
                layer.consumeFromDevice(predecessor, feedForward.scratch());
            }
            if (state.workspace.wrapQ8ActBatch != null) {
                layer.consumeFromDevice(
                        predecessor,
                        state.workspace.wrapQ8ActBatch,
                        state.workspace.wrapQ8ActScales);
            }
            layer.consumeFromDevice(
                    predecessor,
                    state.workspace.wrapSsmQkvBatch,
                    state.workspace.wrapSsmConvOutBatch,
                    state.workspace.wrapSsmZBatch,
                    state.workspace.wrapSsmAlphaBatch,
                    state.workspace.wrapSsmBetaBatch,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapConvState,
                    state.workspace.wrapDeltaState);
            if (scoredAttention()) {
                layer.consumeFromDevice(predecessor, state.workspace.wrapAttnScoresBatch);
            }
            if (state.workspace.wrapAttnStageFP16 != null) {
                layer.consumeFromDevice(predecessor, state.workspace.wrapAttnStageFP16);
            }
            if (state.workspace.wrapDequantScratchFP16 != null) {
                layer.consumeFromDevice(predecessor, state.workspace.wrapDequantScratchFP16);
            }
            if (state.workspace.wrapDequantScratchF32 != null) {
                layer.consumeFromDevice(predecessor, state.workspace.wrapDequantScratchF32);
            }
            if (state.workspace.wrapQ8SplitPartial != null) {
                layer.consumeFromDevice(predecessor, state.workspace.wrapQ8SplitPartial);
            }
        }
    }

    /** Lanes of the 32-query tensor-core attention's workgroup. */
    private static final int TC32_LOCAL = Qwen35BatchKernels.TC32_LANES;

    /** Whether the tensor-core attention takes 32 queries per workgroup: the width divides. */
    private boolean wideTensorCoreAttention() {
        return batchSize % Qwen35BatchKernels.TC32_QUERIES == 0;
    }

    /**
     * Whether the batched attention is the tensor-core kernel: the scored path with its staging
     * allocated (whole 16-query tiles), a 256-wide head, 128 lanes, a context of whole 32-key
     * tiles, on CUDA (the tensor-core guard).
     */
    private boolean tensorCoreAttention(int headDim) {
        return scoredAttention()
                && state.workspace.wrapAttnStageFP16 != null
                && Qwen35BatchKernels.attentionTensorCoreEligible(
                        headDim, ATTENTION_LOCAL, batchSize, config.contextLength())
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /**
     * Whether the batched attention's first pass is the warp-per-dot-product form: a 256-wide head
     * over the 128-lane workgroup, on the backend whose warp shuffle is verified correct — CUDA,
     * the tensor-core guard.
     */
    private static boolean warpAttention(int headDim) {
        return Qwen35BatchKernels.attentionWarpEligible(headDim, ATTENTION_LOCAL)
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /**
     * Whether the batched delta-rule scan is the warp-per-column form: a 128-wide state, on the
     * backend whose warp shuffle is verified correct — CUDA, the tensor-core guard.
     */
    private static boolean warpScan(int stateDim) {
        return Qwen35BatchKernels.deltaWarpEligible(stateDim)
                && TensorCoreSupport.isTensorCoreCapableBackend();
    }

    /**
     * Whether attention runs the FP16 kernel that computes each query-key dot product once: the
     * half-precision store, and the score scratch the state allocates alongside it. Not on Metal,
     * where {@code attentionBatchFP16Paged}, the FP16 twin of the kernel the FP32 cache runs there,
     * is the faster one (M4 Pro, Qwen3.5-0.8B Q4_0, batch 256, pp512 / pp2048 tok/s: 1821 / 947
     * scored).
     */
    private boolean scoredAttention() {
        return fp16Kv()
                && state.workspace.wrapAttnScoresBatch != null
                && !org.beehive.jitllm.runtime.backend.BackendId.METAL.equals(
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend());
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        final int headDim = config.numberOfHeadsKey();

        // A workgroup per row where the group kernel is dispatched, else a lane per row.
        WorkerGrid rmsReduce =
                groupRmsEligible(config.dim())
                        ? WorkerGridFactory.genericWorker(
                                batchSize * Qwen35BatchKernels.RMS_GROUP_LOCAL,
                                Qwen35BatchKernels.RMS_GROUP_LOCAL)
                        : WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApply =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        // One lane per element of the padded chunk.
        WorkerGrid fp16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.dim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid ssmOutFP16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.deltaNetValueDim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid hbFP16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.hiddenDim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid attnOutputFP16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.attentionOutputInputDim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid residualAdd =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid swiglu =
                WorkerGridFactory.genericWorker(batchSize * config.hiddenDim(), ELEMENTWISE_LOCAL);
        WorkerGrid queryGate =
                WorkerGridFactory.genericWorker(
                        batchSize * config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);
        // A workgroup per (row, head) of a lane per element where the group kernel is
        // dispatched, else a lane per (row, head).
        int qkHeads = batchSize * (config.numberOfHeads() + config.numberOfKeyValueHeads());
        WorkerGrid qkNorm =
                groupNormEligible(config.headSize())
                        ? WorkerGridFactory.genericWorker(
                                qkHeads * config.headSize(), config.headSize())
                        : WorkerGridFactory.genericWorker(qkHeads, 1);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * (config.ropeDimensionCount() / 2), 32);
        WorkerGrid kvAppend =
                WorkerGridFactory.genericWorker(batchSize * config.kvDim(), ELEMENTWISE_LOCAL);
        // One workgroup per (row, head); the workgroup's lanes split the causal range. The
        // tensor-core kernel: one per (16-query tile, head).
        boolean tensorCore = tensorCoreAttention(config.headSize());
        int attentionRows =
                tensorCore && wideTensorCoreAttention()
                        ? Qwen35BatchKernels.TC32_QUERIES
                        : Qwen35Configuration.ATTENTION_TILE_ROWS;
        int attentionLocal = tensorCore && wideTensorCoreAttention() ? TC32_LOCAL : ATTENTION_LOCAL;
        int attentionGroups =
                tensorCore
                        ? (batchSize / attentionRows) * config.numberOfHeads()
                        : batchSize * config.numberOfHeads();
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(attentionGroups * attentionLocal, attentionLocal);
        WorkerGrid outputGate =
                WorkerGridFactory.genericWorker(
                        batchSize * config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);

        WorkerGrid convDim =
                WorkerGridFactory.genericWorker(
                        batchSize * config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        // The scans launch one lane per channel or per state column — not per row. The chunk is
        // the loop inside the lane.
        WorkerGrid convChannels =
                WorkerGridFactory.genericWorker(config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        // The shared-state scan's worker is a 32-lane group per (head, 32 columns); the per-lane
        // scan's is one lane per (head, column) in 128-lane groups. Same lane count, different
        // grouping, so the grid has to match the kernel that was dispatched.
        WorkerGrid deltaColumns =
                warpScan(config.headValueDim())
                        ? WorkerGridFactory.genericWorker(
                                config.numberOfValueHeads() * config.headValueDim() * 32,
                                Qwen35BatchKernels.DELTA_WARP_LOCAL)
                        : Qwen35BatchKernels.deltaSharedEligible(config.headValueDim())
                                ? WorkerGridFactory.genericWorker(
                                        config.numberOfValueHeads() * config.headValueDim(),
                                        Qwen35BatchKernels.DELTA_SHARED_COLUMNS)
                                : WorkerGridFactory.genericWorker(
                                        config.numberOfValueHeads() * config.headValueDim(),
                                        ELEMENTWISE_LOCAL);
        WorkerGrid keyDim =
                WorkerGridFactory.genericWorker(
                        batchSize * config.deltaNetKeyDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyHeads =
                groupNormEligible(config.headKeyDim())
                        ? WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfKeyHeads() * config.headKeyDim(),
                                config.headKeyDim())
                        : WorkerGridFactory.genericWorker(batchSize * config.numberOfKeyHeads(), 1);
        WorkerGrid valueHeads =
                groupNormEligible(config.headValueDim())
                        ? WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfValueHeads() * config.headValueDim(),
                                config.headValueDim())
                        : WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfValueHeads(), 1);
        // Elementwise over (row, value head); the kernel guards its lane, so any local size.
        WorkerGrid decayBeta =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfValueHeads(), ELEMENTWISE_LOCAL);

        for (int layer = firstLayer; layer < endLayer; layer++) {
            String prefix = "batchLayer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
            if (feedForward != null) {
                feedForward.addWorkerGrids(scheduler, prefix);
            }
            if (TENSOR_CORES && TensorCoreSupport.isTensorCoreCapableBackend()) {
                scheduler.addWorkerGrid(prefix + "attn_rms_apply_fp16", fp16Convert);
                scheduler.addWorkerGrid(prefix + "ffn_rms_apply_fp16", fp16Convert);
            }
            if (onTensorCores(prefix + "ffn_gate_proj")
                    || simdgroupTasks.containsKey(prefix + "ffn_gate_proj")) {
                scheduler.addWorkerGrid(
                        prefix + "ffn_gate_proj",
                        matVecWorker(prefix + "ffn_gate_proj", config.hiddenDim()));
                scheduler.addWorkerGrid(
                        prefix + "ffn_up_proj",
                        matVecWorker(prefix + "ffn_up_proj", config.hiddenDim()));
                if (!fusedSwigluLayers.contains(prefix)) {
                    scheduler.addWorkerGrid(prefix + "ffn_swiglu", swiglu);
                }
            } else if (q8SwigluLayers.contains(prefix)) {
                for (String projection : List.of("ffn_gate_proj", "ffn_up_proj")) {
                    if (warpMatVecTasks.containsKey(prefix + projection)) {
                        scheduler.addWorkerGrid(
                                prefix + projection,
                                matVecWorker(prefix + projection, config.hiddenDim()));
                    }
                }
                scheduler.addWorkerGrid(prefix + "ffn_swiglu", swiglu);
            } else {
                scheduler.addWorkerGrid(
                        prefix + "ffn_gate_up",
                        matVecWorker(prefix + "ffn_gate_up", config.hiddenDim()));
            }
            scheduler.addWorkerGrid(
                    prefix + "ffn_down_proj", matVecWorker(prefix + "ffn_down_proj", config.dim()));
            if (onTensorCores(prefix + "ffn_down_proj")) {
                // Built only when SwiGLU did not write the FP16 buffer itself, i.e. when the
                // gate/up projections are not on the tensor cores.
                if (hbConvertLayers.contains(prefix)) {
                    scheduler.addWorkerGrid(prefix + "ffn_down_fp16", hbFP16Convert);
                }
            }

            if (config.isRecurrentLayer(layer)) {
                scheduler.addWorkerGrid(
                        prefix + "ssm_qkv_proj",
                        matVecWorker(prefix + "ssm_qkv_proj", config.deltaNetConvDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_gate_proj",
                        matVecWorker(prefix + "ssm_gate_proj", config.deltaNetValueDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_beta_proj",
                        matVecWorker(prefix + "ssm_beta_proj", config.numberOfValueHeads()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_alpha_proj",
                        matVecWorker(prefix + "ssm_alpha_proj", config.numberOfValueHeads()));
                if (alphaBetaFused.contains(prefix)) {
                    scheduler.addWorkerGrid(
                            prefix + "ssm_alpha_beta",
                            WorkerGridFactory.genericWorker(batchSize * 128, 128));
                }
                if (alphaBetaFused.contains(prefix)) {
                    scheduler.addWorkerGrid(
                            prefix + "ssm_alpha_beta",
                            WorkerGridFactory.genericWorker(batchSize * 128, 128));
                }
                scheduler.addWorkerGrid(prefix + "ssm_decay_beta", decayBeta);
                if (parallelConv()) {
                    scheduler.addWorkerGrid(prefix + "ssm_conv", convDim);
                    scheduler.addWorkerGrid(prefix + "ssm_conv_window", convChannels);
                } else {
                    scheduler.addWorkerGrid(prefix + "ssm_conv", convChannels);
                }
                if (!parallelConv()) {
                    scheduler.addWorkerGrid(prefix + "ssm_conv_silu", convDim);
                    scheduler.addWorkerGrid(prefix + "ssm_split_qkv", convDim);
                }
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_q", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_k", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_scale_q", keyDim);
                scheduler.addWorkerGrid(prefix + "ssm_delta_rule", deltaColumns);
                scheduler.addWorkerGrid(prefix + "ssm_gated_norm", valueHeads);
                scheduler.addWorkerGrid(
                        prefix + "ssm_out_proj",
                        matVecWorker(prefix + "ssm_out_proj", config.dim()));
                if (onTensorCores(prefix + "ssm_out_proj")) {
                    if (!quantizeTasks.containsKey(prefix + "ssm_out_q8")) {
                        scheduler.addWorkerGrid(prefix + "ssm_out_fp16", ssmOutFP16Convert);
                    }
                }
            } else {
                scheduler.addWorkerGrid(
                        prefix + "attn_q_proj",
                        matVecWorker(prefix + "attn_q_proj", config.queryGateDim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_k_proj",
                        matVecWorker(prefix + "attn_k_proj", config.kvDim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_v_proj",
                        matVecWorker(prefix + "attn_v_proj", config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_split_query_gate", queryGate);
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                scheduler.addWorkerGrid(prefix + "attn_output_gate", outputGate);
                scheduler.addWorkerGrid(
                        prefix + "attn_output_proj",
                        matVecWorker(prefix + "attn_output_proj", config.dim()));
                if (onTensorCores(prefix + "attn_output_proj")) {
                    if (!quantizeTasks.containsKey(prefix + "attn_output_q8")) {
                        scheduler.addWorkerGrid(prefix + "attn_output_fp16", attnOutputFP16Convert);
                    }
                }
            }
        }
        // The Q8_0 decodes that precede the cuBLAS projections: one lane per weight.
        for (var entry : q8DequantTasks.entrySet()) {
            scheduler.addWorkerGrid(
                    entry.getKey(),
                    WorkerGridFactory.genericWorker(
                            (entry.getValue() + GEMM_LOCAL - 1) / GEMM_LOCAL * GEMM_LOCAL,
                            GEMM_LOCAL));
        }
        // The dequantizations that precede the FP16 GEMMs: one lane per weight element.
        for (var entry : dequantTasks.entrySet()) {
            scheduler.addWorkerGrid(
                    entry.getKey(), WorkerGridFactory.genericWorker(entry.getValue(), GEMM_LOCAL));
        }
        for (var entry : splitReduceTasks.entrySet()) {
            scheduler.addWorkerGrid(
                    entry.getKey(),
                    WorkerGridFactory.genericWorker(
                            (entry.getValue() + GEMM_LOCAL - 1) / GEMM_LOCAL * GEMM_LOCAL,
                            GEMM_LOCAL));
        }
        for (var entry : quantizeTasks.entrySet()) {
            scheduler.addWorkerGrid(
                    entry.getKey(),
                    WorkerGridFactory.genericWorker(entry.getValue(), ELEMENTWISE_LOCAL));
        }
        for (String task : residualPasses) {
            scheduler.addWorkerGrid(task, residualAdd);
        }
    }

    /**
     * One line saying which batched-prefill path this plan took, for the init diagnostics: counted
     * from the tasks the builder registered, so it reports what was built.
     */
    public String describeDispatch() {
        long int8 = q8DirectGemms.size();
        long fp16Pairs = dequantTasks.size();
        long direct = mmaTasks.size();
        if (int8 == 0 && fp16Pairs == 0 && direct == 0) {
            return "scalar kernels (no tensor-core path at width " + batchSize + ")";
        }
        return "width "
                + batchSize
                + ": "
                + int8
                + " projections on the int8 tensor-core pair, "
                + fp16Pairs
                + " on the FP16 dequantize-then-GEMM pair, "
                + direct
                + " on the direct FP16 MMA kernels; tensor cores "
                + (TENSOR_CORES && TensorCoreSupport.isTensorCoreCapableBackend()
                        ? "enabled"
                        : "disabled")
                + ", int8 MMA "
                + (TensorCoreSupport.isInt8MmaCapable() ? "available" : "unavailable");
    }

    /** Whether a projection was placed on either tensor-core path. */
    private boolean onTensorCores(String qualifiedTask) {
        return mmaTasks.containsKey(qualifiedTask) || gemmTasks.containsKey(qualifiedTask);
    }

    /** One workgroup per (row, output row), or per (row tile, output row) where tiled. */
    private WorkerGrid matVecWorker(String qualifiedTask, int rows) {
        Integer simdgroupOutputs = simdgroupTasks.get(qualifiedTask);
        if (simdgroupOutputs != null) {
            int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
            return WorkerGridFactory.genericWorker(
                    ((simdgroupOutputs + 63) / 64) * ((batchSize + 63) / 64) * local, local);
        }
        Integer warpOutputs = warpMatVecTasks.get(qualifiedTask);
        if (warpOutputs != null) {
            if (warpOutputs < 0) {
                // One warp per 4 x 4 tile of (row, output) pairs, four to a 128-lane block.
                int tile = TransformerBatchPrefillKernels.MATVEC_TILE;
                int tiles = ((batchSize + tile - 1) / tile) * (-warpOutputs / tile);
                return WorkerGridFactory.genericWorker(tiles * 32, MATVEC_LOCAL);
            }
            // One warp per (row, output), four to a 128-lane block.
            return WorkerGridFactory.genericWorker(batchSize * warpOutputs * 32, MATVEC_LOCAL);
        }
        Integer gemmCols = gemmTasks.get(qualifiedTask);
        if (gemmCols != null) {
            // gemmMMA's documented worker: (M/128) * 256 by N/128, 256 threads per block; a split
            // GEMM has a column of blocks per split.
            int local =
                    q8DirectGemms.contains(qualifiedTask)
                            ? Int8GemmKernels.Q8_GEMM_THREADS
                            : GEMM_LOCAL;
            WorkerGrid gemm =
                    new uk.ac.manchester.tornado.api.WorkerGrid2D(
                            (batchSize / GEMM_TILE) * local,
                            gemmCols / GEMM_TILE * gemmSplitCounts.getOrDefault(qualifiedTask, 1));
            gemm.setLocalWork(local, 1, 1);
            return gemm;
        }
        Integer mmaCols = mmaTasks.get(qualifiedTask);
        if (mmaCols != null) {
            int rowTilesMma = batchSize / Qwen35MMAKernels.BM;
            int colTilesMma = mmaCols / Qwen35MMAKernels.BN;
            return WorkerGridFactory.genericWorker(
                    rowTilesMma * colTilesMma * Qwen35MMAKernels.LOCAL, Qwen35MMAKernels.LOCAL);
        }
        int tileRows = rowTiles.getOrDefault(qualifiedTask, 1);
        int tileCols = colTiles.getOrDefault(qualifiedTask, 1);
        int rowGroups = (batchSize + tileRows - 1) / tileRows;
        int colGroups = (rows + tileCols - 1) / tileCols;
        return WorkerGridFactory.genericWorker(rowGroups * colGroups * MATVEC_LOCAL, MATVEC_LOCAL);
    }

    /** Layers whose decay and beta projections run as one fused int8 launch, for its grid. */
    private final java.util.Set<String> alphaBetaFused = new java.util.HashSet<>();
}
