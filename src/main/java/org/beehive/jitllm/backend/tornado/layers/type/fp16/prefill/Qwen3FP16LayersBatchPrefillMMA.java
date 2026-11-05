package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.CuDnnPrefillAttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.SplitKGemm;
import org.beehive.jitllm.backend.tornado.layers.TensorCoreBatchAttention;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.ops.LayerAttention;
import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.backend.tornado.ops.jit.JitLayerAttention;
import org.beehive.jitllm.backend.tornado.ops.jit.JitLayerProjections;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasLayerProjections;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CudnnLayerAttention;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cudnn.CuDnn;

/**
 * Qwen3 batched-prefill transformer-layer TaskGraphs on the tensor-core (MMA) pipeline. Mirrors
 * {@link LlamaFP16LayersBatchPrefillMMA} with the Qwen3 architectural additions: per-head Q/K RMS
 * normalization between the QKV projection and RoPE, split-half RoPE pairing, and qDim-shaped
 * attention (qDim = nHeads * headDim, which may differ from the model dim).
 *
 * <p>Tensor-core layer pipeline (13 tasks, all GEMMs on MMA):
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
public class Qwen3FP16LayersBatchPrefillMMA
        implements BatchPrefillTransformerLayerTaskGraphs,
                LayerProjections.Tasks,
                LayerAttention.Tasks {

    /** Legacy detail; the startup report's native-library and MMA lines cover it under -v. */
    private static void logInitialization(String format, Object... args) {
        if (Boolean.getBoolean("jitllm.EnableTimingForTornadoVMInit")) {
            System.err.printf(java.util.Locale.ROOT, format, args);
        }
    }

    @Override
    public String describeProjections() {
        return nativeProjections
                ? "library-selected (MMA use not confirmed)"
                : "FP16 tensor-core MMA";
    }

    @Override
    public String describeNativeLibraries() {
        return nativeProjections
                ? (cudnnAttention ? "cuBLAS, cuDNN (first-chunk attention)" : "cuBLAS")
                : (cudnnAttention ? "cuDNN (first-chunk attention)" : "none");
    }

    @Override
    public String describeAttention() {
        return cudnnAttention
                ? "cuDNN first chunk; tensor-core attention for subsequent chunks"
                : "JIT kernels";
    }

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    private final Qwen3State state;
    private final boolean packedHalf2Attention;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();

    /** One layer's regions, in the order they run. */
    private record LayerRegions(
            Region qkv, Region attention, Region output, Region gateUp, Region down) {}

    /** Every layer's regions. */
    private final List<LayerRegions> regions;

    /** Which of cuBLAS, cuDNN and the JIT kernels runs each region. */
    private final WorkPlan plan;

    /** The plan's record of the weights its steps read, which this class does not need. */
    private final WeightBindings bindings = new WeightBindings();

    /** The plan's arrays: the tasks this class adds name their buffers themselves. */
    private static final Operands NO_OPERANDS =
            Operands.of(java.util.Map.of(), (role, layer) -> null, List.of());

    private final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM); see the Llama
    // planner for the padding rationale. Non-GEMM kernels use the true batchSize.
    private final int paddedBatch;

    /**
     * Rows a native GEMM has to produce.
     *
     * <p>{@code paddedBatch} rounds the chunk width up to 128 because the JIT {@code gemmMMA*}
     * kernels tile 128x128 and their worker grid is derived from it. cuBLAS has no such constraint,
     * and nothing downstream reads the rows above the chunk width: every other worker grid in this
     * class -- the RMS reduces and applies, the Q/K norm, RoPE, attention, the cuDNN adapters,
     * SwiGLU and the residual add -- is sized on {@code batchSize}. So a native projection only
     * needs to produce {@code batchSize} rows, and at a width that is not a multiple of 128 that is
     * strictly less work: 300 rows instead of 384.
     *
     * <p>A JIT projection still gets {@code paddedBatch}, because its grid demands it.
     */
    private int nativeGemmRows() {
        return batchSize;
    }

    private final int nHeadKv;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;

    /**
     * Whether this family's four projections are cuBLAS GEMMs over contiguous weight copies. One
     * field, not one per projection: {@link NativePrefillSupport} answers for all four together,
     * because a configuration with only some of them native has never been validated end to end --
     * and the decode side derives which weights it must upload itself from the same single answer.
     */
    private final boolean nativeProjections;

    /** The fused-attention shape this family resolved against; see {@link #sdpaShape}. */
    private final NativePrefillSupport.SdpaShape sdpaShape;

    /**
     * Whether this PLAN uses cuDNN attention at all -- true in both families when the primary does,
     * which is what decides that a fallback family is needed. {@link #useCudnnAttention} is the
     * per-family answer.
     */
    private final boolean cudnnAttention;

    /**
     * gate and up contiguous along the output dimension, one array per layer, so the packed
     * [gate|up] result can come from a single GEMM ({@link FP16Projections} draws the layout).
     * gemmMMAGateUp reads w1 and w3 as two separate operands and selects between them by output
     * column; cuBLAS takes one B and the binding has no operand offset, so the two have to be
     * contiguous in memory. Built once here, never per prefill, and the originals are left alone
     * because decode and the JIT path still use them.
     */
    private final HalfFloatArray[] nativeLibResidentContiguousGateUp;

    private final int layersPerGraph;

    /**
     * Graph-name prefix for this family. The primary keeps "batchPrefillLayer_", which the decode
     * graphs bind their weights by name from; the fallback family gets its own prefix.
     */
    private final String graphPrefix;

    /**
     * True for the fallback family: every buffer it uses is bound from the primary family's graph
     * rather than declared for upload here, so nothing is allocated or copied twice.
     */
    private final boolean consumeFromPrimary;

    /**
     * cuDNN attention in THIS family. Always false in the fallback family, which exists precisely
     * because cuDNN cannot mask the chunks it takes.
     */
    private final boolean useCudnnAttention;

    /**
     * wq, wk and wv contiguous along the output dimension, one array per layer, so the packed
     * [q|k|v] result can come from a single GEMM ({@link FP16Projections} draws the layout).
     * gemmMMAQKV reads the three as separate operands and selects between them by output column --
     * [0,qDim) from wq, [qDim,qDim+kvDim) from wk, the rest from wv, each indexed row-major
     * [outCol][K]. The slices are NOT equal: for this model 2048 | 1024 | 1024. cuBLAS takes one B
     * and the binding has no operand offset, so the three have to be contiguous in memory, in that
     * order. Built once here, never per prefill, and the originals are left alone because decode
     * and the JIT path still read them.
     */
    private final HalfFloatArray[] nativeLibResidentContiguousQkv;

    /**
     * With cuBLAS, when the copies above are not kept for the whole session: one shared scratch
     * that each layer refills with its q|k|v, then its gate|up, just before the GEMM, in every
     * chunk; {@code null} otherwise. The memory check picks which ({@link #workspace}).
     */
    private final FP16Projections nativeLibScratchContiguousProjections;

    /**
     * Staging for the cuDNN call: contiguous FP16 Q, GQA-expanded K/V, and its output, all {@code
     * [head][tok][headDim]}. Allocated once and reused by every layer graph, because the layers run
     * one after another.
     */
    private final HalfFloatArray cudnnQ;

    private final HalfFloatArray cudnnK;
    private final HalfFloatArray cudnnV;
    private final HalfFloatArray cudnnOut;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;
    private Qwen3FP16LayersBatchPrefillMMA fallbackFamily;

    /**
     * The JIT attention as tensor-core GEMMs, or null for the flash-attention kernel. A primary
     * family on cuDNN has none; its fallback family owns one for the chunks cuDNN cannot take,
     * whose cost would otherwise grow with the start position on the scalar kernel.
     */
    private final TensorCoreBatchAttention tcAttention;

    /** The JIT projections' GEMM, its depth split when the output is too narrow for the device. */
    private final SplitKGemm splitK;

    /**
     * The batched-prefill graphs only run on the CUDA backend (tensor-core gated), which is the
     * same backend the FP16 KV cache path targets.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    /**
     * How many transformer layers share one batch-prefill graph.
     *
     * <p>Grouping changes nothing about task order, buffers or arithmetic: consecutive layers are
     * appended to the same graph, with the same tasks and the same operands. What it changes is how
     * many graph submissions -- and therefore how many device syncs, since {@code
     * TaskGraph.execute} ends with a wait -- a prefill pass costs. Four was measured the knee on
     * this pipeline; the gain is sub-linear above it.
     *
     * <p><b>Gated on the tensor-core backend, and that gate is load-bearing.</b> Grouping renames a
     * layer's graph to the first layer of its group, and the decode layer graphs bind their weights
     * from that name. On a backend where {@code Qwen3FP16PlanComponents} builds the ungrouped
     * {@link Qwen3FP16LayersBatchPrefill} family instead, the graphs are still one per layer, so
     * this has to answer one per layer there too.
     *
     * <p>Static so both sides resolve the same grouping from the same predicate.
     */
    public static int prefillLayersPerGraph() {
        return TensorCoreSupport.isTensorCoreCapableBackend() ? LAYERS_PER_GRAPH : 1;
    }

    /** The measured knee; see {@link #prefillLayersPerGraph()}. */
    private static final int LAYERS_PER_GRAPH = 4;

    /** The batch-prefill graph that owns {@code layerIndex}'s weights. */
    public static String prefillGraphOwning(int layerIndex) {
        int g = prefillLayersPerGraph();
        return GraphNames.batchPrefillLayer((layerIndex / g) * g);
    }

    // @formatter:off
    /**
     * The fused-attention call this family would make, as the capability probe needs to see it.
     *
     * <p>One batch of the whole chunk, one query head per attention head, query and key lengths
     * both the chunk width -- which is what makes cuDNN's top-left causal mask the right mask --
     * and the head dimension the adapters pack into. Built here rather than at the call site so the
     * shape the probe answers for and the shape the task is given cannot drift apart.
     */
    // @formatter:on
    static NativePrefillSupport.SdpaShape sdpaShape(Qwen3Configuration config, int batchSize) {
        int headDim = config.numberOfHeadsValue();
        return new NativePrefillSupport.SdpaShape(
                1,
                config.numberOfHeads(),
                batchSize,
                batchSize,
                headDim,
                (float) (1.0 / Math.sqrt(headDim)),
                true);
    }

    /**
     * How many weights the copies hold when kept for the session: every layer's q|k|v and gate|up.
     * Qwen3-8B: 36 layers of 6144 x 4096 and 24576 x 4096, 4.5 G weights, 8.4 GiB in F16.
     */
    static long nativeLibResidentContiguousSize(Qwen3Configuration config) {
        long qkvRows =
                (long) config.numberOfHeadsKey() * config.numberOfHeads()
                        + 2L * config.numberOfHeadsValue() * config.numberOfKeyValueHeads();
        return config.numberOfLayers() * (qkvRows + 2L * config.hiddenDim()) * config.dim();
    }

    /** How many weights the per-chunk scratch holds: the largest copy, gate|up. */
    static long nativeLibScratchContiguousSize(Qwen3Configuration config) {
        long qkvRows =
                (long) config.numberOfHeadsKey() * config.numberOfHeads()
                        + 2L * config.numberOfHeadsValue() * config.numberOfKeyValueHeads();
        return Math.max(qkvRows, 2L * config.hiddenDim()) * config.dim();
    }

    /**
     * What these layers allocate for one session at {@code batchSize} beyond the cuDNN attention
     * staging, which the configuration accounts; see the constructor. Under cuBLAS, the copies kept
     * for the session, or the scratch refilled per chunk: the memory check picks.
     */
    public static BatchPrefillWorkspace workspace(
            Qwen3Configuration config, int batchSize, ExecutionPolicy policy, boolean fp16Cache) {
        int padded = (batchSize + 127) & ~127;
        int headSize = config.numberOfHeadsValue();
        boolean cudnnAttention =
                fp16Cache
                        && NativePrefillSupport.nativeAttention(
                                policy, sdpaShape(config, batchSize));
        // With cuDNN on the first chunk the fallback family runs the later chunks on the same
        // tensor-core attention, so its buffers are part of the session either way.
        long attention =
                TensorCoreBatchAttention.applies(fp16Cache, padded, headSize)
                        ? TensorCoreBatchAttention.bytes(
                                config.numberOfHeadsKey() * config.numberOfHeads(),
                                config.numberOfHeadsValue() * config.numberOfKeyValueHeads(),
                                headSize,
                                config.numberOfHeads(),
                                padded,
                                config.contextLength())
                        : 0;
        long attentionAndSplitKBytes = attention + SplitKGemm.bytes(padded, config.dim(), false);
        if (!Target.of(policy).cublas()) {
            return BatchPrefillWorkspace.withoutResidencyChoice(attentionAndSplitKBytes);
        }
        long residentContiguousSize = nativeLibResidentContiguousSize(config);
        return new BatchPrefillWorkspace(
                residentContiguousSize,
                attentionAndSplitKBytes + FP16Projections.residentBytes(residentContiguousSize),
                attentionAndSplitKBytes
                        + FP16Projections.workspaceBytes(
                                false,
                                nativeLibScratchContiguousSize(config),
                                padded,
                                config.dim(),
                                2 * config.numberOfLayers(),
                                true));
    }

    public Qwen3FP16LayersBatchPrefillMMA(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.graphPrefix = "batchPrefillLayer_";
        this.consumeFromPrimary = false;
        this.state = state;
        // Resolved once from the session's policy, not read from a class constant.
        this.packedHalf2Attention = state.executionPolicy().packedHalf2Attention();
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        this.kvDim = config.numberOfHeadsValue() * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
        // The FP16 key/value cache is part of the supported configuration, not an extra switch:
        // the cuDNN adapters read K/V straight out of it as FP16, and the gather that would be
        // needed from an FP32 cache has never been written or validated. An FP32 KV session on
        // this device keeps the JIT attention and loses nothing else -- the projections stay
        // native, because they do not touch the cache.
        //
        // The capability question is asked for THIS session's shape, not for the library in the
        // abstract: a fused-SDPA implementation can be present and still decline this device or
        // this tuple, and finding that out at execution time would be far too late.
        boolean fp16Kv = state.usesFp16KeyValueCache();
        this.sdpaShape = sdpaShape(config, batchSize);
        this.regions = regions(config, batchSize);
        this.plan = plan();
        this.cudnnAttention = plan.runs(regions.get(0).attention(), "cuDNN");
        this.useCudnnAttention = this.cudnnAttention;
        this.nativeProjections = plan.runs(regions.get(0).qkv(), "cuBLAS");
        this.layersPerGraph = Math.min(prefillLayersPerGraph(), config.numberOfLayers());
        logInitialization(
                "[jitllm] prefill acceleration: %s%n",
                NativePrefillSupport.describe(state.executionPolicy(), fp16Kv, sdpaShape));
        this.tcAttention =
                !cudnnAttention && TensorCoreBatchAttention.applies(fp16Kv, paddedBatch, nEmbdHead)
                        ? new TensorCoreBatchAttention(
                                state,
                                qDim,
                                kvDim,
                                nEmbdHead,
                                config.numberOfHeads(),
                                gqa,
                                paddedBatch,
                                config.contextLength())
                        : null;
        this.splitK = new SplitKGemm(context, paddedBatch, config.dim(), false);
        int cudnnElems = cudnnAttention ? qDim * batchSize : 0;
        this.cudnnQ = new HalfFloatArray(cudnnElems);
        this.cudnnK = new HalfFloatArray(cudnnElems);
        this.cudnnV = new HalfFloatArray(cudnnElems);
        this.cudnnOut = new HalfFloatArray(cudnnElems);
        if (cudnnAttention) {
            // Zero, not left undefined: the padded rows take part in the attention the library
            // computes, so whatever these held would reach the softmax of the real rows.
            HalfFloat zero = new HalfFloat(0.0f);
            cudnnQ.init(zero);
            cudnnK.init(zero);
            cudnnV.init(zero);
            cudnnOut.init(zero);
            logInitialization(
                    "[jitllm] prefill attention: cuDNN SDPA for the first chunk, tensor-core"
                            + " attention for the rest; staging %d MiB, one set shared by every"
                            + " batch-prefill graph%n",
                    4L * cudnnElems * Short.BYTES / (1024 * 1024));
        }
        boolean keepContiguousBuffers =
                nativeProjections
                        && FP16Projections.resident(
                                state.executionPolicy(), nativeLibResidentContiguousSize(config));
        this.nativeLibScratchContiguousProjections =
                nativeProjections && !keepContiguousBuffers
                        ? new FP16Projections(
                                context,
                                state.workspace.prepacking,
                                nativeLibScratchContiguousSize(config),
                                paddedBatch,
                                config.dim(),
                                false,
                                2 * config.numberOfLayers(),
                                true)
                        : null;
        if (nativeLibScratchContiguousProjections != null) {
            logInitialization(
                    "[jitllm] prefill QKV and gate/up: cuBLAS over one %d MiB scratch, refilled"
                            + " per chunk%n",
                    2L * nativeLibScratchContiguousSize(config) / (1024 * 1024));
        }
        this.nativeLibResidentContiguousGateUp =
                new HalfFloatArray[keepContiguousBuffers ? config.numberOfLayers() : 0];
        if (keepContiguousBuffers) {
            long t0 = System.nanoTime();
            int rowsK = config.hiddenDim() * config.dim();
            for (int l = 0; l < config.numberOfLayers(); l++) {
                HalfFloatArray w1 = weights.w1Layered[l].asHalfFloatArray();
                HalfFloatArray w3 = weights.w3Layered[l].asHalfFloatArray();
                // cuBLAS reads the copy, so the decode graphs upload w1 and w3 themselves.
                state.workspace.weightBindings.recordReplaced(w1);
                state.workspace.weightBindings.recordReplaced(w3);
                HalfFloatArray contiguous = new HalfFloatArray(2 * rowsK);
                for (int i = 0; i < rowsK; i++) {
                    contiguous.set(i, w1.get(i));
                    contiguous.set(rowsK + i, w3.get(i));
                }
                nativeLibResidentContiguousGateUp[l] = contiguous;
            }
            logInitialization(
                    "[jitllm] prefill gate/up: cuBLAS, %d contiguous copies, %d MiB extra resident,"
                            + " built in %.2f s%n",
                    config.numberOfLayers(),
                    (long) config.numberOfLayers() * 2 * rowsK * Short.BYTES / (1024 * 1024),
                    (System.nanoTime() - t0) / 1e9);
        }
        this.nativeLibResidentContiguousQkv =
                new HalfFloatArray[keepContiguousBuffers ? config.numberOfLayers() : 0];
        if (keepContiguousBuffers) {
            long t0 = System.nanoTime();
            for (int l = 0; l < config.numberOfLayers(); l++) {
                HalfFloatArray wq = weights.wqLayered[l].asHalfFloatArray();
                HalfFloatArray wk = weights.wkLayered[l].asHalfFloatArray();
                HalfFloatArray wv = weights.wvLayered[l].asHalfFloatArray();
                state.workspace.weightBindings.recordReplaced(wq);
                state.workspace.weightBindings.recordReplaced(wk);
                state.workspace.weightBindings.recordReplaced(wv);
                HalfFloatArray contiguous =
                        new HalfFloatArray(wq.getSize() + wk.getSize() + wv.getSize());
                int at = 0;
                for (int i = 0; i < wq.getSize(); i++) contiguous.set(at++, wq.get(i));
                for (int i = 0; i < wk.getSize(); i++) contiguous.set(at++, wk.get(i));
                for (int i = 0; i < wv.getSize(); i++) contiguous.set(at++, wv.get(i));
                nativeLibResidentContiguousQkv[l] = contiguous;
            }
            logInitialization(
                    "[jitllm] prefill QKV: cuBLAS, %d contiguous copies (%d|%d|%d cols), %d MiB"
                            + " extra resident, built in %.2f s%n",
                    config.numberOfLayers(),
                    qDim,
                    kvDim,
                    kvDim,
                    (long) config.numberOfLayers()
                            * nativeLibResidentContiguousQkv[0].getSize()
                            * Short.BYTES
                            / (1024 * 1024),
                    (System.nanoTime() - t0) / 1e9);
        }
        int groups = (config.numberOfLayers() + layersPerGraph - 1) / layersPerGraph;
        // Padding advice belongs to the JIT projections and to them only. Their gemmMMA* grids are
        // derived from paddedBatch, so a width that is not a multiple of 128 really does buy rows
        // of arithmetic nobody reads. The native projections produce the chunk's rows instead, so
        // repeating the advice there would send a user to round a width up for no reason.
        if (batchSize != paddedBatch) {
            if (nativeProjections) {
                logInitialization(
                        "[jitllm] prefill native GEMM rows: %d (chunk width) instead of %d"
                                + " (padded)%n",
                        batchSize, paddedBatch);
            } else {
                logInitialization(
                        "[jitllm] prefill batch %d padded to %d for tensor-core tiles; GEMM"
                                + " efficiency is %d/%d — use a multiple of 128 for best"
                                + " throughput.%n",
                        batchSize, paddedBatch, batchSize, paddedBatch);
            }
        }
        if (layersPerGraph > 1) {
            logInitialization(
                    "[jitllm] prefill layer grouping: %d layers per graph, %d graphs instead of %d%n",
                    layersPerGraph, groups, config.numberOfLayers());
        }
        this.layerITGs =
                IntStream.range(0, groups)
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    /**
     * The fallback family: the same layer pipeline with the tensor-core attention (the JIT paged
     * attention where that does not apply) instead of the cuDNN call, for a chunk whose queries do
     * not start at position 0.
     *
     * <p>It shares every array with {@code primary} -- the model's weights, the [q|k|v] and
     * [gate|up] copies, the workspace, the KV cache -- and declares all of them with {@code
     * consumeFromDevice} against the primary's graphs, so nothing is allocated, uploaded or copied
     * a second time. It does not touch the cuDNN staging quartet at all, because it does not call
     * cuDNN. The tensor-core attention's scratch is the one thing it owns: the primary, on cuDNN,
     * has none to share.
     *
     * <p>Only the attention step differs. QKV, gate/up, woProj and w2Proj stay native, with the
     * same weight copies and the same numerical contracts already validated for them.
     */
    private Qwen3FP16LayersBatchPrefillMMA(Qwen3FP16LayersBatchPrefillMMA primary) {
        this.graphPrefix = "batchPrefillFallbackLayer_";
        this.consumeFromPrimary = true;
        this.useCudnnAttention = false;
        this.state = primary.state;
        this.packedHalf2Attention = primary.packedHalf2Attention;
        this.weights = primary.weights;
        this.config = primary.config;
        this.batchSize = primary.batchSize;
        this.paddedBatch = primary.paddedBatch;
        this.cudnnAttention = primary.cudnnAttention;
        this.nativeProjections = primary.nativeProjections;
        this.sdpaShape = primary.sdpaShape;
        this.regions = primary.regions;
        this.plan = plan();
        this.layersPerGraph = primary.layersPerGraph;
        this.nHeadKv = primary.nHeadKv;
        this.nEmbdHead = primary.nEmbdHead;
        this.qDim = primary.qDim;
        this.kvDim = primary.kvDim;
        this.gqa = primary.gqa;
        this.cudnnQ = primary.cudnnQ;
        this.cudnnK = primary.cudnnK;
        this.cudnnV = primary.cudnnV;
        this.cudnnOut = primary.cudnnOut;
        this.nativeLibResidentContiguousGateUp = primary.nativeLibResidentContiguousGateUp;
        this.nativeLibResidentContiguousQkv = primary.nativeLibResidentContiguousQkv;
        this.nativeLibScratchContiguousProjections = primary.nativeLibScratchContiguousProjections;
        this.tcAttention =
                TensorCoreBatchAttention.applies(
                                state.usesFp16KeyValueCache(), paddedBatch, nEmbdHead)
                        ? new TensorCoreBatchAttention(
                                state,
                                qDim,
                                kvDim,
                                nEmbdHead,
                                config.numberOfHeads(),
                                gqa,
                                paddedBatch,
                                config.contextLength())
                        : null;
        this.splitK = primary.splitK;
        int groups = (config.numberOfLayers() + layersPerGraph - 1) / layersPerGraph;
        this.layerITGs =
                IntStream.range(0, groups)
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
        logInitialization(
                "[jitllm] prefill fallback family: %d graphs, %s, all other buffers bound from the"
                        + " primary family%n",
                groups, tcAttention != null ? "tensor-core attention" : "JIT paged attention");
    }

    /**
     * Builds the fallback family on first request, or returns an empty list when there is nothing
     * to fall back from: a primary that is not using cuDNN attention already handles every chunk
     * itself, and the fallback family never has one of its own.
     */
    @Override
    public List<ImmutableTaskGraph> getFallbackLayerImmutableTaskGraphs() {
        if (!cudnnAttention || consumeFromPrimary) {
            return List.of();
        }
        if (fallbackFamily == null) {
            fallbackFamily = new Qwen3FP16LayersBatchPrefillMMA(this);
        }
        return fallbackFamily.layerITGs;
    }

    @Override
    public void updateFallbackGridScheduler(GridScheduler scheduler) {
        if (fallbackFamily != null) {
            fallbackFamily.updateGridScheduler(scheduler);
        }
    }

    // @formatter:off
    /**
     * One graph per <i>group</i> of layers. With {@code layersPerGraph == 1} this is exactly the
     * per-layer graph it has always been; above that, consecutive layers are appended to the same
     * graph in the same order, with the same tasks, the same buffers and the same arithmetic. What
     * changes is only how many graph submissions -- and therefore how many device syncs, since
     * TaskGraph.execute(...) ends with waitOn() -- a prefill pass costs.
     *
     * <p>The graph keeps the name of its FIRST layer, because the decode layer graphs bind their
     * weights from "batchPrefillLayer_&lt;i&gt;" by name and that name now has to resolve to the
     * graph that actually uploaded them.
     */
    private TaskGraph createBatchPrefillLayerTaskGraph(int groupIndex) {
        int firstLayer = groupIndex * layersPerGraph;
        int lastLayer = Math.min(firstLayer + layersPerGraph, config.numberOfLayers()) - 1;
        String graphName = graphPrefix + firstLayer;
        if (lastLayer == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        Object keyCache =
                useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;

        // ── Data Transfers ─────────────────────────────────────────────────────
        // The fallback family owns nothing. Every buffer it touches -- workspace, KV cache, block
        // table and the activation -- is bound from the primary graph covering the same layers, so
        // there is no second allocation and no copy between the two families. Declarations for
        // objects it does not actually use (w1/w3/wq/wk/wv under native projections) reach no task
        // and therefore emit no bytecode, exactly as they already do in the primary.
        String primaryGroup = GraphNames.batchPrefillLayer(firstLayer);
        if (consumeFromPrimary) {
            // The fallback family allocates nothing: every FIRST_EXECUTION buffer is bound from
            // the primary graph that owns it. But the two EVERY_EXECUTION uploads are not
            // ownership, they are how this chunk's start position, valid length, KV slot and block
            // table reach the device on every pass -- so this family performs them itself, exactly
            // as the primary's first graph does. Consuming them instead would leave a fallback
            // chunk reading the previous pass's position.
            //
            // Live state chains WITHIN this family, as it does within the primary: graph 0 takes
            // the activation from prefillActivation and the workspace from the primary's owning
            // graph, and every later graph takes both from the previous fallback graph.
            if (firstLayer == 0) {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
                batchPrefillLayer.consumeFromDevice(
                        primaryGroup,
                        context,
                        state.workspace.attnScaleBatch,
                        state.workspace.ffnScaleBatch,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.qkvResultBatch,
                        keyCache,
                        valueCache,
                        state.workspace.normedXFFNFP16,
                        state.workspace.gateUpResultBatch,
                        state.workspace.attnOutFP16,
                        state.workspace.woOut,
                        state.workspace.wrapHbFP16Batch,
                        state.workspace.w2Out);
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
                batchPrefillLayer.consumeFromDevice(
                        GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
                if (tcAttention != null) {
                    batchPrefillLayer.transferToDevice(
                            DataTransferMode.FIRST_EXECUTION, tcAttention.buffers());
                }
                if (nativeLibScratchContiguousProjections != null) {
                    batchPrefillLayer.consumeFromDevice(
                            primaryGroup, nativeLibScratchContiguousProjections.buffers());
                }
            } else {
                String prev = graphPrefix + (firstLayer - layersPerGraph);
                batchPrefillLayer.consumeFromDevice(
                        prev,
                        context,
                        state.workspace.wrapXBatch,
                        state.workspace.batchStartPosHolder,
                        state.workspace.attnScaleBatch,
                        state.workspace.ffnScaleBatch,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.qkvResultBatch,
                        keyCache,
                        valueCache,
                        state.workspace.normedXFFNFP16,
                        state.workspace.gateUpResultBatch,
                        state.workspace.attnOutFP16,
                        state.workspace.woOut,
                        state.workspace.wrapHbFP16Batch,
                        state.workspace.w2Out);
                batchPrefillLayer.consumeFromDevice(prev, state.workspace.wrapBlockTable);
                if (tcAttention != null) {
                    batchPrefillLayer.consumeFromDevice(prev, tcAttention.buffers());
                }
                if (nativeLibScratchContiguousProjections != null) {
                    batchPrefillLayer.consumeFromDevice(
                            prev, nativeLibScratchContiguousProjections.buffers());
                }
            }
        } else if (firstLayer == 0) {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
            if (useCudnnAttention) {
                // The one allocation of the staging quartet: scratch, private to this execution
                // plan, owned by the first batch-prefill graph and bound by every later one.
                // Declaring transferToDevice for the same array in N graphs would give it N
                // device buffers -- measured, and the only objects that were ever allocated more
                // than once across these graphs.
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, cudnnQ, cudnnK, cudnnV, cudnnOut);
            }
            if (!nativeProjections) {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, splitK.partial());
            }
            if (nativeLibScratchContiguousProjections != null) {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        nativeLibScratchContiguousProjections.buffers());
            }
            if (tcAttention != null) {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, tcAttention.buffers());
            }
        } else {
            String pred = graphPrefix + (firstLayer - layersPerGraph);
            batchPrefillLayer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyCache,
                    valueCache,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            batchPrefillLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
            if (useCudnnAttention) {
                batchPrefillLayer.consumeFromDevice(pred, cudnnQ, cudnnK, cudnnV, cudnnOut);
            }
            if (!nativeProjections) {
                batchPrefillLayer.consumeFromDevice(pred, splitK.partial());
            }
            if (nativeLibScratchContiguousProjections != null) {
                batchPrefillLayer.consumeFromDevice(
                        pred, nativeLibScratchContiguousProjections.buffers());
            }
            if (tcAttention != null) {
                batchPrefillLayer.consumeFromDevice(pred, tcAttention.buffers());
            }
        }

        for (int layerIndex = firstLayer; layerIndex <= lastLayer; layerIndex++) {
            appendBatchPrefillLayer(batchPrefillLayer, layerIndex, dim, hidDim);
        }

        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache, valueCache);

        return batchPrefillLayer;
    }

    /**
     * Appends one transformer layer's tasks to {@code batchPrefillLayer}. Task ids carry a
     * per-layer prefix only when a graph holds more than one layer, so the single-layer graphs keep
     * the ids they always had.
     */
    private void appendBatchPrefillLayer(
            TaskGraph batchPrefillLayer, int layerIndex, int dim, int hidDim) {
        String tp = taskPrefix(layerIndex);
        // Per-layer weights: upload once
        if (consumeFromPrimary) {
            batchPrefillLayer.consumeFromDevice(
                    primaryGraphOwning(layerIndex),
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                    weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray());
        } else {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                    weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray());
        }
        if (nativeLibResidentContiguousQkv.length > 0) {
            // Buffers this class owns, not the model's, so they need their own upload. The five
            // originals stay declared above: the decode layer graphs consume this graph's weight
            // buffers, and Qwen3FP16FFNLayersDecode uploads the ones this graph no longer reads.
            if (consumeFromPrimary) {
                batchPrefillLayer.consumeFromDevice(
                        primaryGraphOwning(layerIndex),
                        nativeLibResidentContiguousGateUp[layerIndex],
                        nativeLibResidentContiguousQkv[layerIndex]);
            } else {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        nativeLibResidentContiguousGateUp[layerIndex],
                        nativeLibResidentContiguousQkv[layerIndex]);
            }
        }

        // ── Attention Block ────────────────────────────────────────────────────
        batchPrefillLayer.task(
                tp + "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                tp + "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        // Q, K, V in ONE tensor-core launch → packed [q|k|v] rows (stride qDim+2*kvDim)
        // QKV: [M=batch, N=qDim+2*kvDim, K=dim], producing the packed [q|k|v] rows that
        // batch_qk_rmsnorm and batch_rope_kv read at stride qDim+2*kvDim. One GEMM over the
        // contiguous copy reproduces that layout exactly -- output column c < qDim comes from a wq
        // row, c < qDim+kvDim from a wk row, the rest from wv -- so the packing survives the swap
        // and the normalization, RoPE and cache-write tasks downstream are untouched. Same
        // contract as the other native projections: FP16 operands, FP32 accumulation, FP32
        // qkvResultBatch.
        plan.emit(batchPrefillLayer, regions.get(layerIndex).qkv(), NO_OPERANDS, bindings);

        // Qwen3: per-head RMS norm on Q and K before RoPE
        batchPrefillLayer.task(
                tp + "batch_qk_rmsnorm",
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

        // Register-partitioned flash attention over the packed buffer.
        // The 'dim' parameter doubles as the packed-Q stride base and the
        // attnOutFP16 row width — both are qDim for Qwen3.
        if (useFp16KVCache()) {
            batchPrefillLayer.task(
                    tp + "batch_rope_kv",
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
        } else {
            batchPrefillLayer.task(
                    tp + "batch_rope_kv",
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
        }
        plan.emit(batchPrefillLayer, regions.get(layerIndex).attention(), NO_OPERANDS, bindings);

        // Output projection: [M=batch, N=dim, K=qDim]
        plan.emit(batchPrefillLayer, regions.get(layerIndex).output(), NO_OPERANDS, bindings);

        // ── FFN Block ──────────────────────────────────────────────────────────
        batchPrefillLayer.task(
                tp + "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        batchPrefillLayer.task(
                tp + "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        // Gate/up: [M=batch, N=2*hidDim, K=dim], producing the packed [gate|up] rows the SwiGLU
        // kernel reads as rowBase+i and rowBase+hidDim+i. One GEMM over the contiguous copy
        // reproduces that layout exactly -- output column c < hidDim comes from a gate row,
        // c >= hidDim from an up row -- so the fusion is preserved rather than split into two
        // calls. Same contract as woProj/w2Proj: FP16 operands, FP32 accumulation, FP32 output.
        plan.emit(batchPrefillLayer, regions.get(layerIndex).gateUp(), NO_OPERANDS, bindings);

        batchPrefillLayer.task(
                tp + "swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                hidDim);
        // Down projection: [M=batch, N=dim, K=hidDim]. Same contract as woProj -- FP16
        // operands, FP32 accumulation, FP32 w2Out -- and w2Resid still follows it, so the
        // residual is added after the projection exactly as before.
        plan.emit(batchPrefillLayer, regions.get(layerIndex).down(), NO_OPERANDS, bindings);
        batchPrefillLayer.task(
                tp + "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);
    }

    // @formatter:on

    @Override
    public boolean cublasForm(Step step) {
        return true;
    }

    @Override
    public void project(TaskGraph graph, Step step, Operands operands, boolean cublas) {
        int layerIndex = step.layer();
        switch (step.first(MatMul.class).weight().role()) {
            case ATTENTION_QKV -> qkvProjection(graph, layerIndex, cublas);
            case ATTENTION_OUTPUT -> outputProjection(graph, layerIndex, cublas);
            case FFN_GATE_UP -> gateUpProjection(graph, layerIndex, cublas);
            case FFN_DOWN -> downProjection(graph, layerIndex, cublas);
            default ->
                    throw new IllegalArgumentException("not a projection of this layer: " + step);
        }
    }

    /**
     * Whether {@code step}'s attention has a cuDNN form: the primary family's, with an FP16
     * key/value cache, where cuDNN takes this attention shape. The fallback family has none.
     */
    @Override
    public boolean cudnnForm(Step step) {
        return state.usesFp16KeyValueCache()
                && !consumeFromPrimary
                && NativePrefillSupport.nativeAttention(state.executionPolicy(), sdpaShape);
    }

    @Override
    public void attend(TaskGraph batchPrefillLayer, Step step, boolean cudnn) {
        int layerIndex = step.layer();
        String tp = taskPrefix(layerIndex);
        if (useFp16KVCache()) {
            if (cudnn) {
                // Q from the packed FP32 QKV buffer, K/V gathered out of the paged cache with
                // the 16:8 group expansion cuDNN's single head count cannot express, then the
                // library call, then the output back into attnOutFP16's [tok][qDim] layout.
                // Same graph as the surrounding JIT tasks: a library task does observe their
                // writes, so no extra graph boundary is needed. The staging quartet is declared
                // once, by the graph that owns it, above.
                batchPrefillLayer.task(
                        tp + "cudnn_pack_q",
                        CuDnnPrefillAttentionKernels::packQ,
                        state.workspace.batchStartPosHolder,
                        state.workspace.qkvResultBatch,
                        cudnnQ,
                        nEmbdHead,
                        batchSize,
                        qDim + 2 * kvDim);
                batchPrefillLayer.task(
                        tp + "cudnn_gather_kv",
                        CuDnnPrefillAttentionKernels::gatherKvExpanded,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapBlockTable,
                        cudnnK,
                        cudnnV,
                        nEmbdHead,
                        batchSize,
                        kvDim,
                        gqa,
                        layerIndex,
                        state.kvBlockCfg,
                        state.kvBlockStride);
                batchPrefillLayer.libraryTask(
                        tp + "cudnn_sdpa",
                        CuDnn::sdpaForward,
                        cudnnQ,
                        cudnnK,
                        cudnnV,
                        cudnnOut,
                        sdpaShape.batch(),
                        sdpaShape.heads(),
                        sdpaShape.seqQ(),
                        sdpaShape.seqKv(),
                        sdpaShape.headDim(),
                        sdpaShape.scale(),
                        sdpaShape.causal());
                batchPrefillLayer.task(
                        tp + "cudnn_scatter",
                        CuDnnPrefillAttentionKernels::scatterAttnOut,
                        cudnnOut,
                        state.workspace.attnOutFP16,
                        nEmbdHead,
                        batchSize,
                        qDim);
            } else if (tcAttention != null) {
                tcAttention.appendTasks(batchPrefillLayer, context, layerIndex, tp);
            } else {
                batchPrefillLayer.task(
                        tp + "batch_attention",
                        packedHalf2Attention
                                ? TransformerPagedKvBatchPrefillKernels
                                        ::batchedFlashAttentionFP16OutKVFP16PackedTilePaged
                                : TransformerPagedKvBatchPrefillKernels
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
                    tp + "batch_attention",
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
    }

    private void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex, boolean cublas) {
        String tp = taskPrefix(layerIndex);
        int dim = config.dim();
        if (cublas && nativeLibScratchContiguousProjections != null) {
            nativeLibScratchContiguousProjections.project(
                    batchPrefillLayer,
                    tp + "qkvProj",
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    nativeGemmRows(),
                    dim,
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex]);
        } else if (cublas) {
            batchPrefillLayer.libraryTask(
                    tp + "qkvProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    qDim + 2 * kvDim,
                    nativeGemmRows(),
                    dim,
                    1.0f,
                    nativeLibResidentContiguousQkv[layerIndex],
                    dim,
                    state.workspace.wrapXbFP16Batch,
                    dim,
                    0.0f,
                    state.workspace.qkvResultBatch,
                    qDim + 2 * kvDim);
        } else {
            batchPrefillLayer.task(
                    tp + "qkvProj",
                    TransformerBatchPrefillKernels::gemmMMAQKV,
                    context,
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    paddedBatch,
                    qDim,
                    kvDim,
                    dim);
        }
    }

    private void outputProjection(TaskGraph batchPrefillLayer, int layerIndex, boolean cublas) {
        String tp = taskPrefix(layerIndex);
        int dim = config.dim();
        if (cublas) {
            // gemmMMA computes C[m][n] = sum_k A[m][k] * B[n][k] with A and B both row-major
            // and B already stored transposed, i.e. C = A * B^T. cuBLAS is column-major, where
            // that same buffer layout reads as C' = B'^T * A' -- hence OP_T on the weight,
            // OP_N on the activation, and (m,n) swapped to (dim, batch).
            // FP16 in, FP32 out, FP32 accumulate: same accumulation precision and the same
            // woOut representation the fused RMS/residual consumer expects.
            batchPrefillLayer.libraryTask(
                    tp + "woProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    dim,
                    nativeGemmRows(),
                    qDim,
                    1.0f,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    qDim,
                    state.workspace.attnOutFP16,
                    qDim,
                    0.0f,
                    state.workspace.woOut,
                    dim);
        } else {
            splitK.gemm(
                    batchPrefillLayer,
                    tp + "woProj",
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    qDim);
        }
    }

    private void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex, boolean cublas) {
        String tp = taskPrefix(layerIndex);
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        if (cublas && nativeLibScratchContiguousProjections != null) {
            nativeLibScratchContiguousProjections.project(
                    batchPrefillLayer,
                    tp + "gateUpProj",
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    nativeGemmRows(),
                    dim,
                    weights.w1Layered[layerIndex],
                    weights.w3Layered[layerIndex]);
        } else if (cublas) {
            batchPrefillLayer.libraryTask(
                    tp + "gateUpProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    2 * hidDim,
                    nativeGemmRows(),
                    dim,
                    1.0f,
                    nativeLibResidentContiguousGateUp[layerIndex],
                    dim,
                    state.workspace.normedXFFNFP16,
                    dim,
                    0.0f,
                    state.workspace.gateUpResultBatch,
                    2 * hidDim);
        } else {
            batchPrefillLayer.task(
                    tp + "gateUpProj",
                    TransformerBatchPrefillKernels::gemmMMAGateUp,
                    context,
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    paddedBatch,
                    hidDim,
                    dim);
        }
    }

    private void downProjection(TaskGraph batchPrefillLayer, int layerIndex, boolean cublas) {
        String tp = taskPrefix(layerIndex);
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        if (cublas) {
            batchPrefillLayer.libraryTask(
                    tp + "w2Proj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    dim,
                    nativeGemmRows(),
                    hidDim,
                    1.0f,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    hidDim,
                    state.workspace.wrapHbFP16Batch,
                    hidDim,
                    0.0f,
                    state.workspace.w2Out,
                    dim);
        } else {
            splitK.gemm(
                    batchPrefillLayer,
                    tp + "w2Proj",
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    paddedBatch,
                    dim,
                    hidDim);
        }
    }

    /** The plan over {@link #regions}: cuBLAS and cuDNN first, then the JIT kernels. */
    private WorkPlan plan() {
        List<Region> all = new ArrayList<>();
        for (LayerRegions r : regions) {
            all.addAll(List.of(r.qkv(), r.attention(), r.output(), r.gateUp(), r.down()));
        }
        return WorkPlan.of(
                all,
                List.of(
                        new CublasLayerProjections(this),
                        new CudnnLayerAttention(this),
                        new JitLayerProjections(this),
                        new JitLayerAttention(this)),
                Target.of(state.executionPolicy()),
                context,
                state.workspace.prepacking);
    }

    /** Every layer's regions: q|k|v, the attention, the output projection, gate|up and down. */
    private static List<LayerRegions> regions(Qwen3Configuration config, int batchSize) {
        int dim = config.dim();
        int head = config.numberOfHeadsValue();
        int qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        int kvDim = head * config.numberOfKeyValueHeads();
        int hidden = config.hiddenDim();
        List<LayerRegions> layers = new ArrayList<>(config.numberOfLayers());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            Attention attention =
                    new Attention(
                            new OperandRef.Activation("qkv"),
                            new OperandRef.Activation("qkv"),
                            new OperandRef.Activation("qkv"),
                            new OperandRef.Activation("attn"),
                            config.numberOfHeads(),
                            config.numberOfKeyValueHeads(),
                            head,
                            (float) (1.0 / Math.sqrt(head)),
                            OptionalInt.empty(),
                            DataType.F16);
            layers.add(
                    new LayerRegions(
                            projection(
                                    i,
                                    batchSize,
                                    "q|k|v",
                                    TensorRole.ATTENTION_QKV,
                                    "attnIn",
                                    "qkv",
                                    qDim + 2 * kvDim,
                                    dim),
                            new Region(i, "attention", batchSize, List.of(attention)),
                            projection(
                                    i,
                                    batchSize,
                                    "attention output",
                                    TensorRole.ATTENTION_OUTPUT,
                                    "attn",
                                    "o",
                                    dim,
                                    qDim),
                            projection(
                                    i,
                                    batchSize,
                                    "gate|up",
                                    TensorRole.FFN_GATE_UP,
                                    "ffnIn",
                                    "gateUp",
                                    2 * hidden,
                                    dim),
                            projection(
                                    i,
                                    batchSize,
                                    "down",
                                    TensorRole.FFN_DOWN,
                                    "hidden",
                                    "down",
                                    dim,
                                    hidden)));
        }
        return layers;
    }

    private static Region projection(
            int layer,
            int tokens,
            String name,
            TensorRole role,
            String input,
            String output,
            int rows,
            int columns) {
        return new Region(
                layer,
                name,
                tokens,
                List.of(
                        new MatMul(
                                new OperandRef.Weight(
                                        role.name().toLowerCase(java.util.Locale.ROOT), role),
                                new OperandRef.Activation(input),
                                new OperandRef.Activation(output),
                                rows,
                                columns,
                                tokens,
                                DataType.F16)));
    }

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128; // BM
        int nBlocks = N / 128; // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** The PRIMARY family's graph that owns this layer's weights and their copies. */
    private String primaryGraphOwning(int layerIndex) {
        return GraphNames.batchPrefillLayer((layerIndex / layersPerGraph) * layersPerGraph);
    }

    /** Empty while a graph holds one layer, so the existing ids are untouched. */
    private String taskPrefix(int layerIndex) {
        return layersPerGraph == 1 ? "" : "L" + layerIndex + "_";
    }

    static WorkerGrid elementwiseGrid(int n) { // n must be a multiple of 256
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();

        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);

        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

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

        // MMA grids
        WorkerGrid mmaQkvWorker = mmaGrid(paddedBatch, qDim + 2 * kvDim); // fused QKV
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim); // woProj, w2Proj
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim); // fused W1/W3

        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim); // w2Resid
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim); // swiglu

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = graphPrefix + (i / layersPerGraph) * layersPerGraph + "." + taskPrefix(i);
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            if (!nativeProjections) {
                scheduler.addWorkerGrid(p + "qkvProj", mmaQkvWorker);
            }
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (useCudnnAttention) {
                WorkerGrid cudnnWorker = elementwiseGrid(qDim * batchSize);
                scheduler.addWorkerGrid(p + "cudnn_pack_q", cudnnWorker);
                scheduler.addWorkerGrid(p + "cudnn_gather_kv", cudnnWorker);
                scheduler.addWorkerGrid(p + "cudnn_scatter", cudnnWorker);
            } else if (tcAttention != null) {
                tcAttention.addGrids(
                        scheduler,
                        graphPrefix + (i / layersPerGraph) * layersPerGraph,
                        taskPrefix(i));
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            if (!nativeProjections) {
                scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", ffnRmsApplyWorker);
            if (!nativeProjections) {
                scheduler.addWorkerGrid(p + "gateUpProj", mmaGateUpWorker);
            }
            scheduler.addWorkerGrid(p + "swiglu", ewHidWorker);
            if (!nativeProjections) {
                scheduler.addWorkerGrid(p + "w2Proj", mmaDimWorker);
            }
            scheduler.addWorkerGrid(p + "w2Resid", ewDimWorker);
        }
        if (!nativeProjections) {
            splitK.addGrids(scheduler);
        }
        if (nativeLibScratchContiguousProjections != null) {
            // The refill tasks' grids; the GEMMs are cuBLAS calls, which take none.
            nativeLibScratchContiguousProjections.addGrids(scheduler);
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
