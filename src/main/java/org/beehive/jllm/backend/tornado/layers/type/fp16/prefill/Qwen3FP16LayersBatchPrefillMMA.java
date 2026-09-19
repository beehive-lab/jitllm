package org.beehive.jllm.backend.tornado.layers.type.fp16.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jllm.backend.tornado.NativePrefillSupport;
import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.backend.tornado.kernels.CuDnnPrefillAttentionKernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
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
public class Qwen3FP16LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    private final Qwen3State state;
    private final boolean packedHalf2Attention;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
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
     * Whether this family's four projections are cuBLAS GEMMs over stacked weights. One field, not
     * one per projection: {@link NativePrefillSupport} answers for all four together, because a
     * configuration with only some of them native has never been validated end to end -- and the
     * decode side derives which weights it must upload itself from the same single answer.
     */
    private final boolean nativeProjections;

    /** Whether this PLAN uses cuDNN attention anywhere; see {@link #useCudnnAttention}. */
    private final boolean cudnnAttention;

    /**
     * gate and up stacked along the output dimension, one array per layer, so the packed [gate|up]
     * result can come from a single GEMM. gemmMMAGateUp reads w1 and w3 as two separate operands
     * and selects between them by output column; cuBLAS takes one B and the binding has no operand
     * offset, so the two have to be adjacent in memory. Built once here, never per prefill, and the
     * originals are left alone because decode and the JIT path still use them.
     */
    private final HalfFloatArray[] gateUpCat;

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

    /** cuDNN attention in THIS family. False in the fallback family whatever the flag says. */
    private final boolean useCudnnAttention;

    /**
     * wq, wk and wv stacked along the output dimension, one array per layer, so the packed [q|k|v]
     * result can come from a single GEMM. gemmMMAQKV reads the three as separate operands and
     * selects between them by output column -- [0,qDim) from wq, [qDim,qDim+kvDim) from wk, the
     * rest from wv, each indexed row-major [outCol][K]. The slices are NOT equal: for this model
     * 2048 | 1024 | 1024. cuBLAS takes one B and the binding has no operand offset, so the three
     * have to be adjacent in memory, in that order. Built once here, never per prefill, and the
     * originals are left alone because decode and the JIT path still read them.
     */
    private final HalfFloatArray[] qkvCat;

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
        return "batchPrefillLayer_" + (layerIndex / g) * g;
    }

    /**
     * Whether the batch-prefill projections are single stacked cuBLAS GEMMs.
     *
     * <p><b>Single source of truth, and the reason it is a static.</b> The decode layer graphs
     * consume this layer's weights from the batch-prefill graph, and this answer decides whether
     * that graph still reads {@code wq}/{@code wk}/{@code wv} and {@code w1}/{@code w3} at all.
     * TornadoVM builds a graph from its tasks' argument lists, so a weight the producer declares
     * but hands to no task is never uploaded and the consumer binds a buffer nothing wrote. Both
     * sides ask this one method; see {@code Qwen3FP16FFNLayersDecode.weightsNotProvidedBySource}.
     */
    public static boolean nativeProjections() {
        return NativePrefillSupport.nativeProjections();
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
        if (batchSize % 128 != 0) {
            System.out.printf(
                    "[jllm] prefill batch %d padded to %d for tensor-core tiles; GEMM efficiency is"
                            + " %d/%d — use a multiple of 128 for best throughput.%n",
                    batchSize, paddedBatch, batchSize, paddedBatch);
        }
        // The FP16 key/value cache is part of the supported configuration, not an extra switch:
        // the cuDNN adapters read K/V straight out of it as FP16, and the gather that would be
        // needed from an FP32 cache has never been written or validated. An FP32 KV session on
        // this device keeps the JIT attention and loses nothing else -- the projections stay
        // native, because they do not touch the cache.
        this.cudnnAttention =
                NativePrefillSupport.nativeAttention() && state.usesFp16KeyValueCache();
        this.useCudnnAttention = this.cudnnAttention;
        this.nativeProjections = nativeProjections();
        this.layersPerGraph = Math.min(prefillLayersPerGraph(), config.numberOfLayers());
        System.out.printf("[jllm] prefill acceleration: %s%n", NativePrefillSupport.describe());
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        this.kvDim = config.numberOfHeadsValue() * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
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
            System.out.printf(
                    "[jllm] prefill attention: cuDNN SDPA for the first chunk, batched JIT paged"
                            + " attention for the rest; staging %d MiB, one set shared by every"
                            + " batch-prefill graph%n",
                    4L * cudnnElems * Short.BYTES / (1024 * 1024));
        }
        this.gateUpCat = new HalfFloatArray[nativeProjections ? config.numberOfLayers() : 0];
        if (nativeProjections) {
            long t0 = System.nanoTime();
            int rowsK = config.hiddenDim() * config.dim();
            for (int l = 0; l < config.numberOfLayers(); l++) {
                HalfFloatArray w1 = weights.w1Layered[l].asHalfFloatArray();
                HalfFloatArray w3 = weights.w3Layered[l].asHalfFloatArray();
                HalfFloatArray cat = new HalfFloatArray(2 * rowsK);
                for (int i = 0; i < rowsK; i++) {
                    cat.set(i, w1.get(i));
                    cat.set(rowsK + i, w3.get(i));
                }
                gateUpCat[l] = cat;
            }
            System.out.printf(
                    "[jllm] prefill gate/up: cuBLAS, %d stacked weights, %d MiB extra resident,"
                            + " built in %.2f s%n",
                    config.numberOfLayers(),
                    (long) config.numberOfLayers() * 2 * rowsK * Short.BYTES / (1024 * 1024),
                    (System.nanoTime() - t0) / 1e9);
        }
        this.qkvCat = new HalfFloatArray[nativeProjections ? config.numberOfLayers() : 0];
        if (nativeProjections) {
            long t0 = System.nanoTime();
            for (int l = 0; l < config.numberOfLayers(); l++) {
                HalfFloatArray wq = weights.wqLayered[l].asHalfFloatArray();
                HalfFloatArray wk = weights.wkLayered[l].asHalfFloatArray();
                HalfFloatArray wv = weights.wvLayered[l].asHalfFloatArray();
                HalfFloatArray cat = new HalfFloatArray(wq.getSize() + wk.getSize() + wv.getSize());
                int at = 0;
                for (int i = 0; i < wq.getSize(); i++) cat.set(at++, wq.get(i));
                for (int i = 0; i < wk.getSize(); i++) cat.set(at++, wk.get(i));
                for (int i = 0; i < wv.getSize(); i++) cat.set(at++, wv.get(i));
                qkvCat[l] = cat;
            }
            System.out.printf(
                    "[jllm] prefill QKV: cuBLAS, %d stacked weights (%d|%d|%d cols), %d MiB extra"
                            + " resident, built in %.2f s%n",
                    config.numberOfLayers(),
                    qDim,
                    kvDim,
                    kvDim,
                    (long) config.numberOfLayers()
                            * qkvCat[0].getSize()
                            * Short.BYTES
                            / (1024 * 1024),
                    (System.nanoTime() - t0) / 1e9);
        }
        int groups = (config.numberOfLayers() + layersPerGraph - 1) / layersPerGraph;
        if (nativeProjections && batchSize != paddedBatch) {
            System.out.printf(
                    "[jllm] prefill native GEMM rows: %d (chunk width) instead of %d (padded)%n",
                    batchSize, paddedBatch);
        }
        if (layersPerGraph > 1) {
            System.out.printf(
                    "[jllm] prefill layer grouping: %d layers per graph, %d graphs instead of %d%n",
                    layersPerGraph, groups, config.numberOfLayers());
        }
        this.layerITGs =
                IntStream.range(0, groups)
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    /**
     * The fallback family: the same layer pipeline with the JIT paged attention instead of the
     * cuDNN call, for a chunk whose queries do not start at position 0.
     *
     * <p>It shares every array with {@code primary} -- the model's weights, the stacked [q|k|v] and
     * [gate|up] weights, the workspace, the KV cache -- and declares all of them with {@code
     * consumeFromDevice} against the primary's graphs, so nothing is allocated, uploaded or copied
     * a second time. It does not touch the cuDNN staging quartet at all, because it does not call
     * cuDNN.
     *
     * <p>Only the attention step differs. QKV, gate/up, woProj and w2Proj stay native, with the
     * same stacked weights and the same numerical contracts already validated for them.
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
        this.gateUpCat = primary.gateUpCat;
        this.qkvCat = primary.qkvCat;
        int groups = (config.numberOfLayers() + layersPerGraph - 1) / layersPerGraph;
        this.layerITGs =
                IntStream.range(0, groups)
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
        System.out.printf(
                "[jllm] prefill fallback family: %d graphs, JIT paged attention, all buffers bound"
                        + " from the primary family%n",
                groups);
    }

    /**
     * Builds the fallback family for {@code primary}, or {@code null} when there is nothing to fall
     * back from -- if the primary is not using cuDNN attention it already handles every chunk.
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
        String primaryGroup = "batchPrefillLayer_" + firstLayer;
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
                        "prefillActivation", state.workspace.wrapXBatch);
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
            batchPrefillLayer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            if (useCudnnAttention) {
                // The one allocation of the staging quartet: scratch, private to this execution
                // plan, owned by the first batch-prefill graph and bound by every later one.
                // Declaring transferToDevice for the same array in N graphs would give it N
                // device buffers -- measured, and the only objects that were ever allocated more
                // than once across these graphs.
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, cudnnQ, cudnnK, cudnnV, cudnnOut);
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
        if (nativeProjections) {
            // A buffer this class owns, not one of the model's, so it needs its own upload. w1/w3
            // stay declared above: the decode layer graphs consume this graph's weight buffers,
            // and Qwen3FP16FFNLayersDecode uploads the two this graph no longer reads.
            if (consumeFromPrimary) {
                batchPrefillLayer.consumeFromDevice(
                        primaryGraphOwning(layerIndex), gateUpCat[layerIndex]);
            } else {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, gateUpCat[layerIndex]);
            }
        }
        if (nativeProjections) {
            // Same arrangement for QKV: wq/wk/wv stay declared above and Qwen3FP16FFNLayersDecode
            // uploads the three this graph no longer reads.
            if (consumeFromPrimary) {
                batchPrefillLayer.consumeFromDevice(
                        primaryGraphOwning(layerIndex), qkvCat[layerIndex]);
            } else {
                batchPrefillLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, qkvCat[layerIndex]);
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
        // stacked weight reproduces that layout exactly -- output column c < qDim comes from a wq
        // row, c < qDim+kvDim from a wk row, the rest from wv -- so the packing survives the swap
        // and the normalization, RoPE and cache-write tasks downstream are untouched. Same
        // contract as the other native projections: FP16 operands, FP32 accumulation, FP32
        // qkvResultBatch.
        if (nativeProjections) {
            batchPrefillLayer.libraryTask(
                    tp + "qkvProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    qDim + 2 * kvDim,
                    nativeGemmRows(),
                    dim,
                    1.0f,
                    qkvCat[layerIndex],
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

        // Qwen3: per-head RMS norm on Q and K before RoPE
        batchPrefillLayer.task(
                tp + "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNormPacked,
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

            if (useCudnnAttention) {
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
                        1,
                        config.numberOfHeads(),
                        batchSize,
                        batchSize,
                        nEmbdHead,
                        (float) (1.0 / Math.sqrt(nEmbdHead)),
                        true);
                batchPrefillLayer.task(
                        tp + "cudnn_scatter",
                        CuDnnPrefillAttentionKernels::scatterAttnOut,
                        cudnnOut,
                        state.workspace.attnOutFP16,
                        nEmbdHead,
                        batchSize,
                        qDim);
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

        // Output projection: [M=batch, N=dim, K=qDim]
        if (nativeProjections) {
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
            batchPrefillLayer.task(
                    tp + "woProj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    qDim);
        }

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
        // kernel reads as rowBase+i and rowBase+hidDim+i. One GEMM over the stacked weight
        // reproduces that layout exactly -- output column c < hidDim comes from a gate row,
        // c >= hidDim from an up row -- so the fusion is preserved rather than split into two
        // calls. Same contract as woProj/w2Proj: FP16 operands, FP32 accumulation, FP32 output.
        if (nativeProjections) {
            batchPrefillLayer.libraryTask(
                    tp + "gateUpProj",
                    CuBlas::cublasGemmExFP16FP32,
                    1,
                    0,
                    2 * hidDim,
                    nativeGemmRows(),
                    dim,
                    1.0f,
                    gateUpCat[layerIndex],
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
        if (nativeProjections) {
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
            batchPrefillLayer.task(
                    tp + "w2Proj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    paddedBatch,
                    dim,
                    hidDim);
        }
        batchPrefillLayer.task(
                tp + "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);
    }

    // @formatter:on

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128; // BM
        int nBlocks = N / 128; // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** The PRIMARY family's graph that owns this layer's weights and stacked weights. */
    private String primaryGraphOwning(int layerIndex) {
        return "batchPrefillLayer_" + (layerIndex / layersPerGraph) * layersPerGraph;
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

        // Q/K per-head RMS norm: one nEmbdHead-thread workgroup per (token, head)
        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (nHeads + nHeadKv) * nEmbdHead, nEmbdHead);

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
