package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.Glm5NextBatchArrays;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.weights.Glm5NextLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.Glm5NextTornadoWeights;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The batched-prefill {@code glm5next} layers: the whole trunk over one chunk of tokens, in one
 * task graph, the same steps as {@link Glm5NextLayers} with a row per token.
 *
 * <p>Everything a token does alone — the hyper-connections, the norms, the projections, the
 * indexer's keys and scores, the feed-forward — runs over every row at once. What a token does
 * after the tokens before it — the KDA convolution and delta rule, and attention over the cache —
 * either walks the rows in order inside one launch or reads the cache the chunk has just written up
 * to the row's own position.
 *
 * <p>The latent cache, the KDA windows and matrices and the indexer's caches are the decode plan's
 * own arrays, kept on the device after the chunk; the decode layers take them, and every weight,
 * from this graph by name.
 */
// @formatter:on
public class Glm5NextBatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    public static final String GRAPH = "glm5nextPrefill";

    private final Glm5NextState gs;
    private final Glm5NextTornadoWeights weights;
    private final Glm5NextConfiguration config;
    private final Glm5NextLayerWeights<TornadoTensor> w;
    private final int batch;
    private final KernelContext context = new KernelContext();
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();
    private final ImmutableTaskGraph graph;

    public Glm5NextBatchPrefillLayers(
            Glm5NextState state,
            Glm5NextTornadoWeights weights,
            Glm5NextConfiguration config,
            int batch) {
        if (state.workspace.glm5nextBatch == null) {
            throw new IllegalStateException(
                    "the glm5next batched prefill needs a state built for batch " + batch);
        }
        this.gs = state;
        this.weights = weights;
        this.config = config;
        this.w = weights.layers;
        this.batch = batch;
        int[] shape = {
            config.dim(),
            config.expertHiddenDim(),
            config.sharedHiddenDim(),
            config.expertsUsed(),
            Math.min(config.expertChunkSize(), config.expertCount()),
            batch
        };
        for (int i = 0; i < shape.length; i++) {
            state.workspace.glm5nextBatch.expertShape.set(i, shape[i]);
        }
        TaskGraph g = new TaskGraph(GRAPH);
        bind(g);
        var ws = state.workspace;
        task(
                g,
                "broadcast",
                lanes(batch * config.hcCount() * config.dim(), 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcBroadcast,
                                context,
                                ws.wrapXBatch,
                                ws.glm5nextBatch.streams,
                                config.dim(),
                                config.hcCount(),
                                batch));
        for (int l = 0; l < config.numberOfLayers(); l++) {
            appendLayer(g, l);
        }
        g.persistOnDevice(persistent());
        this.graph = g.snapshot();
    }

    // ── binding ───────────────────────────────────────────────────────────────

    /** What the decode layers continue from: the caches and the recurrent state. */
    public Object[] persistent() {
        var ws = gs.workspace;
        var d = ws.glm5next;
        return new Object[] {
            ws.wrapKeyCache,
            ws.wrapConvState,
            ws.wrapDeltaState,
            d.indexKeyCache,
            d.poolGateCache,
            d.poolKeys
        };
    }

    /** Every weight this graph binds, which the decode layers take from it by name. */
    public static List<Object> weightArrays(
            Glm5NextLayerWeights<TornadoTensor> w, Glm5NextConfiguration config) {
        List<Object> arrays = new ArrayList<>();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            arrays.addAll(Glm5NextLayers.layerArrays(w, config, l));
        }
        return arrays;
    }

    private void bind(TaskGraph g) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        g.consumeFromDevice("prefillActivation", ws.wrapXBatch);
        g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
        g.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                ws.glm5next.noRope,
                b.streams,
                b.residual,
                b.mixes,
                b.coeffs,
                b.normed,
                b.quants,
                b.scales,
                b.sums,
                b.a8,
                b.a8Scales,
                b.branch,
                b.kdaQ,
                b.kdaK,
                b.kdaV,
                b.kdaConvQ,
                b.kdaConvK,
                b.kdaConvV,
                b.kdaDecayLow,
                b.kdaGateLow,
                b.kdaDecay,
                b.kdaBeta,
                b.kdaGate,
                b.kdaOut,
                b.queryLatent,
                b.kvLatent,
                b.query,
                b.absorbed,
                b.latentOut,
                b.attnOut,
                b.indexRawKey,
                b.indexGate,
                b.indexQuery,
                b.indexHeadWeights,
                b.poolScores,
                b.selected,
                b.hidden,
                b.hidden2,
                b.routerLogits,
                b.expertIds,
                b.expertWeights,
                b.moeHidden,
                b.moeQuants,
                b.moeScales,
                b.moeSums,
                b.expertShape,
                b.sortedToken,
                b.position,
                b.tiles,
                b.routedHidden,
                b.routedHidden8,
                b.routedHiddenScales,
                b.routedOut,
                b.sharedGate,
                b.sharedUp,
                b.sharedOut);
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, persistent());
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, weightArrays(w, config).toArray());
    }

    private void task(TaskGraph g, String name, WorkerGrid grid, TaskAdder adder) {
        adder.add(name);
        grids.put(GRAPH + "." + name, grid);
    }

    @FunctionalInterface
    private interface TaskAdder {
        void add(String taskName);
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var m = w.mhc();
        String p = "l" + l + "_";
        hyperConnectionPre(g, p + "hc_attn", m.attnFn()[l], m.attnBase()[l], m.attnScale()[l]);
        norm(g, p + "attn_norm", w.attnNorm()[l]);
        if (config.isKda(l)) {
            kda(g, p, l);
        } else {
            sparseAttention(g, p, l);
        }
        hyperConnectionPost(g, p + "hc_attn");
        hyperConnectionPre(g, p + "hc_ffn", m.ffnFn()[l], m.ffnBase()[l], m.ffnScale()[l]);
        norm(g, p + "ffn_norm", w.ffnNorm()[l]);
        if (config.isMoe(l)) {
            mixtureOfExperts(g, p, l);
        } else {
            denseFeedForward(g, p, l);
        }
        hyperConnectionPost(g, p + "hc_ffn");
    }

    private void hyperConnectionPre(
            TaskGraph g, String name, TornadoTensor fn, TornadoTensor base, TornadoTensor scale) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        final int dim = config.dim();
        final int n = config.hcCount();
        final int mixCount = config.hcMixCount();
        task(
                g,
                name + "_mix",
                groups(batch * mixCount, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcMixes,
                                context,
                                b.streams,
                                fn.asFloatArray(),
                                b.mixes,
                                n * dim,
                                mixCount,
                                config.rmsNormEps(),
                                batch));
        task(
                g,
                name + "_coef",
                lanes(batch, 32),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcCoefficients,
                                context,
                                b.mixes,
                                base.asFloatArray(),
                                scale.asFloatArray(),
                                b.coeffs,
                                n,
                                config.hcSinkhornIterations(),
                                config.hcEps(),
                                config.hcPostScale(),
                                batch));
        task(
                g,
                name + "_collapse",
                lanes(batch * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcCollapse,
                                context,
                                b.streams,
                                b.coeffs,
                                b.residual,
                                ws.wrapXBatch,
                                dim,
                                n,
                                batch));
    }

    private void hyperConnectionPost(TaskGraph g, String name) {
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        final int dim = config.dim();
        final int n = config.hcCount();
        task(
                g,
                name + "_expand",
                lanes(batch * n * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcExpand,
                                context,
                                b.residual,
                                b.branch,
                                b.coeffs,
                                b.streams,
                                dim,
                                n,
                                batch));
    }

    /** The sublayer's norm of {@code wrapXBatch} into {@code normed}, quantized. */
    private void norm(TaskGraph g, String name, TornadoTensor weight) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        final int dim = config.dim();
        task(
                g,
                name,
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::rmsNorm,
                                context,
                                ws.wrapXBatch,
                                b.normed,
                                weight.asFloatArray(),
                                dim,
                                config.rmsNormEps(),
                                batch));
        quantize(g, name + "_q8", b.normed, dim);
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    private void kda(TaskGraph g, String p, int l) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        var k = w.kda();
        final int dim = config.dim();
        final int heads = config.kdaHeads();
        final int hd = config.kdaHeadDim();
        final int width = config.kdaDim();
        final int kernel = config.kdaConvKernel();
        final int index = config.kdaIndex(l);
        final int window = width * (kernel - 1);
        final int headLocal = powerOfTwoAtMost(hd, Glm5NextBatchKernels.GROUP);

        project(g, p + "kda_q", k.q()[l], b.kdaQ, dim, width);
        project(g, p + "kda_k", k.k()[l], b.kdaK, dim, width);
        project(g, p + "kda_v", k.v()[l], b.kdaV, dim, width);
        project(g, p + "kda_f_a", k.decayA()[l], b.kdaDecayLow, dim, hd);
        project(g, p + "kda_g_a", k.gateA()[l], b.kdaGateLow, dim, hd);
        project(g, p + "kda_beta", k.beta()[l], b.kdaBeta, dim, heads);
        FloatArray[][] convs = {
            {b.kdaQ, k.convQ()[l].asFloatArray(), b.kdaConvQ},
            {b.kdaK, k.convK()[l].asFloatArray(), b.kdaConvK},
            {b.kdaV, k.convV()[l].asFloatArray(), b.kdaConvV}
        };
        for (int c = 0; c < 3; c++) {
            FloatArray in = convs[c][0];
            FloatArray taps = convs[c][1];
            FloatArray out = convs[c][2];
            int offset = (index * 3 + c) * window;
            task(
                    g,
                    p + "kda_conv" + c,
                    lanes(width, 128),
                    t ->
                            g.task(
                                    t,
                                    Glm5NextBatchKernels::causalConv1d,
                                    context,
                                    in,
                                    taps,
                                    ws.wrapConvState,
                                    out,
                                    ws.batchStartPosHolder,
                                    width,
                                    kernel,
                                    offset));
            task(
                    g,
                    p + "kda_silu" + c,
                    lanes(batch * width, 128),
                    t ->
                            g.task(
                                    t,
                                    Qwen35DeltaNetKernels::siluInPlace,
                                    context,
                                    out,
                                    batch * width));
        }
        float queryScale = (float) (1.0 / Math.sqrt(hd));
        task(
                g,
                p + "kda_norm_q",
                groups(batch * heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaL2Norm,
                                context,
                                b.kdaConvQ,
                                batch * heads,
                                hd,
                                queryScale,
                                Glm5NextConfiguration.KDA_L2_NORM_EPS));
        task(
                g,
                p + "kda_norm_k",
                groups(batch * heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaL2Norm,
                                context,
                                b.kdaConvK,
                                batch * heads,
                                hd,
                                1.0f,
                                Glm5NextConfiguration.KDA_L2_NORM_EPS));
        quantize(g, p + "kda_f_low_q8", b.kdaDecayLow, hd);
        project(g, p + "kda_f_b", k.decayB()[l], b.kdaDecay, hd, width);
        quantize(g, p + "kda_g_low_q8", b.kdaGateLow, hd);
        project(g, p + "kda_g_b", k.gateB()[l], b.kdaGate, hd, width);
        task(
                g,
                p + "kda_decay",
                lanes(batch * width, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::kdaDecayAndBeta,
                                context,
                                b.kdaDecay,
                                k.dtBias()[l].asFloatArray(),
                                k.aLog()[l].asFloatArray(),
                                b.kdaBeta,
                                width,
                                hd,
                                heads,
                                config.kdaGateLowerBound(),
                                batch));
        int stateOffset = index * heads * hd * hd;
        task(
                g,
                p + "kda_delta",
                lanes(width, Math.min(hd, 128)),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::kdaDeltaRule,
                                context,
                                b.kdaConvQ,
                                b.kdaConvK,
                                b.kdaConvV,
                                b.kdaDecay,
                                b.kdaBeta,
                                ws.wrapDeltaState,
                                b.kdaOut,
                                ws.batchStartPosHolder,
                                heads,
                                hd,
                                stateOffset));
        task(
                g,
                p + "kda_gated_norm",
                groups(batch * heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaGatedNorm,
                                context,
                                b.kdaOut,
                                b.kdaGate,
                                k.outNorm()[l].asFloatArray(),
                                batch * heads,
                                hd,
                                config.rmsNormEps()));
        quantize(g, p + "kda_out_q8", b.kdaOut, width);
        project(g, p + "kda_output", k.output()[l], b.branch, width, dim);
    }

    // ── sparse attention ──────────────────────────────────────────────────────

    private void sparseAttention(TaskGraph g, String p, int l) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        var s = w.sparse();
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int rank = config.kvLoraRank();
        final int keyHead = config.keyHeadDim();
        final int valueHead = config.valueHeadDim();
        final int qLora = config.qLoraRank();
        final int ih = config.indexerHeads();
        final int id = config.indexerHeadDim();
        final int pool = config.indexerPoolSize();
        final int maxPools = config.maxPools();
        final int cacheOffset = gs.latentCacheOffset(config, l);
        final int indexOffset = gs.indexerCacheOffset(config, l);
        final int poolOffset = gs.poolKeyOffset(config, l);
        final int idLocal = powerOfTwoAtMost(id, Glm5NextBatchKernels.GROUP);
        if (rank > Glm5NextBatchKernels.MAX_RANK) {
            throw new UnsupportedOperationException(
                    "the glm5next batched attention holds a latent of at most "
                            + Glm5NextBatchKernels.MAX_RANK);
        }

        project(g, p + "q_a", s.qA()[l], b.queryLatent, dim, qLora);
        project(g, p + "kv_a", s.kvA()[l], b.kvLatent, dim, rank);
        project(g, p + "idx_k", s.indexK()[l], b.indexRawKey, dim, id);
        project(g, p + "idx_pool_gate", s.poolGate()[l], b.indexGate, dim, id);
        project(g, p + "idx_weights", s.indexWeights()[l], b.indexHeadWeights, dim, ih);
        task(
                g,
                p + "q_a_norm",
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::rmsNorm,
                                context,
                                b.queryLatent,
                                b.queryLatent,
                                s.qANorm()[l].asFloatArray(),
                                qLora,
                                config.rmsNormEps(),
                                batch));
        task(
                g,
                p + "kv_a_norm",
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::rmsNorm,
                                context,
                                b.kvLatent,
                                b.kvLatent,
                                s.kvANorm()[l].asFloatArray(),
                                rank,
                                config.rmsNormEps(),
                                batch));
        quantize(g, p + "q_latent_q8", b.queryLatent, qLora);
        project(g, p + "q_b", s.qB()[l], b.query, qLora, config.queryDim());
        project(g, p + "idx_q_b", s.indexQ()[l], b.indexQuery, qLora, ih * id);
        task(
                g,
                p + "cache",
                lanes(batch * rank, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::storeRows,
                                context,
                                b.kvLatent,
                                ws.wrapKeyCache,
                                ws.batchStartPosHolder,
                                rank,
                                cacheOffset,
                                batch));
        task(
                g,
                p + "absorb",
                lanes(batch * heads * rank * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::projectSlicedF32,
                                context,
                                s.kB()[l].asByteArray(),
                                b.query,
                                b.absorbed,
                                keyHead,
                                heads * rank,
                                rank,
                                config.queryDim(),
                                batch));
        task(
                g,
                p + "idx_store",
                groups(batch, idLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::indexerStore,
                                context,
                                b.indexRawKey,
                                b.indexGate,
                                s.indexKNorm()[l].asFloatArray(),
                                s.indexKNormBias()[l].asFloatArray(),
                                ws.glm5next.indexKeyCache,
                                ws.glm5next.poolGateCache,
                                ws.batchStartPosHolder,
                                id,
                                indexOffset,
                                Glm5NextConfiguration.INDEXER_NORM_EPS));
        int poolSlots = batch / pool + 2;
        task(
                g,
                p + "idx_pool_keys",
                lanes(poolSlots * id, 32),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::indexerPoolKeys,
                                context,
                                ws.glm5next.indexKeyCache,
                                ws.glm5next.poolGateCache,
                                s.poolApe()[l].asFloatArray(),
                                ws.glm5next.poolKeys,
                                ws.batchStartPosHolder,
                                id,
                                pool,
                                indexOffset,
                                poolOffset,
                                poolSlots));
        task(
                g,
                p + "idx_scores",
                lanes(batch * maxPools, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::indexerScores,
                                context,
                                b.indexQuery,
                                b.indexHeadWeights,
                                ws.glm5next.poolKeys,
                                b.poolScores,
                                ws.batchStartPosHolder,
                                ih,
                                id,
                                pool,
                                maxPools,
                                poolOffset,
                                batch));
        task(
                g,
                p + "idx_select",
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::indexerSelect,
                                context,
                                b.poolScores,
                                b.selected,
                                ws.batchStartPosHolder,
                                pool,
                                maxPools,
                                config.indexerPoolBudget()));
        task(
                g,
                p + "attention",
                groups(batch * heads, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::sparseLatentAttention,
                                context,
                                b.absorbed,
                                ws.wrapKeyCache,
                                b.selected,
                                b.latentOut,
                                ws.batchStartPosHolder,
                                heads,
                                rank,
                                pool,
                                maxPools,
                                cacheOffset,
                                config.indexerSelectTail() ? 1 : 0,
                                config.attentionScale()));
        task(
                g,
                p + "decompress",
                lanes(batch * heads * valueHead * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::projectSlicedF32,
                                context,
                                s.vB()[l].asByteArray(),
                                b.latentOut,
                                b.attnOut,
                                rank,
                                heads * valueHead,
                                valueHead,
                                heads * rank,
                                batch));
        quantize(g, p + "attn_out_q8", b.attnOut, config.attentionOutputInputDim());
        project(
                g,
                p + "attn_output",
                s.output()[l],
                b.branch,
                config.attentionOutputInputDim(),
                dim);
    }

    // ── feed-forward ──────────────────────────────────────────────────────────

    private void denseFeedForward(TaskGraph g, String p, int l) {
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        var f = w.ffn();
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        project(g, p + "ffn_gate", f.gate()[l], b.hidden, dim, hidden);
        project(g, p + "ffn_up", f.up()[l], b.hidden2, dim, hidden);
        task(
                g,
                p + "ffn_swiglu",
                lanes(batch * hidden, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::clampedSwiGLU,
                                context,
                                b.hidden,
                                b.hidden2,
                                batch * hidden,
                                config.swigluLimit()));
        quantize(g, p + "ffn_hidden_q8", b.hidden, hidden);
        project(g, p + "ffn_down", f.down()[l], b.branch, hidden, dim);
    }

    /** Whether the grouped int8 expert kernels take this model's widths: whole 128-column tiles. */
    private boolean groupedExperts() {
        return config.dim() % Qwen35MoeBatchKernels.TILE_COLS == 0
                && config.expertHiddenDim() % Qwen35MoeBatchKernels.TILE_COLS == 0
                && gemmShape(config.dim(), config.sharedHiddenDim())
                && gemmShape(config.sharedHiddenDim(), config.dim());
    }

    // @formatter:off
    /**
     * The mixture of experts on int8 tensor cores:
     *
     * <ol>
     *   <li>sigmoid routing; the assignments sorted by expert and cut into 64-row tiles;
     *   <li>per stack of experts, the grouped clamped gate/up over the tiles of its experts, then
     *       the grouped down projection, the hidden activations quantized between;
     *   <li>the shared expert over every row as GEMMs, the clamp between;
     *   <li>each row's routed outputs weighted and added to the shared one, into {@code branch}.
     * </ol>
     */
    // @formatter:on
    private void groupedMixtureOfExperts(TaskGraph g, String p, int l) {
        var ws = gs.workspace;
        Glm5NextBatchArrays b = ws.glm5nextBatch;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int assignments = batch * used;
        List<TornadoTensor> gates = f.gateExperts().get(l);
        List<TornadoTensor> ups = f.upExperts().get(l);
        List<TornadoTensor> downs = f.downExperts().get(l);
        int split = gates.size() == 1 ? experts : config.expertChunkSize();
        int[][] ranges =
                gates.size() == 1
                        ? new int[][] {{0, experts}}
                        : new int[][] {{0, split}, {split, experts}};

        route(g, p, l);
        task(
                g,
                p + "group",
                groups(1, Qwen35MoeBatchKernels.GROUP_THREADS),
                t ->
                        g.task(
                                t,
                                Qwen35MoeBatchKernels::groupByExpert,
                                context,
                                b.expertIds,
                                ws.batchStartPosHolder,
                                b.sortedToken,
                                b.position,
                                b.tiles,
                                experts,
                                used,
                                batch));
        for (int s = 0; s < ranges.length; s++) {
            int lo = ranges[s][0];
            int hi = ranges[s][1];
            ByteArray gate = gates.get(s).asByteArray();
            ByteArray up = ups.get(s).asByteArray();
            WorkerGrid2D grid =
                    new WorkerGrid2D(
                            hidden
                                    / Qwen35MoeBatchKernels.TILE_COLS
                                    * Qwen35MoeBatchKernels.GEMM_THREADS,
                            b.maxTiles);
            grid.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
            task(
                    g,
                    p + "experts_gate_up" + s,
                    grid,
                    t ->
                            g.task(
                                    t,
                                    Glm5NextBatchKernels::groupedGateUpClamped,
                                    context,
                                    b.a8,
                                    b.a8Scales,
                                    b.sortedToken,
                                    b.tiles,
                                    gate,
                                    up,
                                    b.routedHidden,
                                    dim,
                                    hidden,
                                    lo,
                                    hi,
                                    config.swigluLimit()));
        }
        task(
                g,
                p + "experts_hidden_q8",
                lanes(assignments * hidden, 128),
                t ->
                        g.task(
                                t,
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                context,
                                b.routedHidden,
                                b.routedHidden8,
                                b.routedHiddenScales,
                                hidden));
        for (int s = 0; s < ranges.length; s++) {
            int lo = ranges[s][0];
            int hi = ranges[s][1];
            ByteArray down = downs.get(s).asByteArray();
            WorkerGrid2D grid =
                    new WorkerGrid2D(
                            dim
                                    / Qwen35MoeBatchKernels.TILE_COLS
                                    * Qwen35MoeBatchKernels.GEMM_THREADS,
                            b.maxTiles);
            grid.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
            task(
                    g,
                    p + "experts_down" + s,
                    grid,
                    t ->
                            g.task(
                                    t,
                                    Glm5NextBatchKernels::groupedDown,
                                    context,
                                    b.routedHidden8,
                                    b.routedHiddenScales,
                                    b.tiles,
                                    down,
                                    b.routedOut,
                                    dim,
                                    hidden,
                                    lo,
                                    hi));
        }
        // The shared expert reads the same quantized input as the routed ones, so it runs after
        // their gate/up and before its own hidden activation replaces that input.
        project(g, p + "shared_gate", f.sharedGate()[l], b.sharedGate, dim, shared);
        project(g, p + "shared_up", f.sharedUp()[l], b.sharedUp, dim, shared);
        task(
                g,
                p + "shared_swiglu",
                lanes(batch * shared, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::clampedSwiGLU,
                                context,
                                b.sharedGate,
                                b.sharedUp,
                                batch * shared,
                                config.swigluLimit()));
        quantize(g, p + "shared_hidden_q8", b.sharedGate, shared);
        project(g, p + "shared_down", f.sharedDown()[l], b.sharedOut, shared, dim);
        task(
                g,
                p + "experts_combine",
                lanes(batch * dim, 256),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::combineExperts,
                                context,
                                b.routedOut,
                                b.position,
                                b.expertWeights,
                                b.sharedOut,
                                b.branch,
                                ws.batchStartPosHolder,
                                dim,
                                used));
    }

    /**
     * The router logits and the sigmoid top-k, into {@code expertIds} and {@code expertWeights}.
     */
    private void route(TaskGraph g, String p, int l) {
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        task(
                g,
                p + "router",
                lanes(batch * experts * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::routerLogits,
                                context,
                                b.normed,
                                f.router()[l].asFloatArray(),
                                b.routerLogits,
                                dim,
                                experts,
                                batch));
        task(
                g,
                p + "router_topk",
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::routerTopKSigmoid,
                                context,
                                b.routerLogits,
                                f.routerBias()[l].asFloatArray(),
                                b.expertIds,
                                b.expertWeights,
                                experts,
                                used,
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale(),
                                batch));
    }

    private void mixtureOfExperts(TaskGraph g, String p, int l) {
        if (groupedExperts()) {
            groupedMixtureOfExperts(g, p, l);
            return;
        }
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int moeHidden = used * hidden + shared;
        List<TornadoTensor> gates = f.gateExperts().get(l);
        List<TornadoTensor> ups = f.upExperts().get(l);
        List<TornadoTensor> downs = f.downExperts().get(l);
        int last = gates.size() - 1;
        int split = gates.size() == 1 ? experts : config.expertChunkSize();
        ByteArray gate0 = gates.get(0).asByteArray(), gate1 = gates.get(last).asByteArray();
        ByteArray up0 = ups.get(0).asByteArray(), up1 = ups.get(last).asByteArray();
        ByteArray down0 = downs.get(0).asByteArray(), down1 = downs.get(last).asByteArray();
        task(
                g,
                p + "router",
                lanes(batch * experts * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::routerLogits,
                                context,
                                b.normed,
                                f.router()[l].asFloatArray(),
                                b.routerLogits,
                                dim,
                                experts,
                                batch));
        task(
                g,
                p + "router_topk",
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::routerTopKSigmoid,
                                context,
                                b.routerLogits,
                                f.routerBias()[l].asFloatArray(),
                                b.expertIds,
                                b.expertWeights,
                                experts,
                                used,
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale(),
                                batch));
        task(
                g,
                p + "experts_gate_up",
                lanes(batch * moeHidden * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::expertsGateUpClamped,
                                context,
                                b.quants,
                                b.scales,
                                b.expertIds,
                                gate0,
                                up0,
                                gate1,
                                up1,
                                f.sharedGate()[l].asByteArray(),
                                f.sharedUp()[l].asByteArray(),
                                b.moeHidden,
                                b.expertShape,
                                config.swigluLimit()));
        task(
                g,
                p + "experts_hidden_q8",
                lanes(batch * moeHidden, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                b.moeHidden,
                                b.moeQuants,
                                b.moeScales,
                                b.moeSums));
        task(
                g,
                p + "experts_down",
                lanes(batch * dim * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::expertsDown,
                                context,
                                b.moeQuants,
                                b.moeScales,
                                b.expertIds,
                                b.expertWeights,
                                down0,
                                down1,
                                f.sharedDown()[l].asByteArray(),
                                b.branch,
                                dim,
                                hidden,
                                shared,
                                used,
                                split,
                                batch));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /**
     * Every row's first {@code n} values of {@code x}, quantized per block in both forms: packed
     * for the DP4A kernels and bytewise for the int8 GEMMs, so a projection may take either.
     */
    private void quantize(TaskGraph g, String name, FloatArray x, int n) {
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        task(
                g,
                name,
                lanes(batch * n, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                x,
                                b.quants,
                                b.scales,
                                b.sums));
        task(
                g,
                name + "_i8",
                lanes(batch * n, 128),
                t ->
                        g.task(
                                t,
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                context,
                                x,
                                b.a8,
                                b.a8Scales,
                                n));
    }

    /** Whether {@link Int8GemmKernels#gemmInt8Q8_0} takes a projection of this shape. */
    static boolean gemmShape(int n, int d) {
        return d % Int8GemmKernels.I8_BN == 0 && n % Int8GemmKernels.I8_BK == 0;
    }

    // @formatter:off
    /**
     * {@code out_b = w . x_b} for every row; {@code n} inputs, {@code d} outputs. On the int8
     * tensor-core GEMM when the shape allows it — outputs in whole 128-column tiles, inputs in
     * whole 64-element rounds — over the chunk's rows padded to whole 128-row tiles, of which it
     * computes the real ones; otherwise a DP4A warp per row and output.
     */
    // @formatter:on
    private void project(
            TaskGraph g, String name, TornadoTensor weight, FloatArray out, int n, int d) {
        Glm5NextBatchArrays b = gs.workspace.glm5nextBatch;
        var ws = gs.workspace;
        if (gemmShape(n, d)) {
            int padded =
                    (batch + Int8GemmKernels.I8_BM - 1)
                            / Int8GemmKernels.I8_BM
                            * Int8GemmKernels.I8_BM;
            WorkerGrid2D grid =
                    new WorkerGrid2D(
                            padded / Int8GemmKernels.I8_BM * Int8GemmKernels.Q8_GEMM_THREADS,
                            d / Int8GemmKernels.I8_BN);
            grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
            task(
                    g,
                    name,
                    grid,
                    t ->
                            g.task(
                                    t,
                                    Int8GemmKernels::gemmInt8Q8_0,
                                    context,
                                    b.a8,
                                    b.a8Scales,
                                    weight.asByteArray(),
                                    out,
                                    out,
                                    padded,
                                    d,
                                    n,
                                    Int8GemmKernels.EPILOGUE_STORE,
                                    out,
                                    1,
                                    ws.batchStartPosHolder,
                                    d,
                                    0));
            return;
        }
        task(
                g,
                name,
                lanes(batch * d * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::projectQ8_0,
                                context,
                                b.quants,
                                b.scales,
                                out,
                                weight.asByteArray(),
                                n,
                                d,
                                batch));
    }

    private static int powerOfTwoAtMost(int n, int cap) {
        int p = 1;
        while (p * 2 <= Math.min(n, cap)) {
            p *= 2;
        }
        return p;
    }

    private static WorkerGrid lanes(int count, int local) {
        int global = (count + local - 1) / local * local;
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    private static WorkerGrid groups(int count, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(count * local);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    // ── BatchPrefillTransformerLayerTaskGraphs ────────────────────────────────

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return List.of(graph);
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return GRAPH;
    }

    @Override
    public String describeProjections() {
        return "int8 tensor-core GEMMs on Q8_0 weights where the shape allows, DP4A otherwise";
    }

    @Override
    public String describeAttention() {
        return "sparse latent attention, online softmax over position tiles";
    }
}
