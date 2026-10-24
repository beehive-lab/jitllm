package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.Glm5NextDeviceArrays;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.weights.Glm5NextLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.Glm5NextTornadoWeights;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The single-token {@code glm5next} layers (GLM-5.3-Flash), the whole trunk in one task graph.
 *
 * <p>The first layer copies the embedding into every residual stream. Each layer then runs two
 * sublayers, each inside a hyper-connection — the coefficients from the streams, the collapse, the
 * sublayer's norm and quantization, the sublayer into {@code branch}, the expansion back into the
 * streams:
 *
 * <ul>
 *   <li>KDA: the q, k, v, decay, gate and beta projections; three short convolutions; the L2 norms;
 *       the per-channel decay; the delta rule against the layer's state; the gated norm; {@code
 *       attn_output};
 *   <li>sparse attention: {@code q_a} and {@code kv_a}, the indexer's projections, the latent
 *       norms, {@code q_b}; the latent into the cache; the absorbed query; the indexer's key, pool
 *       key, scores and selection; attention over the selection; {@code attn_v_b}; {@code
 *       attn_output};
 *   <li>the feed-forward: the clamped SwiGLU of a leading block, or sigmoid routing, the routed and
 *       shared experts' clamped SwiGLU and their down projection.
 * </ul>
 *
 * <p>The last layer averages the streams into {@code wrapX} for the logits. Q8_0 projections only;
 * the routed experts may be stored in at most two stacks.
 */
// @formatter:on
public class Glm5NextLayers
        extends AbstractTransformerLayerTaskGraphs<Glm5NextTornadoWeights, Glm5NextConfiguration> {

    private final Glm5NextState gs;
    private final Glm5NextLayerWeights<TornadoTensor> w;
    private final String activationGraphName;

    /** Every task's worker grid, by qualified name, recorded as the graph is built. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

    public Glm5NextLayers(
            String taskGraphName,
            Glm5NextState state,
            Glm5NextTornadoWeights weights,
            Glm5NextConfiguration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        this(taskGraphName, state, weights, config, schedulerType, activationGraphName, null);
    }

    /**
     * The decode layers of a plan that also prefills in batches: they take the caches, the
     * recurrent state and every weight from the prefill graph, so the plan holds one copy of each.
     */
    public Glm5NextLayers(
            String taskGraphName,
            Glm5NextState state,
            Glm5NextTornadoWeights weights,
            Glm5NextConfiguration config,
            SchedulerType schedulerType,
            String activationGraphName,
            Glm5NextBatchPrefillLayers prefill) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.gs = state;
        this.w = weights.layers;
        this.activationGraphName = activationGraphName;
        this.prefill = prefill;
        requireSupported();
        setupFFNLayers();
    }

    /** In the batched plan, the prefill layers whose arrays these share; otherwise null. */
    private final Glm5NextBatchPrefillLayers prefill;

    private void requireSupported() {
        if (state.usesFp16KeyValueCache()) {
            throw new UnsupportedOperationException("glm5next keeps an FP32 latent cache");
        }
        for (int l = 0; l < config.numberOfLayers(); l++) {
            List<TornadoTensor> projections = new ArrayList<>();
            if (config.isKda(l)) {
                var k = w.kda();
                projections.addAll(
                        List.of(
                                k.q()[l],
                                k.k()[l],
                                k.v()[l],
                                k.decayA()[l],
                                k.decayB()[l],
                                k.beta()[l],
                                k.gateA()[l],
                                k.gateB()[l],
                                k.output()[l]));
            } else {
                var s = w.sparse();
                projections.addAll(
                        List.of(
                                s.qA()[l],
                                s.qB()[l],
                                s.kvA()[l],
                                s.kB()[l],
                                s.vB()[l],
                                s.output()[l],
                                s.indexQ()[l],
                                s.indexK()[l],
                                s.indexWeights()[l],
                                s.poolGate()[l]));
            }
            var f = w.ffn();
            if (config.isMoe(l)) {
                if (f.gateExperts().get(l).size() > 2) {
                    throw new UnsupportedOperationException(
                            "glm5next on the accelerator reads the routed experts from at most two"
                                    + " stacks; layer "
                                    + l
                                    + " holds "
                                    + f.gateExperts().get(l).size());
                }
                projections.addAll(f.gateExperts().get(l));
                projections.addAll(f.upExperts().get(l));
                projections.addAll(f.downExperts().get(l));
                projections.addAll(List.of(f.sharedGate()[l], f.sharedUp()[l], f.sharedDown()[l]));
            } else {
                projections.addAll(List.of(f.gate()[l], f.up()[l], f.down()[l]));
            }
            for (TornadoTensor t : projections) {
                if (t.dataType() != DataType.Q8_0) {
                    throw new UnsupportedOperationException(
                            "glm5next on the accelerator reads Q8_0 projections only; layer "
                                    + l
                                    + " holds "
                                    + t.dataType());
                }
            }
        }
    }

    // ── graph ─────────────────────────────────────────────────────────────────

    @Override
    protected void setupFFNLayers() {
        TaskGraph graph = new TaskGraph(layerGraphName(0));
        for (int l = 0; l < config.numberOfLayers(); l++) {
            appendLayer(graph, l);
        }
        lastFFNLayerTaskGraphID = graph.getTaskGraphName();
        ffnLayerITGs = List.of(graph.snapshot());
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        throw new UnsupportedOperationException("the glm5next trunk is one task graph");
    }

    private void task(TaskGraph graph, int layer, String name, WorkerGrid grid, TaskAdder adder) {
        String task = "l" + layer + "_" + name;
        adder.add(task);
        grids.put(graph.getTaskGraphName() + "." + task, grid);
    }

    @FunctionalInterface
    private interface TaskAdder {
        void add(String taskName);
    }

    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    private void bindScratch(TaskGraph graph) {
        var ws = gs.workspace;
        Glm5NextDeviceArrays d = ws.glm5next;
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder);
        graph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                ws.wrapXb,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapXbSums,
                ws.wrapHb,
                ws.wrapHb2,
                ws.wrapQueryLatent,
                ws.wrapQ,
                ws.wrapCompressedKv,
                ws.wrapAbsorbedQuery,
                ws.wrapLatentOut,
                ws.wrapAtt,
                ws.wrapRouterLogits,
                ws.wrapSelectedExperts,
                ws.wrapRoutingWeights,
                ws.wrapMoeHidden,
                ws.wrapMoeHiddenQuants,
                ws.wrapMoeHiddenQScales,
                ws.wrapMoeHiddenQSums,
                d.streams,
                d.residual,
                d.mixes,
                d.mixPartials,
                d.coeffs,
                d.branch,
                d.kdaQ,
                d.kdaK,
                d.kdaV,
                d.kdaConvQ,
                d.kdaConvK,
                d.kdaConvV,
                d.kdaDecayLow,
                d.kdaGateLow,
                d.kdaDecay,
                d.kdaBeta,
                d.kdaGate,
                d.kdaOut,
                d.attnOut,
                d.indexRawKey,
                d.indexGate,
                d.indexQuery,
                d.indexHeadWeights,
                d.poolScores,
                d.selected,
                d.noRope);
        Object[] persistent = {
            ws.wrapKeyCache,
            ws.wrapConvState,
            ws.wrapDeltaState,
            d.indexKeyCache,
            d.poolGateCache,
            d.poolKeys
        };
        if (prefill != null) {
            graph.consumeFromDevice(Glm5NextBatchPrefillLayers.GRAPH, persistent);
        } else {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, persistent);
        }
    }

    private void bindWeights(TaskGraph graph, int l) {
        if (prefill != null) {
            graph.consumeFromDevice(
                    Glm5NextBatchPrefillLayers.GRAPH, prefillArrays(w, config, l).toArray());
            List<Object> own = notInPrefill(w, config, l);
            if (!own.isEmpty()) {
                graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, own.toArray());
            }
        } else {
            graph.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, layerArrays(w, config, l).toArray());
        }
    }

    /**
     * The arrays of layer {@code l} the batched prefill never reads, so never holds on the device:
     * {@code attn_k_b} and {@code attn_v_b} when it attends on tensor cores from FP16 copies.
     */
    static List<Object> notInPrefill(
            Glm5NextLayerWeights<TornadoTensor> w, Glm5NextConfiguration config, int l) {
        if (config.isKda(l) || !config.tensorCoreAttention()) {
            return List.of();
        }
        return List.of(deviceArray(w.sparse().kB()[l]), deviceArray(w.sparse().vB()[l]));
    }

    /** The arrays of layer {@code l} the batched prefill binds and the decode takes from it. */
    static List<Object> prefillArrays(
            Glm5NextLayerWeights<TornadoTensor> w, Glm5NextConfiguration config, int l) {
        List<Object> arrays = layerArrays(w, config, l);
        List<Object> skip = notInPrefill(w, config, l);
        arrays.removeIf(a -> skip.stream().anyMatch(o -> o == a));
        return arrays;
    }

    /** Every device array layer {@code l} reads, in the order the graphs bind them. */
    static List<Object> layerArrays(
            Glm5NextLayerWeights<TornadoTensor> w, Glm5NextConfiguration config, int l) {
        List<Object> arrays = new ArrayList<>();
        var m = w.mhc();
        for (TornadoTensor t :
                new TornadoTensor[] {
                    m.attnFn()[l],
                    m.attnBase()[l],
                    m.attnScale()[l],
                    m.ffnFn()[l],
                    m.ffnBase()[l],
                    m.ffnScale()[l],
                    w.attnNorm()[l],
                    w.ffnNorm()[l]
                }) {
            arrays.add(deviceArray(t));
        }
        if (config.isKda(l)) {
            var k = w.kda();
            for (TornadoTensor t :
                    new TornadoTensor[] {
                        k.q()[l],
                        k.k()[l],
                        k.v()[l],
                        k.convQ()[l],
                        k.convK()[l],
                        k.convV()[l],
                        k.decayA()[l],
                        k.decayB()[l],
                        k.dtBias()[l],
                        k.aLog()[l],
                        k.beta()[l],
                        k.gateA()[l],
                        k.gateB()[l],
                        k.outNorm()[l],
                        k.output()[l]
                    }) {
                arrays.add(deviceArray(t));
            }
        } else {
            var s = w.sparse();
            for (TornadoTensor t :
                    new TornadoTensor[] {
                        s.qA()[l],
                        s.qANorm()[l],
                        s.qB()[l],
                        s.kvA()[l],
                        s.kvANorm()[l],
                        s.kB()[l],
                        s.vB()[l],
                        s.output()[l],
                        s.indexQ()[l],
                        s.indexK()[l],
                        s.indexKNorm()[l],
                        s.indexKNormBias()[l],
                        s.indexWeights()[l],
                        s.poolApe()[l],
                        s.poolGate()[l]
                    }) {
                arrays.add(deviceArray(t));
            }
        }
        var f = w.ffn();
        if (config.isMoe(l)) {
            arrays.add(deviceArray(f.router()[l]));
            arrays.add(deviceArray(f.routerBias()[l]));
            for (List<TornadoTensor> stacks :
                    List.of(f.gateExperts().get(l), f.upExperts().get(l), f.downExperts().get(l))) {
                for (TornadoTensor t : stacks) {
                    arrays.add(deviceArray(t));
                }
            }
            arrays.add(deviceArray(f.sharedGate()[l]));
            arrays.add(deviceArray(f.sharedUp()[l]));
            arrays.add(deviceArray(f.sharedDown()[l]));
        } else {
            arrays.add(deviceArray(f.gate()[l]));
            arrays.add(deviceArray(f.up()[l]));
            arrays.add(deviceArray(f.down()[l]));
        }
        return arrays;
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var ws = gs.workspace;
        Glm5NextDeviceArrays d = ws.glm5next;
        final int dim = config.dim();
        final int n = config.hcCount();
        final float eps = config.rmsNormEps();
        var m = w.mhc();

        if (l == 0) {
            g.consumeFromDevice(activationGraphName, ws.wrapX);
            bindScratch(g);
            task(
                    g,
                    l,
                    "broadcast",
                    lanes(n * dim, 128),
                    t ->
                            g.task(
                                    t,
                                    Glm5NextKernels::hcBroadcast,
                                    context,
                                    ws.wrapX,
                                    d.streams,
                                    dim,
                                    n));
        }
        bindWeights(g, l);

        hyperConnectionPre(g, l, "hc_attn", m.attnFn()[l], m.attnBase()[l], m.attnScale()[l]);
        normQuantize(g, l, "attn_norm", w.attnNorm()[l], dim, eps);
        if (config.isKda(l)) {
            kda(g, l);
        } else {
            sparseAttention(g, l);
        }
        hyperConnectionPost(g, l, "hc_attn");

        hyperConnectionPre(g, l, "hc_ffn", m.ffnFn()[l], m.ffnBase()[l], m.ffnScale()[l]);
        normQuantize(g, l, "ffn_norm", w.ffnNorm()[l], dim, eps);
        if (config.isMoe(l)) {
            mixtureOfExperts(g, l);
        } else {
            denseFeedForward(g, l);
        }
        hyperConnectionPost(g, l, "hc_ffn");

        if (l == config.numberOfLayers() - 1) {
            task(
                    g,
                    l,
                    "streams_mean",
                    lanes(dim, 128),
                    t -> g.task(t, Glm5NextKernels::hcMean, context, d.streams, ws.wrapX, dim, n));
            g.persistOnDevice(ws.wrapX);
        }
    }

    private void hyperConnectionPre(
            TaskGraph g,
            int l,
            String name,
            TornadoTensor fn,
            TornadoTensor base,
            TornadoTensor scale) {
        var ws = gs.workspace;
        Glm5NextDeviceArrays d = ws.glm5next;
        final int dim = config.dim();
        final int n = config.hcCount();
        final int mixCount = config.hcMixCount();
        task(
                g,
                l,
                name + "_mix",
                groups(mixCount * Glm5NextKernels.HC_CHUNKS, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcMixes,
                                context,
                                d.streams,
                                fn.asFloatArray(),
                                d.mixPartials,
                                n * dim,
                                mixCount));
        task(
                g,
                l,
                name + "_coef",
                groups(1, hcLanes(mixCount)),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcCoefficients,
                                context,
                                d.mixPartials,
                                base.asFloatArray(),
                                scale.asFloatArray(),
                                d.mixes,
                                d.coeffs,
                                n,
                                n * dim,
                                config.hcSinkhornIterations(),
                                config.hcEps(),
                                config.hcPostScale(),
                                config.rmsNormEps()));
        task(
                g,
                l,
                name + "_collapse",
                lanes(dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcCollapse,
                                context,
                                d.streams,
                                d.coeffs,
                                d.residual,
                                ws.wrapX,
                                dim,
                                n));
    }

    private void hyperConnectionPost(TaskGraph g, int l, String name) {
        Glm5NextDeviceArrays d = gs.workspace.glm5next;
        final int dim = config.dim();
        final int n = config.hcCount();
        task(
                g,
                l,
                name + "_expand",
                lanes(n * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcExpand,
                                context,
                                d.residual,
                                d.branch,
                                d.coeffs,
                                d.streams,
                                dim,
                                n));
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    private void kda(TaskGraph g, int l) {
        var ws = gs.workspace;
        Glm5NextDeviceArrays d = ws.glm5next;
        var k = w.kda();
        final int dim = config.dim();
        final int heads = config.kdaHeads();
        final int hd = config.kdaHeadDim();
        final int width = config.kdaDim();
        final int kernel = config.kdaConvKernel();
        final int index = config.kdaIndex(l);
        final int window = width * (kernel - 1);
        final int headLocal = powerOfTwoAtMost(hd, Glm5NextKernels.GROUP);

        project(g, l, "kda_q", k.q()[l], d.kdaQ, dim, width);
        project(g, l, "kda_k", k.k()[l], d.kdaK, dim, width);
        project(g, l, "kda_v", k.v()[l], d.kdaV, dim, width);
        project(g, l, "kda_f_a", k.decayA()[l], d.kdaDecayLow, dim, hd);
        project(g, l, "kda_g_a", k.gateA()[l], d.kdaGateLow, dim, hd);
        project(g, l, "kda_beta", k.beta()[l], d.kdaBeta, dim, heads);

        FloatArray[][] convs = {
            {d.kdaQ, k.convQ()[l].asFloatArray(), d.kdaConvQ},
            {d.kdaK, k.convK()[l].asFloatArray(), d.kdaConvK},
            {d.kdaV, k.convV()[l].asFloatArray(), d.kdaConvV}
        };
        for (int c = 0; c < 3; c++) {
            FloatArray in = convs[c][0];
            FloatArray taps = convs[c][1];
            FloatArray out = convs[c][2];
            int offset = (index * 3 + c) * window;
            task(
                    g,
                    l,
                    "kda_conv" + c,
                    lanes(width, 128),
                    t ->
                            g.task(
                                    t,
                                    Qwen35DeltaNetKernels::causalConv1d,
                                    context,
                                    in,
                                    taps,
                                    ws.wrapConvState,
                                    out,
                                    width,
                                    kernel,
                                    offset));
            task(
                    g,
                    l,
                    "kda_silu" + c,
                    lanes(width, 128),
                    t -> g.task(t, Qwen35DeltaNetKernels::siluInPlace, context, out, width));
        }
        float queryScale = (float) (1.0 / Math.sqrt(hd));
        task(
                g,
                l,
                "kda_norm_q",
                groups(heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaL2Norm,
                                context,
                                d.kdaConvQ,
                                heads,
                                hd,
                                queryScale,
                                Glm5NextConfiguration.KDA_L2_NORM_EPS));
        task(
                g,
                l,
                "kda_norm_k",
                groups(heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaL2Norm,
                                context,
                                d.kdaConvK,
                                heads,
                                hd,
                                1.0f,
                                Glm5NextConfiguration.KDA_L2_NORM_EPS));

        quantize(g, l, "kda_f_low_q8", d.kdaDecayLow, hd);
        project(g, l, "kda_f_b", k.decayB()[l], d.kdaDecay, hd, width);
        quantize(g, l, "kda_g_low_q8", d.kdaGateLow, hd);
        project(g, l, "kda_g_b", k.gateB()[l], d.kdaGate, hd, width);
        task(
                g,
                l,
                "kda_decay",
                lanes(width, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaDecayAndBeta,
                                context,
                                d.kdaDecay,
                                k.dtBias()[l].asFloatArray(),
                                k.aLog()[l].asFloatArray(),
                                d.kdaBeta,
                                width,
                                hd,
                                heads,
                                config.kdaGateLowerBound()));
        int stateOffset = index * heads * hd * hd;
        task(
                g,
                l,
                "kda_delta",
                lanes(width, Math.min(hd, 128)),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaDeltaRule,
                                context,
                                d.kdaConvQ,
                                d.kdaConvK,
                                d.kdaConvV,
                                d.kdaDecay,
                                d.kdaBeta,
                                ws.wrapDeltaState,
                                d.kdaOut,
                                heads,
                                hd,
                                stateOffset));
        task(
                g,
                l,
                "kda_gated_norm",
                groups(heads, headLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::kdaGatedNorm,
                                context,
                                d.kdaOut,
                                d.kdaGate,
                                k.outNorm()[l].asFloatArray(),
                                heads,
                                hd,
                                config.rmsNormEps()));
        quantize(g, l, "kda_out_q8", d.kdaOut, width);
        project(g, l, "kda_output", k.output()[l], d.branch, width, dim);
    }

    // ── sparse attention ──────────────────────────────────────────────────────

    private void sparseAttention(TaskGraph g, int l) {
        var ws = gs.workspace;
        Glm5NextDeviceArrays d = ws.glm5next;
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
        final int contextLength = config.contextLength();
        final int maxPools = config.maxPools();
        final int cacheOffset = gs.latentCacheOffset(config, l);
        final int indexOffset = gs.indexerCacheOffset(config, l);
        final int poolOffset = gs.poolKeyOffset(config, l);
        final int idLocal = powerOfTwoAtMost(id, Glm5NextKernels.GROUP);

        task(
                g,
                l,
                "q_kv_a",
                lanes((qLora + rank) * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::twoProjectionsQ8_0,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                s.qA()[l].asByteArray(),
                                ws.wrapQueryLatent,
                                s.kvA()[l].asByteArray(),
                                ws.wrapCompressedKv,
                                dim,
                                qLora,
                                rank));
        project(g, l, "idx_k", s.indexK()[l], d.indexRawKey, dim, id);
        project(g, l, "idx_pool_gate", s.poolGate()[l], d.indexGate, dim, id);
        project(g, l, "idx_weights", s.indexWeights()[l], d.indexHeadWeights, dim, ih);
        task(
                g,
                l,
                "latent_norms",
                lanes(2 * DeepSeek2Kernels.GROUP, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::latentNorms,
                                context,
                                ws.wrapQueryLatent,
                                s.qANorm()[l].asFloatArray(),
                                ws.wrapCompressedKv,
                                s.kvANorm()[l].asFloatArray(),
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums,
                                qLora,
                                rank,
                                config.rmsNormEps()));
        project(g, l, "q_b", s.qB()[l], ws.wrapQ, qLora, config.queryDim());
        project(g, l, "idx_q_b", s.indexQ()[l], d.indexQuery, qLora, ih * id);
        task(
                g,
                l,
                "cache",
                lanes(rank, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::ropeAndCache,
                                context,
                                ws.positionHolder,
                                ws.wrapQ,
                                ws.wrapCompressedKv,
                                d.noRope,
                                d.noRope,
                                ws.wrapKeyCache,
                                heads,
                                keyHead,
                                keyHead,
                                0,
                                rank,
                                cacheOffset));
        task(
                g,
                l,
                "absorb",
                lanes(heads * rank * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::absorbQuery,
                                context,
                                s.kB()[l].asByteArray(),
                                ws.wrapQ,
                                ws.wrapAbsorbedQuery,
                                heads,
                                rank,
                                keyHead,
                                0,
                                keyHead));
        task(
                g,
                l,
                "idx_store",
                groups(1, idLocal),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::indexerStore,
                                context,
                                d.indexRawKey,
                                d.indexGate,
                                s.indexKNorm()[l].asFloatArray(),
                                s.indexKNormBias()[l].asFloatArray(),
                                d.indexKeyCache,
                                d.poolGateCache,
                                ws.positionHolder,
                                id,
                                indexOffset,
                                Glm5NextConfiguration.INDEXER_NORM_EPS));
        task(
                g,
                l,
                "idx_pool_key",
                lanes(id, 32),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::indexerPoolKey,
                                context,
                                d.indexKeyCache,
                                d.poolGateCache,
                                s.poolApe()[l].asFloatArray(),
                                d.poolKeys,
                                ws.positionHolder,
                                id,
                                pool,
                                indexOffset,
                                poolOffset));
        task(
                g,
                l,
                "idx_scores",
                lanes(maxPools * Glm5NextKernels.INDEX_LANES, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::indexerScores,
                                context,
                                d.indexQuery,
                                d.indexHeadWeights,
                                d.poolKeys,
                                d.poolScores,
                                ws.positionHolder,
                                ih,
                                id,
                                pool,
                                maxPools,
                                poolOffset));
        task(
                g,
                l,
                "idx_select",
                groups(1, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::indexerSelect,
                                context,
                                d.poolScores,
                                d.selected,
                                ws.positionHolder,
                                pool,
                                maxPools,
                                config.indexerPoolBudget()));
        task(
                g,
                l,
                "attention",
                groups(heads, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::sparseLatentAttention,
                                context,
                                ws.wrapAbsorbedQuery,
                                ws.wrapKeyCache,
                                d.selected,
                                ws.wrapAtt,
                                ws.wrapLatentOut,
                                ws.positionHolder,
                                heads,
                                rank,
                                pool,
                                contextLength,
                                cacheOffset,
                                config.indexerSelectTail() ? 1 : 0,
                                config.attentionScale()));
        task(
                g,
                l,
                "decompress",
                lanes(heads * valueHead * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::decompressLatent,
                                context,
                                s.vB()[l].asByteArray(),
                                ws.wrapLatentOut,
                                d.attnOut,
                                heads,
                                valueHead,
                                rank));
        quantize(g, l, "attn_out_q8", d.attnOut, config.attentionOutputInputDim());
        project(
                g,
                l,
                "attn_output",
                s.output()[l],
                d.branch,
                config.attentionOutputInputDim(),
                dim);
    }

    // ── feed-forward ──────────────────────────────────────────────────────────

    private void denseFeedForward(TaskGraph g, int l) {
        var ws = gs.workspace;
        var f = w.ffn();
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        project(g, l, "ffn_gate", f.gate()[l], ws.wrapHb, dim, hidden);
        project(g, l, "ffn_up", f.up()[l], ws.wrapHb2, dim, hidden);
        task(
                g,
                l,
                "ffn_swiglu",
                lanes(hidden, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::clampedSwiGLU,
                                context,
                                ws.wrapHb,
                                ws.wrapHb2,
                                hidden,
                                config.swigluLimit()));
        quantize(g, l, "ffn_hidden_q8", ws.wrapHb, hidden);
        project(g, l, "ffn_down", f.down()[l], ws.glm5next.branch, hidden, dim);
    }

    private void mixtureOfExperts(TaskGraph g, int l) {
        var ws = gs.workspace;
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
                l,
                "router",
                lanes(experts * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::routerLogits,
                                context,
                                ws.wrapXb,
                                f.router()[l].asFloatArray(),
                                ws.wrapRouterLogits,
                                dim,
                                experts));
        task(
                g,
                l,
                "router_topk",
                groups(1, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::routerTopKSigmoid,
                                context,
                                ws.wrapRouterLogits,
                                f.routerBias()[l].asFloatArray(),
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                experts,
                                used,
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale()));
        task(
                g,
                l,
                "experts_gate_up",
                lanes(moeHidden * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::expertsGateUpClamped,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapSelectedExperts,
                                gate0,
                                up0,
                                gate1,
                                up1,
                                f.sharedGate()[l].asByteArray(),
                                f.sharedUp()[l].asByteArray(),
                                ws.wrapMoeHidden,
                                dim,
                                hidden,
                                shared,
                                used,
                                split,
                                config.swigluLimit()));
        task(
                g,
                l,
                "experts_hidden_q8",
                lanes(moeHidden, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                ws.wrapMoeHidden,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapMoeHiddenQSums));
        task(
                g,
                l,
                "experts_down",
                lanes(dim * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::expertsDown,
                                context,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                down0,
                                down1,
                                f.sharedDown()[l].asByteArray(),
                                ws.glm5next.branch,
                                dim,
                                hidden,
                                shared,
                                used,
                                split));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /** The sublayer's RMS norm of {@code wrapX} into {@code wrapXb}, quantized for projections. */
    private void normQuantize(
            TaskGraph g, int l, String name, TornadoTensor weight, int n, float eps) {
        var ws = gs.workspace;
        task(
                g,
                l,
                name,
                lanes(n, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::rmsNormQuantize,
                                context,
                                ws.wrapX,
                                ws.wrapXb,
                                weight.asFloatArray(),
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums,
                                n,
                                eps));
    }

    /** {@code x}'s first {@code n} values in Q8 blocks, into the shared quantized activation. */
    private void quantize(TaskGraph g, int l, String name, FloatArray x, int n) {
        var ws = gs.workspace;
        task(
                g,
                l,
                name,
                lanes(n, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                x,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums));
    }

    /** {@code out = w . x}, the activation quantized; {@code n} inputs, {@code d} outputs. */
    private void project(
            TaskGraph g, int l, String name, TornadoTensor weight, FloatArray out, int n, int d) {
        var ws = gs.workspace;
        int local = projectionLocal(n);
        task(
                g,
                l,
                name,
                lanes(d * local, local),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                out,
                                weight.asByteArray(),
                                n,
                                d,
                                local));
    }

    /** Lanes per output row: as many as the row has blocks, between one and four warps. */
    private static int projectionLocal(int n) {
        int blocks = n / 32;
        return blocks >= 128 ? 128 : blocks >= 64 ? 64 : 32;
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

    /** The power-of-two workgroup {@link Glm5NextKernels#hcCoefficients} needs for the mixes. */
    private static int hcLanes(int mixCount) {
        if (mixCount > Glm5NextKernels.HC_MAX_MIXES) {
            throw new IllegalArgumentException("too many mHC mixing coefficients: " + mixCount);
        }
        return Math.max(32, Integer.highestOneBit(mixCount - 1) << 1);
    }

    private static WorkerGrid groups(int count, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(count * local);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
        return scheduler;
    }
}
