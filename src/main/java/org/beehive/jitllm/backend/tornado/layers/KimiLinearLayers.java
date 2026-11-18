package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2Kernels;
import org.beehive.jitllm.backend.tornado.kernels.KdaKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.jitllm.backend.tornado.kernels.RoutedExpertKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.KdaDeviceArrays;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.KimiLinearTornadoWeights;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;
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
 * The single-token {@code kimi-linear} layers (Kimi-Linear-48B-A3B), a stage's layers in one task
 * graph.
 *
 * <p>Per layer, the input norm and its quantization, then the token mixer:
 *
 * <ul>
 *   <li>KDA: the q, k, v, decay, gate and beta projections; three short convolutions and their
 *       SiLU; the L2 norms; the softplus decay; the delta rule against the layer's state; the gated
 *       norm; {@code attn_output} into the residual;
 *   <li>MLA: {@code attn_q} and {@code attn_kv_a_mqa}; the latent norm; the cache row (the rotation
 *       by the identity tables leaves the key tail as it is); each head's query absorbed through
 *       {@code attn_k_b}; attention over the latent rows in slices, merged; {@code attn_v_b};
 *       {@code attn_output} into the residual.
 * </ul>
 *
 * <p>Then the feed-forward's norm and either the dense SwiGLU of a leading block or sigmoid
 * routing, the routed and shared experts' SwiGLU and their down projection, added into the
 * residual. Q8_0 projections; the routed experts all Q8_0 or all Q4_0.
 */
// @formatter:on
public class KimiLinearLayers
        extends AbstractTransformerLayerTaskGraphs<
                KimiLinearTornadoWeights, KimiLinearConfiguration> {

    private final KimiLinearState ks;
    private final KimiLinearLayerWeights<TornadoTensor> w;
    private final String activationGraphName;
    private final boolean fp16Cache;

    /** Every task's worker grid, by qualified name, recorded as the graph is built. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

    public KimiLinearLayers(
            KimiLinearState state,
            KimiLinearTornadoWeights weights,
            KimiLinearConfiguration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        this(
                state,
                weights,
                config,
                schedulerType,
                activationGraphName,
                0,
                config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)}, one stage of a model split across devices. The
     * first takes the hidden state from the graph named {@code activationGraphName}.
     */
    public KimiLinearLayers(
            KimiLinearState state,
            KimiLinearTornadoWeights weights,
            KimiLinearConfiguration config,
            SchedulerType schedulerType,
            String activationGraphName,
            int firstLayer,
            int endLayer) {
        super(state, weights, config, schedulerType);
        this.ks = state;
        this.w = weights.layers;
        this.activationGraphName = activationGraphName;
        this.fp16Cache = state.usesFp16KeyValueCache();
        restrictToLayers(firstLayer, endLayer);
        requireSupported();
        setupFFNLayers();
    }

    private void requireSupported() {
        for (int l = firstLayer; l < endLayer(config.numberOfLayers()); l++) {
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
                var m = w.mla();
                projections.addAll(
                        List.of(m.q()[l], m.kvAMqa()[l], m.kB()[l], m.vB()[l], m.output()[l]));
            }
            var f = w.ffn();
            if (config.isDenseLayer(l)) {
                projections.addAll(List.of(f.gate()[l], f.up()[l], f.down()[l]));
            } else {
                projections.addAll(List.of(f.sharedGate()[l], f.sharedUp()[l], f.sharedDown()[l]));
                DataType routed = f.gateExperts()[l].dataType();
                for (TornadoTensor t :
                        List.of(f.gateExperts()[l], f.upExperts()[l], f.downExperts()[l])) {
                    if (t.dataType() != routed
                            || (routed != DataType.Q8_0 && routed != DataType.Q4_0)) {
                        throw new UnsupportedOperationException(
                                "kimi-linear on the accelerator reads its routed experts all in"
                                        + " Q8_0 or all in Q4_0; layer "
                                        + l
                                        + " holds "
                                        + t.dataType());
                    }
                }
            }
            for (TornadoTensor t : projections) {
                if (t.dataType() != DataType.Q8_0) {
                    throw new UnsupportedOperationException(
                            "kimi-linear on the accelerator reads Q8_0 projections only; layer "
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
        int end = endLayer(config.numberOfLayers());
        TaskGraph graph = new TaskGraph(layerGraphName(firstLayer));
        for (int l = firstLayer; l < end; l++) {
            appendLayer(graph, l);
        }
        lastFFNLayerTaskGraphID = graph.getTaskGraphName();
        ffnLayerITGs = List.of(graph.snapshot());
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        throw new UnsupportedOperationException("a kimi-linear stage is one task graph");
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

    /** The latent cache in the representation the state allocated. */
    private Object cache() {
        return fp16Cache ? ks.workspace.wrapKeyCacheFP16 : ks.workspace.wrapKeyCache;
    }

    private void bindScratch(TaskGraph graph) {
        var ws = ks.workspace;
        KdaDeviceArrays d = ws.kda;
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder);
        graph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                ws.wrapXb,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapXbSums,
                ws.wrapHb,
                ws.wrapQ,
                ws.wrapCompressedKv,
                ws.wrapAbsorbedQuery,
                ws.wrapLatentOut,
                ws.wrapAtt,
                ws.wrapAttSplit,
                ws.wrapRouterLogits,
                ws.wrapSelectedExperts,
                ws.wrapRoutingWeights,
                ws.wrapMoeHidden,
                ws.wrapMoeHiddenQuants,
                ws.wrapMoeHiddenQScales,
                ws.wrapMoeHiddenQSums,
                d.q,
                d.k,
                d.v,
                d.convQ,
                d.convK,
                d.convV,
                d.decayLow,
                d.gateLow,
                d.decay,
                d.beta,
                d.gate,
                d.out,
                d.branch,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());
        graph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION, cache(), ws.wrapConvState, ws.wrapDeltaState);
    }

    private void bindWeights(TaskGraph graph, int l) {
        graph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION, layerArrays(w, config, l).toArray());
    }

    /** Every device array layer {@code l} reads, in the order the graphs bind them. */
    static List<Object> layerArrays(
            KimiLinearLayerWeights<TornadoTensor> w, KimiLinearConfiguration config, int l) {
        List<TornadoTensor> tensors = new ArrayList<>(List.of(w.attnNorm()[l], w.ffnNorm()[l]));
        if (config.isKda(l)) {
            var k = w.kda();
            tensors.addAll(
                    List.of(
                            k.q()[l],
                            k.k()[l],
                            k.v()[l],
                            k.convQ()[l],
                            k.convK()[l],
                            k.convV()[l],
                            k.decayA()[l],
                            k.decayB()[l],
                            k.dtBias()[l],
                            k.a()[l],
                            k.beta()[l],
                            k.gateA()[l],
                            k.gateB()[l],
                            k.outNorm()[l],
                            k.output()[l]));
        } else {
            var m = w.mla();
            tensors.addAll(
                    List.of(
                            m.q()[l],
                            m.kvAMqa()[l],
                            m.kvANorm()[l],
                            m.kB()[l],
                            m.vB()[l],
                            m.output()[l]));
        }
        var f = w.ffn();
        if (config.isDenseLayer(l)) {
            tensors.addAll(List.of(f.gate()[l], f.up()[l], f.down()[l]));
        } else {
            tensors.addAll(
                    List.of(
                            f.router()[l],
                            f.routerBias()[l],
                            f.gateExperts()[l],
                            f.upExperts()[l],
                            f.downExperts()[l],
                            f.sharedGate()[l],
                            f.sharedUp()[l],
                            f.sharedDown()[l]));
        }
        List<Object> arrays = new ArrayList<>();
        for (TornadoTensor t : tensors) {
            arrays.add(deviceArray(t));
        }
        return arrays;
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var ws = ks.workspace;
        final int dim = config.dim();
        final float eps = config.rmsNormEps();

        if (l == firstLayer) {
            g.consumeFromDevice(activationGraphName, ws.wrapX);
            bindScratch(g);
        }
        bindWeights(g, l);

        normQuantize(g, l, "attn_norm", w.attnNorm()[l], dim, eps);
        if (config.isKda(l)) {
            kda(g, l);
        } else {
            latentAttention(g, l);
        }

        normQuantize(g, l, "ffn_norm", w.ffnNorm()[l], dim, eps);
        if (config.isDenseLayer(l)) {
            denseFeedForward(g, l);
        } else {
            mixtureOfExperts(g, l);
        }

        if (l == endLayer(config.numberOfLayers()) - 1) {
            g.persistOnDevice(ws.wrapX, cache(), ws.wrapConvState, ws.wrapDeltaState);
        }
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    private void kda(TaskGraph g, int l) {
        var ws = ks.workspace;
        KdaDeviceArrays d = ws.kda;
        var k = w.kda();
        final int dim = config.dim();
        final int heads = config.kdaHeads();
        final int hd = config.kdaHeadDim();
        final int width = config.kdaDim();
        final int kernel = config.kdaConvKernel();
        final int index = config.kdaIndex(l);
        final int window = width * (kernel - 1);
        final int headLocal = powerOfTwoAtMost(hd, KdaKernels.GROUP);
        final float eps = config.rmsNormEps();

        project(g, l, "kda_q", k.q()[l], d.q, dim, width);
        project(g, l, "kda_k", k.k()[l], d.k, dim, width);
        project(g, l, "kda_v", k.v()[l], d.v, dim, width);
        project(g, l, "kda_f_a", k.decayA()[l], d.decayLow, dim, hd);
        project(g, l, "kda_g_a", k.gateA()[l], d.gateLow, dim, hd);
        project(g, l, "kda_beta", k.beta()[l], d.beta, dim, heads);

        FloatArray[][] convs = {
            {d.q, k.convQ()[l].asFloatArray(), d.convQ},
            {d.k, k.convK()[l].asFloatArray(), d.convK},
            {d.v, k.convV()[l].asFloatArray(), d.convV}
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
                                KdaKernels::kdaL2Norm,
                                context,
                                d.convQ,
                                heads,
                                hd,
                                queryScale,
                                eps));
        task(
                g,
                l,
                "kda_norm_k",
                groups(heads, headLocal),
                t -> g.task(t, KdaKernels::kdaL2Norm, context, d.convK, heads, hd, 1.0f, eps));

        quantize(g, l, "kda_f_low_q8", d.decayLow, hd);
        project(g, l, "kda_f_b", k.decayB()[l], d.decay, hd, width);
        quantize(g, l, "kda_g_low_q8", d.gateLow, hd);
        project(g, l, "kda_g_b", k.gateB()[l], d.gate, hd, width);
        task(
                g,
                l,
                "kda_decay",
                lanes(width, 128),
                t ->
                        g.task(
                                t,
                                KdaKernels::kdaSoftplusDecayAndBeta,
                                context,
                                d.decay,
                                k.dtBias()[l].asFloatArray(),
                                k.a()[l].asFloatArray(),
                                d.beta,
                                width,
                                hd,
                                heads));
        int stateOffset = index * heads * hd * hd;
        task(
                g,
                l,
                "kda_delta",
                groups(
                        heads * ((hd + KdaKernels.DELTA_COLUMNS - 1) / KdaKernels.DELTA_COLUMNS),
                        KdaKernels.DELTA_COLUMNS * KdaKernels.DELTA_CHUNKS),
                t ->
                        g.task(
                                t,
                                KdaKernels::kdaDeltaRule,
                                context,
                                d.convQ,
                                d.convK,
                                d.convV,
                                d.decay,
                                d.beta,
                                ws.wrapDeltaState,
                                d.out,
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
                                KdaKernels::kdaGatedNorm,
                                context,
                                d.out,
                                d.gate,
                                k.outNorm()[l].asFloatArray(),
                                heads,
                                hd,
                                eps));
        quantize(g, l, "kda_out_q8", d.out, width);
        projectResidual(g, l, "kda_output", k.output()[l], width, dim);
    }

    // ── MLA ───────────────────────────────────────────────────────────────────

    private void latentAttention(TaskGraph g, int l) {
        var ws = ks.workspace;
        var m = w.mla();
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int rank = config.kvLoraRank();
        final int rope = config.ropeDim();
        final int nope = config.nopeDim();
        final int qHead = config.queryHeadDim();
        final int valueHead = config.valueHeadDim();
        final int attnOut = config.attentionOutputInputDim();
        final int splits = KimiLinearConfiguration.DECODE_ATTENTION_SPLITS;
        final int cacheOffset = config.mlaIndex(l) * config.contextLength() * config.keyWidth();

        task(
                g,
                l,
                "q_kv_a",
                lanes((config.queryDim() + rank + rope) * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::twoProjectionsQ8_0,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                m.q()[l].asByteArray(),
                                ws.wrapQ,
                                m.kvAMqa()[l].asByteArray(),
                                ws.wrapCompressedKv,
                                dim,
                                config.queryDim(),
                                rank + rope));
        task(
                g,
                l,
                "kv_norm",
                groups(1, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::rmsNorm,
                                context,
                                ws.wrapCompressedKv,
                                ws.wrapCompressedKv,
                                m.kvANorm()[l].asFloatArray(),
                                rank,
                                config.rmsNormEps()));
        int ropeLanes = heads * rope / 2 + rope / 2 + rank;
        task(
                g,
                l,
                "cache",
                lanes(ropeLanes, 128),
                t -> {
                    if (fp16Cache) {
                        g.task(
                                t,
                                DeepSeek2Kernels::ropeAndCacheFP16,
                                context,
                                ws.positionHolder,
                                ws.wrapQ,
                                ws.wrapCompressedKv,
                                weights.freq_cis_realFlat.asFloatArray(),
                                weights.freq_cis_imagFlat.asFloatArray(),
                                ws.wrapKeyCacheFP16,
                                heads,
                                qHead,
                                nope,
                                rope,
                                rank,
                                cacheOffset);
                    } else {
                        g.task(
                                t,
                                DeepSeek2Kernels::ropeAndCache,
                                context,
                                ws.positionHolder,
                                ws.wrapQ,
                                ws.wrapCompressedKv,
                                weights.freq_cis_realFlat.asFloatArray(),
                                weights.freq_cis_imagFlat.asFloatArray(),
                                ws.wrapKeyCache,
                                heads,
                                qHead,
                                nope,
                                rope,
                                rank,
                                cacheOffset);
                    }
                });
        int absorbWarps = heads * rank + (heads * rope + 31) / 32;
        task(
                g,
                l,
                "absorb",
                lanes(absorbWarps * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::absorbQuery,
                                context,
                                m.kB()[l].asByteArray(),
                                ws.wrapQ,
                                ws.wrapAbsorbedQuery,
                                heads,
                                rank,
                                nope,
                                rope,
                                qHead));
        task(
                g,
                l,
                "attention",
                lanes(heads * splits * DeepSeek2Kernels.GROUP, DeepSeek2Kernels.GROUP),
                t -> {
                    if (fp16Cache) {
                        g.task(
                                t,
                                DeepSeek2Kernels::attentionSplitFP16,
                                context,
                                ws.positionHolder,
                                ws.wrapAbsorbedQuery,
                                ws.wrapKeyCacheFP16,
                                ws.wrapAtt,
                                ws.wrapAttSplit,
                                heads,
                                rank,
                                rope,
                                cacheOffset,
                                config.contextLength(),
                                splits,
                                config.attentionScale());
                    } else {
                        g.task(
                                t,
                                DeepSeek2Kernels::attentionSplit,
                                context,
                                ws.positionHolder,
                                ws.wrapAbsorbedQuery,
                                ws.wrapKeyCache,
                                ws.wrapAtt,
                                ws.wrapAttSplit,
                                heads,
                                rank,
                                rope,
                                cacheOffset,
                                config.contextLength(),
                                splits,
                                config.attentionScale());
                    }
                });
        task(
                g,
                l,
                "attention_combine",
                lanes(heads * rank, 256),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::combineSplits,
                                context,
                                ws.wrapAttSplit,
                                ws.wrapLatentOut,
                                heads,
                                rank,
                                splits));
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
                                m.vB()[l].asByteArray(),
                                ws.wrapLatentOut,
                                ws.wrapXb,
                                heads,
                                valueHead,
                                rank));
        quantize(g, l, "attn_out_q8", ws.wrapXb, attnOut);
        projectResidual(g, l, "attn_output", m.output()[l], attnOut, dim);
    }

    // ── feed-forward ──────────────────────────────────────────────────────────

    private void denseFeedForward(TaskGraph g, int l) {
        var ws = ks.workspace;
        var f = w.ffn();
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        final int local = projectionLocal(dim);
        task(
                g,
                l,
                "ffn_gate_up",
                lanes(hidden * local, local),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapHb,
                                f.gate()[l].asByteArray(),
                                f.up()[l].asByteArray(),
                                dim,
                                hidden,
                                local));
        quantize(g, l, "ffn_hidden_q8", ws.wrapHb, hidden);
        projectResidual(g, l, "ffn_down", f.down()[l], hidden, dim);
    }

    private void mixtureOfExperts(TaskGraph g, int l) {
        var ws = ks.workspace;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int moeHidden = used * hidden + shared;
        ByteArray gate = f.gateExperts()[l].asByteArray();
        ByteArray up = f.upExperts()[l].asByteArray();
        ByteArray down = f.downExperts()[l].asByteArray();
        boolean q4 = f.gateExperts()[l].dataType() == DataType.Q4_0;

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
                groups(1, RoutedExpertKernels.GROUP),
                t ->
                        g.task(
                                t,
                                RoutedExpertKernels::routerTopKSigmoid,
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
                                q4
                                        ? RoutedExpertKernels::expertsGateUpClampedQ4_0
                                        : RoutedExpertKernels::expertsGateUpClamped,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapSelectedExperts,
                                gate,
                                up,
                                gate,
                                up,
                                f.sharedGate()[l].asByteArray(),
                                f.sharedUp()[l].asByteArray(),
                                ws.wrapMoeHidden,
                                dim,
                                hidden,
                                shared,
                                used,
                                experts,
                                Float.POSITIVE_INFINITY));
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
                                q4
                                        ? RoutedExpertKernels::expertsDownQ4_0
                                        : RoutedExpertKernels::expertsDown,
                                context,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                down,
                                down,
                                f.sharedDown()[l].asByteArray(),
                                ws.kda.branch,
                                dim,
                                hidden,
                                shared,
                                used,
                                experts));
        task(
                g,
                l,
                "experts_residual",
                lanes(dim, 128),
                t ->
                        g.task(
                                t,
                                RoutedExpertKernels::addInto,
                                context,
                                ws.wrapX,
                                ws.kda.branch,
                                dim));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /** The sublayer's RMS norm of {@code wrapX} into {@code wrapXb}, quantized for projections. */
    private void normQuantize(
            TaskGraph g, int l, String name, TornadoTensor weight, int n, float eps) {
        var ws = ks.workspace;
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
        var ws = ks.workspace;
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
        var ws = ks.workspace;
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

    /** {@code x += w . activation}, the activation quantized. */
    private void projectResidual(
            TaskGraph g, int l, String name, TornadoTensor weight, int n, int d) {
        var ws = ks.workspace;
        int local = projectionLocal(n);
        task(
                g,
                l,
                name,
                lanes(d * local, local),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ8_0DP4A
                                        ::matrixVectorGenericWithResidualQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapX,
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

    /** {@code count} lanes rounded up to whole workgroups of {@code local}. */
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

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
        return scheduler;
    }
}
