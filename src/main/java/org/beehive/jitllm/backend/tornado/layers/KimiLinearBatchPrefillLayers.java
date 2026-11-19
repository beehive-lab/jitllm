package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.BatchMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.KdaBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.KdaKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.KdaBatchArrays;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.KimiLinearTornadoWeights;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;
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
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The batched prefill of {@code kimi-linear} layers: a chunk of prompt tokens per execution, a
 * stage's layers in one task graph, a row per token.
 *
 * <ul>
 *   <li>projections: block-scaled int8 tensor-core GEMMs on the Q8_0 weights; {@code attn_kv_a_mqa}
 *       and {@code ssm_beta} are padded to whole 128-column tiles;
 *   <li>KDA: the convolution and the delta rule walk the chunk's rows in order inside one launch
 *       ({@link KdaBatchKernels}), and leave the windows and matrices where the decode resumes;
 *   <li>MLA: deepseek2's absorbed latent attention as FP16 tensor-core GEMMs, the key tail rotated
 *       by the identity tables;
 *   <li>experts: sigmoid routing, the assignments grouped by expert, grouped int8 GEMMs (Q8_0 as
 *       qwen35moe's, Q4_0 recentred as their tiles are staged), the shared expert as GEMMs.
 * </ul>
 *
 * <p>The latent cache and the KDA state are the decode plan's own arrays; the decode layers take
 * them, and the weights this graph binds, from this graph by name.
 */
// @formatter:on
public class KimiLinearBatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    public static final String GRAPH = "kimiLinearPrefill";

    private final KimiLinearState state;
    private final KimiLinearTornadoWeights weights;
    private final KimiLinearLayerWeights<TornadoTensor> w;
    private final KimiLinearConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int firstLayer;
    private final int endLayer;
    private final int batch;
    private final boolean fp16Cache;
    private final ByteArray[] kvAPadded;
    private final ByteArray[] betaPadded;
    private final HalfFloatArray[] kBF16;
    private final HalfFloatArray[] vBF16;
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();
    private final ImmutableTaskGraph graph;

    public KimiLinearBatchPrefillLayers(
            KimiLinearState state,
            KimiLinearTornadoWeights weights,
            KimiLinearConfiguration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        if (state.paddedBatch == 0 || state.workspace.kdaBatch == null) {
            throw new IllegalStateException(
                    "the kimi-linear batched prefill needs a state built for batch " + batchSize);
        }
        PrefillFamily family = PrefillFamily.of(state);
        if (family.isNarrow()) {
            throw PrefillFamily.decline(
                    "the kimi-linear batched prefill runs every chunk at the plan's width");
        }
        this.state = state;
        this.weights = weights;
        this.w = weights.layers;
        this.config = config;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.batch = state.paddedBatch;
        this.fp16Cache = state.usesFp16KeyValueCache();
        int layers = config.numberOfLayers();
        this.kvAPadded = new ByteArray[layers];
        this.betaPadded = new ByteArray[layers];
        this.kBF16 = new HalfFloatArray[layers];
        this.vBF16 = new HalfFloatArray[layers];
        for (int l = firstLayer; l < endLayer; l++) {
            if (config.isKda(l)) {
                betaPadded[l] =
                        padRows(
                                w.kda().beta()[l].asByteArray(),
                                config.kdaHeads(),
                                KdaBatchArrays.BETA_PADDED);
            } else {
                kvAPadded[l] =
                        padRows(
                                w.mla().kvAMqa()[l].asByteArray(),
                                config.compressedKeyValueDim(),
                                KimiLinearState.KV_A_PADDED);
                kBF16[l] = toHalf(w.mla().kB()[l].asByteArray());
                vBF16[l] = toHalf(w.mla().vB()[l].asByteArray());
            }
        }
        TaskGraph g = family.taskGraph(GRAPH);
        bind(g);
        for (int l = firstLayer; l < endLayer; l++) {
            appendLayer(g, l);
        }
        g.persistOnDevice(persistent());
        g.persistOnDevice(state.workspace.wrapXBatch);
        this.graph = g.snapshot();
    }

    // ── host-side weight forms ────────────────────────────────────────────────

    /** {@code rows} Q8_0 rows of {@code dim} values padded with zero rows to {@code padded}. */
    private ByteArray padRows(ByteArray q8, int rows, int padded) {
        int rowBytes = config.dim() / 32 * 34;
        ByteArray out = new ByteArray(padded * rowBytes);
        for (int i = 0; i < rows * rowBytes; i++) {
            out.set(i, q8.get(i));
        }
        return out;
    }

    /** Q8_0 blocks decoded to FP16, element for element. */
    private static HalfFloatArray toHalf(ByteArray q8) {
        int blocks = q8.getSize() / 34;
        HalfFloatArray out = new HalfFloatArray(blocks * 32);
        for (int b = 0; b < blocks; b++) {
            float scale = q8.getHalfFloat(b * 34).getFloat32();
            for (int i = 0; i < 32; i++) {
                out.set(b * 32 + i, new HalfFloat(scale * q8.get(b * 34 + 2 + i)));
            }
        }
        return out;
    }

    // ── binding ───────────────────────────────────────────────────────────────

    private Object cache() {
        return fp16Cache ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** What the decode layers continue from: the latent cache and the KDA state. */
    public Object[] persistent() {
        var ws = state.workspace;
        return new Object[] {cache(), ws.wrapConvState, ws.wrapDeltaState};
    }

    /** The weights of layer {@code l} this graph binds in the decode's own form. */
    public static List<Object> sharedWeights(
            KimiLinearLayerWeights<TornadoTensor> w, KimiLinearConfiguration config, int l) {
        List<Object> arrays = KimiLinearLayers.layerArrays(w, config, l);
        List<Object> skip = notShared(w, config, l);
        arrays.removeIf(a -> skip.stream().anyMatch(o -> o == a));
        return arrays;
    }

    /** The weights of layer {@code l} the prefill reads in another form; the decode binds them. */
    public static List<Object> notShared(
            KimiLinearLayerWeights<TornadoTensor> w, KimiLinearConfiguration config, int l) {
        if (config.isKda(l)) {
            return List.of(w.kda().beta()[l].asByteArray());
        }
        var m = w.mla();
        return List.of(
                m.kvAMqa()[l].asByteArray(), m.kB()[l].asByteArray(), m.vB()[l].asByteArray());
    }

    private void bind(TaskGraph g) {
        var ws = state.workspace;
        KdaBatchArrays k = ws.kdaBatch;
        if (firstLayer == 0) {
            g.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, ws.wrapXBatch);
        } else {
            g.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, ws.wrapXBatch);
        }
        g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
        g.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                ws.wrapNormedBatch,
                ws.compressedKvBatch,
                ws.queryBatch,
                ws.queryF16Batch,
                ws.absorbedBatch,
                ws.absorbedF16Batch,
                ws.keysF16,
                ws.latentTF16,
                ws.scoresBatch,
                ws.probsF16Batch,
                ws.latentBatch,
                ws.latentF16Batch,
                ws.attnOutBatch,
                ws.gateBatch,
                ws.hiddenBatch,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                ws.wrapQ8SplitPartial,
                ws.wrapMoeLogitsBatch,
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedGateBatch,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                ws.wrapMoeOutBatch,
                ws.wrapMoeSharedGateUp,
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                ws.wrapMoeSharedOut,
                k.q,
                k.k,
                k.v,
                k.convQ,
                k.convK,
                k.convV,
                k.decayLow,
                k.gateLow,
                k.decay,
                k.betaWide,
                k.beta,
                k.gate,
                k.out,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, persistent());
        List<Object> own = new ArrayList<>();
        for (int l = firstLayer; l < endLayer; l++) {
            own.addAll(sharedWeights(w, config, l));
            if (config.isKda(l)) {
                own.add(betaPadded[l]);
            } else {
                own.add(kvAPadded[l]);
                own.add(kBF16[l]);
                own.add(vBF16[l]);
            }
        }
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, own.toArray());
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var ws = state.workspace;
        final int dim = config.dim();
        final String p = "l" + l + "_";
        rmsRows(
                g,
                p + "attn_norm",
                ws.wrapXBatch,
                ws.wrapNormedBatch,
                w.attnNorm()[l],
                dim,
                dim,
                dim);
        quantize(g, p + "attn_q8", ws.wrapNormedBatch, dim);
        if (config.isKda(l)) {
            kda(g, p, l);
        } else {
            latentAttention(g, p, l);
        }
        rmsRows(
                g,
                p + "ffn_norm",
                ws.wrapXBatch,
                ws.wrapNormedBatch,
                w.ffnNorm()[l],
                dim,
                dim,
                dim);
        quantize(g, p + "ffn_q8", ws.wrapNormedBatch, dim);
        if (config.isDenseLayer(l)) {
            denseFeedForward(g, p, l);
        } else {
            experts(g, p, l);
        }
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    private void kda(TaskGraph g, String p, int l) {
        var ws = state.workspace;
        KdaBatchArrays b = ws.kdaBatch;
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
        if (hd > KdaBatchKernels.MAX_KDA_HEAD) {
            throw new UnsupportedOperationException(
                    "the batched delta rule holds KDA heads of at most "
                            + KdaBatchKernels.MAX_KDA_HEAD);
        }

        gemm(
                g,
                p + "kda_q",
                k.q()[l].asByteArray(),
                b.q,
                width,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kda_k",
                k.k()[l].asByteArray(),
                b.k,
                width,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kda_v",
                k.v()[l].asByteArray(),
                b.v,
                width,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kda_f_a",
                k.decayA()[l].asByteArray(),
                b.decayLow,
                hd,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kda_g_a",
                k.gateA()[l].asByteArray(),
                b.gateLow,
                hd,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kda_beta",
                betaPadded[l],
                b.betaWide,
                KdaBatchArrays.BETA_PADDED,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);

        FloatArray[][] convs = {
            {b.q, k.convQ()[l].asFloatArray(), b.convQ},
            {b.k, k.convK()[l].asFloatArray(), b.convK},
            {b.v, k.convV()[l].asFloatArray(), b.convV}
        };
        for (int c = 0; c < 3; c++) {
            FloatArray in = convs[c][0];
            FloatArray taps = convs[c][1];
            FloatArray out = convs[c][2];
            int offset = (index * 3 + c) * window;
            g.task(
                    p + "kda_conv" + c,
                    KdaBatchKernels::causalConv1d,
                    context,
                    in,
                    taps,
                    ws.wrapConvState,
                    out,
                    ws.batchStartPosHolder,
                    width,
                    kernel,
                    offset);
            grid(g, p + "kda_conv" + c, lanes(width, 128));
            g.task(
                    p + "kda_silu" + c,
                    Qwen35DeltaNetKernels::siluInPlace,
                    context,
                    out,
                    batch * width);
            grid(g, p + "kda_silu" + c, lanes(batch * width, 128));
        }
        g.task(
                p + "kda_norm_q",
                KdaKernels::kdaL2Norm,
                context,
                b.convQ,
                batch * heads,
                hd,
                (float) (1.0 / Math.sqrt(hd)),
                eps);
        grid(g, p + "kda_norm_q", groups(batch * heads, headLocal));
        g.task(
                p + "kda_norm_k",
                KdaKernels::kdaL2Norm,
                context,
                b.convK,
                batch * heads,
                hd,
                1.0f,
                eps);
        grid(g, p + "kda_norm_k", groups(batch * heads, headLocal));

        quantize(g, p + "kda_f_low_q8", b.decayLow, hd);
        gemm(
                g,
                p + "kda_f_b",
                k.decayB()[l].asByteArray(),
                b.decay,
                width,
                hd,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        quantize(g, p + "kda_g_low_q8", b.gateLow, hd);
        gemm(
                g,
                p + "kda_g_b",
                k.gateB()[l].asByteArray(),
                b.gate,
                width,
                hd,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        g.task(
                p + "kda_decay",
                KdaBatchKernels::kdaSoftplusDecayAndBeta,
                context,
                b.decay,
                k.dtBias()[l].asFloatArray(),
                k.a()[l].asFloatArray(),
                b.betaWide,
                b.beta,
                width,
                hd,
                heads,
                KdaBatchArrays.BETA_PADDED,
                batch);
        grid(g, p + "kda_decay", lanes(batch * width, 128));
        int stateOffset = index * heads * hd * hd;
        g.task(
                p + "kda_delta",
                KdaBatchKernels::kdaDeltaRule,
                context,
                b.convQ,
                b.convK,
                b.convV,
                b.decay,
                b.beta,
                ws.wrapDeltaState,
                b.out,
                ws.batchStartPosHolder,
                heads,
                hd,
                stateOffset);
        grid(
                g,
                p + "kda_delta",
                groups(
                        heads * ((hd + KdaKernels.DELTA_COLUMNS - 1) / KdaKernels.DELTA_COLUMNS),
                        KdaKernels.DELTA_COLUMNS * KdaKernels.DELTA_CHUNKS));
        g.task(
                p + "kda_gated_norm",
                KdaKernels::kdaGatedNorm,
                context,
                b.out,
                b.gate,
                k.outNorm()[l].asFloatArray(),
                batch * heads,
                hd,
                eps);
        grid(g, p + "kda_gated_norm", groups(batch * heads, headLocal));
        quantize(g, p + "kda_out_q8", b.out, width);
        gemm(
                g,
                p + "kda_output",
                k.output()[l].asByteArray(),
                ws.wrapXBatch,
                dim,
                width,
                Int8GemmKernels.EPILOGUE_RESIDUAL,
                true);
    }

    // ── MLA ───────────────────────────────────────────────────────────────────

    private void latentAttention(TaskGraph g, String p, int l) {
        var ws = state.workspace;
        var m = w.mla();
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int rank = config.kvLoraRank();
        final int rope = config.ropeDim();
        final int nope = config.nopeDim();
        final int qHead = config.queryHeadDim();
        final int qDim = config.queryDim();
        final int valueHead = config.valueHeadDim();
        final int attnOut = config.attentionOutputInputDim();
        final int keyWidth = config.keyWidth();
        final int kvPad = KimiLinearState.KV_A_PADDED;
        final int npad = state.paddedPositions;
        final int rows = batch * heads;
        final int cacheOffset = config.mlaIndex(l) * config.contextLength() * keyWidth;

        gemm(
                g,
                p + "q",
                m.q()[l].asByteArray(),
                ws.queryBatch,
                qDim,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kv_a",
                kvAPadded[l],
                ws.compressedKvBatch,
                kvPad,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        rmsRows(
                g,
                p + "kv_a_norm",
                ws.compressedKvBatch,
                ws.compressedKvBatch,
                m.kvANorm()[l],
                rank,
                kvPad,
                kvPad);

        if (fp16Cache) {
            g.task(
                    p + "rope_cache",
                    DeepSeek2BatchKernels::ropeAndCacheRowsFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.queryBatch,
                    ws.compressedKvBatch,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCacheFP16,
                    heads,
                    qHead,
                    nope,
                    rope,
                    rank,
                    qDim,
                    kvPad,
                    cacheOffset);
        } else {
            g.task(
                    p + "rope_cache",
                    DeepSeek2BatchKernels::ropeAndCacheRows,
                    context,
                    ws.batchStartPosHolder,
                    ws.queryBatch,
                    ws.compressedKvBatch,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCache,
                    heads,
                    qHead,
                    nope,
                    rope,
                    rank,
                    qDim,
                    kvPad,
                    cacheOffset);
        }
        grid(g, p + "rope_cache", lanes(batch * (heads * rope / 2 + rope / 2 + rank), 256));
        toHalf(g, p + "q_f16", ws.queryBatch, ws.queryF16Batch, batch * qDim);
        mma(
                g,
                p + "absorb",
                ws.queryF16Batch,
                kBF16[l],
                ws.absorbedBatch,
                batch,
                rank,
                nope,
                qDim,
                nope,
                heads * keyWidth,
                qHead,
                rank * nope,
                keyWidth,
                heads);
        g.task(
                p + "queries",
                DeepSeek2BatchKernels::assembleQueries,
                context,
                ws.absorbedBatch,
                ws.queryBatch,
                ws.absorbedF16Batch,
                heads,
                rank,
                rope,
                nope,
                qHead,
                qDim,
                rows * keyWidth);
        grid(g, p + "queries", lanes(rows * keyWidth, 256));
        if (fp16Cache) {
            g.task(
                    p + "cache_f16",
                    DeepSeek2BatchKernels::cacheToHalfFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKeyCacheFP16,
                    ws.keysF16,
                    ws.latentTF16,
                    cacheOffset,
                    rank,
                    rope,
                    npad);
        } else {
            g.task(
                    p + "cache_f16",
                    DeepSeek2BatchKernels::cacheToHalf,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKeyCache,
                    ws.keysF16,
                    ws.latentTF16,
                    cacheOffset,
                    rank,
                    rope,
                    npad);
        }
        grid(g, p + "cache_f16", lanes(npad * keyWidth, 256));
        mma(
                g,
                p + "scores",
                ws.absorbedF16Batch,
                ws.keysF16,
                ws.scoresBatch,
                rows,
                npad,
                keyWidth,
                keyWidth,
                keyWidth,
                npad,
                0,
                0,
                0,
                1,
                1,
                0);
        g.task(
                p + "softmax",
                BatchMmaKernels::causalSoftmax,
                context,
                ws.batchStartPosHolder,
                ws.scoresBatch,
                ws.probsF16Batch,
                heads,
                npad,
                config.attentionScale());
        grid(
                g,
                p + "softmax",
                lanes(rows * DeepSeek2BatchKernels.GROUP, DeepSeek2BatchKernels.GROUP));
        mma(
                g,
                p + "attend",
                ws.probsF16Batch,
                ws.latentTF16,
                ws.latentBatch,
                rows,
                rank,
                npad,
                npad,
                npad,
                rank,
                0,
                0,
                0,
                1,
                0,
                1);
        toHalf(g, p + "latent_f16", ws.latentBatch, ws.latentF16Batch, rows * rank);
        mma(
                g,
                p + "decompress",
                ws.latentF16Batch,
                vBF16[l],
                ws.attnOutBatch,
                batch,
                valueHead,
                rank,
                heads * rank,
                rank,
                attnOut,
                rank,
                valueHead * rank,
                valueHead,
                heads);
        quantize(g, p + "attn_out_q8", ws.attnOutBatch, attnOut);
        gemm(
                g,
                p + "wo",
                m.output()[l].asByteArray(),
                ws.wrapXBatch,
                dim,
                attnOut,
                Int8GemmKernels.EPILOGUE_RESIDUAL,
                true);
    }

    // ── feed-forward ──────────────────────────────────────────────────────────

    private void denseFeedForward(TaskGraph g, String p, int l) {
        var ws = state.workspace;
        var f = w.ffn();
        final int dim = config.dim();
        final int hidden = config.hiddenDim();
        gemm(
                g,
                p + "ffn_gate",
                f.gate()[l].asByteArray(),
                ws.gateBatch,
                hidden,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemmGated(
                g,
                p + "ffn_up",
                f.up()[l].asByteArray(),
                ws.hiddenBatch,
                ws.gateBatch,
                hidden,
                dim);
        quantize(g, p + "ffn_hidden_q8", ws.hiddenBatch, hidden);
        gemm(
                g,
                p + "ffn_down",
                f.down()[l].asByteArray(),
                ws.wrapXBatch,
                dim,
                hidden,
                Int8GemmKernels.EPILOGUE_RESIDUAL,
                true);
    }

    private void experts(TaskGraph g, String p, int l) {
        var ws = state.workspace;
        var f = w.ffn();
        int dim = config.dim();
        int experts = config.expertCount();
        int used = config.expertsUsed();
        int hidden = config.expertHiddenDim();
        int shared = config.sharedHiddenDim();
        int assignments = batch * used;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, experts);
        boolean q4 = f.gateExperts()[l].dataType() == DataType.Q4_0;

        g.task(
                p + "router",
                Qwen35MoeBatchKernels::routerTiled,
                context,
                ws.wrapNormedBatch,
                f.router()[l].asFloatArray(),
                ws.wrapMoeLogitsBatch,
                ws.batchStartPosHolder,
                dim,
                experts);
        WorkerGrid router =
                new WorkerGrid2D(
                        batch / Qwen35MoeBatchKernels.ROUTER_TILE * 256,
                        experts / Qwen35MoeBatchKernels.ROUTER_TILE);
        router.setLocalWork(256, 1, 1);
        grid(g, p + "router", router);
        g.task(
                p + "topk",
                DeepSeek2BatchKernels::routerTopKSigmoidBatch,
                context,
                ws.wrapMoeLogitsBatch,
                f.routerBias()[l].asFloatArray(),
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedGateBatch,
                ws.batchStartPosHolder,
                experts,
                used,
                config.expertWeightsNorm() ? 1 : 0,
                config.expertWeightsScale());
        grid(g, p + "topk", lanes(batch * 32, 32));
        g.task(
                p + "group",
                Qwen35MoeBatchKernels::groupByExpert,
                context,
                ws.wrapMoeIdsBatch,
                ws.batchStartPosHolder,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                experts,
                used,
                batch);
        grid(
                g,
                p + "group",
                lanes(Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS));
        ByteArray gate = f.gateExperts()[l].asByteArray();
        ByteArray up = f.upExperts()[l].asByteArray();
        ByteArray down = f.downExperts()[l].asByteArray();
        if (q4) {
            g.task(
                    p + "gate_up",
                    Glm5NextBatchKernels::groupedGateUpClampedQ4_0,
                    context,
                    ws.wrapQ8ActBatch,
                    ws.wrapQ8ActScales,
                    ws.wrapMoeSortedToken,
                    ws.wrapMoeTiles,
                    gate,
                    up,
                    ws.wrapMoeHiddenBatch,
                    dim,
                    hidden,
                    0,
                    experts,
                    Float.POSITIVE_INFINITY);
        } else {
            g.task(
                    p + "gate_up",
                    Qwen35MoeBatchKernels::groupedGateUpQ8_0,
                    context,
                    ws.wrapQ8ActBatch,
                    ws.wrapQ8ActScales,
                    ws.wrapMoeSortedToken,
                    ws.wrapMoeTiles,
                    gate,
                    up,
                    ws.wrapMoeHiddenBatch,
                    dim,
                    hidden);
        }
        WorkerGrid gateUp =
                new WorkerGrid2D(
                        hidden
                                / Qwen35MoeBatchKernels.TILE_COLS
                                * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        gateUp.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        grid(g, p + "gate_up", gateUp);
        g.task(
                p + "hidden_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                hidden);
        grid(g, p + "hidden_q8", lanes(assignments * hidden, 128));
        if (q4) {
            g.task(
                    p + "down",
                    Glm5NextBatchKernels::groupedDownQ4_0,
                    context,
                    ws.wrapMoeHiddenQ8,
                    ws.wrapMoeHiddenScales,
                    ws.wrapMoeTiles,
                    down,
                    ws.wrapMoeOutBatch,
                    dim,
                    hidden,
                    0,
                    experts);
        } else {
            g.task(
                    p + "down",
                    Qwen35MoeBatchKernels::groupedDownQ8_0,
                    context,
                    ws.wrapMoeHiddenQ8,
                    ws.wrapMoeHiddenScales,
                    ws.wrapMoeTiles,
                    down,
                    ws.wrapMoeOutBatch,
                    dim,
                    hidden);
        }
        WorkerGrid downGrid =
                new WorkerGrid2D(
                        dim / Qwen35MoeBatchKernels.TILE_COLS * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        downGrid.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        grid(g, p + "down", downGrid);

        // The shared expert over every row of the chunk; its weight is one.
        gemm(
                g,
                p + "shared_gate",
                f.sharedGate()[l].asByteArray(),
                ws.wrapMoeSharedGateUp,
                shared,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemmGated(
                g,
                p + "shared_up",
                f.sharedUp()[l].asByteArray(),
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedGateUp,
                shared,
                dim);
        g.task(
                p + "shared_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                shared);
        grid(g, p + "shared_q8", lanes(batch * shared, 128));
        g.task(
                p + "shared_down",
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                f.sharedDown()[l].asByteArray(),
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedOut,
                batch,
                dim,
                shared,
                Int8GemmKernels.EPILOGUE_STORE,
                ws.wrapMoeSharedOut,
                1,
                ws.batchStartPosHolder,
                dim,
                0);
        grid(g, p + "shared_down", gemmGrid(dim, 1));
        g.task(
                p + "combine",
                Qwen35MoeBatchKernels::combine,
                context,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedGateBatch,
                ws.wrapXBatch,
                ws.batchStartPosHolder,
                dim,
                used);
        grid(g, p + "combine", lanes(batch * dim, 256));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    private void grid(TaskGraph g, String task, WorkerGrid grid) {
        grids.put(g.getTaskGraphName() + "." + task, grid);
    }

    private void rmsRows(
            TaskGraph g,
            String task,
            FloatArray in,
            FloatArray out,
            TornadoTensor weight,
            int n,
            int inStride,
            int outStride) {
        g.task(
                task,
                DeepSeek2BatchKernels::rmsNormRows,
                context,
                in,
                out,
                weight.asFloatArray(),
                state.workspace.batchStartPosHolder,
                n,
                inStride,
                outStride,
                config.rmsNormEps());
        grid(g, task, lanes(batch * DeepSeek2BatchKernels.GROUP, DeepSeek2BatchKernels.GROUP));
    }

    /** The chunk's {@code k}-wide rows of {@code x} in int8 blocks, for the GEMM after. */
    private void quantize(TaskGraph g, String task, FloatArray x, int k) {
        var ws = state.workspace;
        g.task(
                task,
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                x,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                k);
        grid(g, task, lanes(batch * k, 128));
    }

    private void toHalf(TaskGraph g, String task, FloatArray in, HalfFloatArray out, int n) {
        g.task(task, BatchMmaKernels::toHalf, context, in, out, n);
        grid(g, task, lanes(n, 256));
    }

    /** {@code out (+)= x8 . w} over the chunk; {@code n} outputs over {@code k} inputs. */
    private void gemm(
            TaskGraph g,
            String task,
            ByteArray weight,
            FloatArray out,
            int n,
            int k,
            int epilogue,
            boolean split) {
        var ws = state.workspace;
        int splits = split ? gemmSplits(n, k) : 1;
        FloatArray partial = splits > 1 ? ws.wrapQ8SplitPartial : out;
        g.task(
                task,
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                weight,
                out,
                out,
                batch,
                n,
                k,
                epilogue,
                partial,
                splits,
                ws.batchStartPosHolder,
                n,
                0);
        grid(g, task, gemmGrid(n, splits));
        if (splits > 1) {
            int total = batch * n;
            g.task(
                    task + "_reduce",
                    Int8GemmKernels::reduceSplitsQ8_0,
                    context,
                    ws.wrapQ8SplitPartial,
                    out,
                    total,
                    splits,
                    epilogue,
                    ws.batchStartPosHolder,
                    n);
            grid(g, task + "_reduce", lanes(total, 256));
        }
    }

    /** {@code out = silu(gate) * (x8 . w)} over the chunk. */
    private void gemmGated(
            TaskGraph g,
            String task,
            ByteArray weight,
            FloatArray out,
            FloatArray gate,
            int n,
            int k) {
        var ws = state.workspace;
        g.task(
                task,
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                weight,
                out,
                gate,
                batch,
                n,
                k,
                Int8GemmKernels.EPILOGUE_SWIGLU,
                out,
                1,
                ws.batchStartPosHolder,
                n,
                0);
        grid(g, task, gemmGrid(n, 1));
    }

    private void mma(
            TaskGraph g,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols,
            int batches) {
        mma(g, task, a, b, c, m, n, k, lda, ldb, ldc, aBatch, bBatch, cBatchCols, batches, 0, 0);
    }

    /** {@link #mma} with N or K bounded by the chunk's last position, rounded to whole tiles. */
    private void mma(
            TaskGraph g,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols,
            int batches,
            int boundN,
            int boundK) {
        g.task(
                task,
                BatchMmaKernels::gemmMMAStrided,
                context,
                a,
                b,
                c,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                aBatch,
                bBatch,
                cBatchCols,
                state.workspace.batchStartPosHolder,
                boundN,
                boundK,
                0,
                0,
                1,
                0);
        WorkerGrid3D grid = new WorkerGrid3D((m / 128) * 256, n / 128, batches);
        grid.setLocalWork(256, 1, 1);
        grid(g, task, grid);
    }

    private WorkerGrid gemmGrid(int n, int splits) {
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        (batch / Int8GemmKernels.I8_BM) * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN * splits);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        return grid;
    }

    private int gemmSplits(int outputs, int k) {
        int sms = org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
        if (sms <= 0) {
            return 1;
        }
        int tiles = (batch / Int8GemmKernels.I8_BM) * (outputs / Int8GemmKernels.I8_BN);
        int rounds = k / Int8GemmKernels.I8_BK;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= 8 && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return bestCost * 10 <= unsplit * 9 ? best : 1;
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
        return "int8 tensor-core GEMMs on Q8_0 weights; FP16 tensor-core latent attention";
    }
}
