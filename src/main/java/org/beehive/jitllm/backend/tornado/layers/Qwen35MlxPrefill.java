package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The {@code qwen35} batched prefill's projections and attention on Apple MLX ({@code
 * --with-native-libraries} on Metal), for the dense feed-forward layers.
 *
 * <p><b>What runs on MLX.</b> Every projection, each layer's in two or three GEMMs per branch:
 *
 * <ul>
 *   <li>attention layers: query/gate, key and value stacked into one GEMM, and the output
 *       projection; the attention itself through {@link MlxPrefillAttention} over the gathered
 *       cache where MLX's fused prefill attention takes the head width ({@link #runsAttention}),
 *       otherwise on the Metal kernels;
 *   <li>delta-net layers: the convolution input ({@code attn_qkv}), the output gate, beta and alpha
 *       stacked into one GEMM, and {@code ssm_out};
 *   <li>the feed-forward: gate and up stacked into one GEMM, SwiGLU, and the down projection.
 * </ul>
 *
 * The norms, the query/key norms and rotation, the causal convolution, the delta rule and the gates
 * stay on the Metal kernels, which read and write the same FP32 buffers as before.
 *
 * <p><b>FP16 range.</b> MLX multiplies FP16 into FP16. The normed chunk is narrowed as it is, as in
 * the other MLX families. The inputs of the projections that add into the residual stream — the
 * gated attention output, the gated delta-net readout and SwiGLU's product — are scaled per row so
 * that their largest magnitude is at most {@link MlxPrefillKernels#PROJECTION_INPUT_LIMIT}, and the
 * scale is undone in FP32 when the product is added to the residual stream.
 *
 * <p><b>Weights.</b> Each stack in MLX's form ({@link MlxPrefill#projections()}): Q8_0 or Q4_0
 * decoded once into FP16, or repacked into MLX's affine format. The prefill graphs then bind none
 * of the model's own projections, so the decode graphs upload those themselves ({@link
 * #replacesProjections}). A projection of another type is refused by name rather than run on the
 * JIT kernels, so that answer is never wrong.
 *
 * <p><b>Buffers.</b> MLX takes an array only at its exact extent, so every operand and product has
 * its own FP16 buffer of {@code batchSize} rows. Each is uploaded by the first layer graph that
 * uses it and bound from that graph by every later one.
 */
// @formatter:on
public final class Qwen35MlxPrefill {

    /** Threads per row of the row-scale reductions. */
    private static final int ROW_THREADS = 256;

    private final Qwen35State state;
    private final Qwen35TornadoWeights weights;
    private final Qwen35Configuration config;
    private final KernelContext context;
    private final int batchSize;
    private final int firstLayer;
    private final int firstAttention;
    private final int firstRecurrent;

    private final int dim;
    private final int hidden;
    private final int queryGateDim;
    private final int kvDim;
    private final int attnDim;
    private final int convDim;
    private final int valueDim;
    private final int valueHeads;

    private final MlxPrefillProjections.Matrix[] inStack;
    private final MlxPrefillProjections.Matrix[] outProjection;
    private final MlxPrefillProjections.Matrix[] gateUp;
    private final MlxPrefillProjections.Matrix[] down;

    private final MlxPrefillAttention attention;

    private final MlxPrefill.Projections form;

    // Every layer.
    private final HalfFloatArray normedHalf; // B × dim
    private final FloatArray inScale; // B
    private final HalfFloatArray dimOut; // B × dim
    private final HalfFloatArray gateUpOut; // B × 2·hidden
    private final HalfFloatArray hbHalf; // B × hidden
    private final FloatArray downScale; // B

    // Attention layers.
    private final HalfFloatArray xbHalf; // B × attnDim
    private final HalfFloatArray attnInOut; // B × (queryGateDim + 2·kvDim)

    // Delta-net layers.
    private final HalfFloatArray ssmOutHalf; // B × valueDim
    private final HalfFloatArray ssmInOut; // B × (convDim + valueDim + 2·valueHeads)

    /**
     * The MLX prefill a session running {@code policy} builds over this model, or empty when its
     * batched prefill keeps the Metal kernels. The one answer the prefill layers, the decode layers
     * that bind their weights and the memory preflight all read.
     */
    public static Optional<MlxPrefill> resolve(
            ExecutionPolicy policy, DataType weights, Qwen35Configuration config) {
        return MlxPrefill.resolve(policy, weights, config);
    }

    /**
     * Whether the batched prefill under {@code policy} multiplies the projections on MLX, so the
     * decode graphs upload the model's projections rather than bind them from it.
     */
    public static boolean replacesProjections(
            ExecutionPolicy policy, DataType weights, Qwen35Configuration config) {
        return resolve(policy, weights, config).isPresent();
    }

    public Qwen35MlxPrefill(
            MlxPrefill mlx,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            KernelContext context,
            int batchSize,
            int firstLayer,
            int endLayer) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.context = context;
        this.batchSize = batchSize;
        this.firstLayer = firstLayer;
        this.dim = config.dim();
        this.hidden = config.hiddenDim();
        this.queryGateDim = config.queryGateDim();
        this.kvDim = config.kvDim();
        this.attnDim = config.attentionOutputInputDim();
        this.convDim = config.deltaNetConvDim();
        this.valueDim = config.deltaNetValueDim();
        this.valueHeads = config.numberOfValueHeads();

        int layers = config.numberOfLayers();
        this.inStack = new MlxPrefillProjections.Matrix[layers];
        this.outProjection = new MlxPrefillProjections.Matrix[layers];
        this.gateUp = new MlxPrefillProjections.Matrix[layers];
        this.down = new MlxPrefillProjections.Matrix[layers];
        int attentionLayer = -1;
        int recurrentLayer = -1;
        MlxPrefill.Projections form = mlx.projections();
        this.form = form;
        for (int l = firstLayer; l < endLayer; l++) {
            String block = "blk." + l;
            if (config.isRecurrentLayer(l)) {
                recurrentLayer = recurrentLayer < 0 ? l : recurrentLayer;
                inStack[l] =
                        MlxPrefillProjections.matrix(
                                form,
                                require(
                                        mlx,
                                        block + ".attn_qkv|attn_gate|ssm_beta|ssm_alpha",
                                        weights.ssmQkv[l],
                                        weights.ssmGate[l],
                                        weights.ssmBeta[l],
                                        weights.ssmAlpha[l]));
                outProjection[l] =
                        MlxPrefillProjections.matrix(
                                form, require(mlx, block + ".ssm_out", weights.ssmOut[l]));
            } else {
                attentionLayer = attentionLayer < 0 ? l : attentionLayer;
                inStack[l] =
                        MlxPrefillProjections.matrix(
                                form,
                                require(
                                        mlx,
                                        block + ".attn_q|attn_k|attn_v",
                                        weights.wqLayered[l],
                                        weights.wkLayered[l],
                                        weights.wvLayered[l]));
                outProjection[l] =
                        MlxPrefillProjections.matrix(
                                form, require(mlx, block + ".attn_output", weights.woLayered[l]));
            }
            gateUp[l] =
                    MlxPrefillProjections.matrix(
                            form,
                            require(
                                    mlx,
                                    block + ".ffn_gate|ffn_up",
                                    weights.w1Layered[l],
                                    weights.w3Layered[l]));
            down[l] =
                    MlxPrefillProjections.matrix(
                            form, require(mlx, block + ".ffn_down", weights.w2Layered[l]));
        }
        this.firstAttention = attentionLayer;
        this.firstRecurrent = recurrentLayer;

        this.normedHalf = new HalfFloatArray(batchSize * dim);
        this.inScale = new FloatArray(batchSize);
        this.dimOut = new HalfFloatArray(batchSize * dim);
        this.gateUpOut = new HalfFloatArray(batchSize * 2 * hidden);
        this.hbHalf = new HalfFloatArray(batchSize * hidden);
        this.downScale = new FloatArray(batchSize);
        this.xbHalf = new HalfFloatArray(batchSize * attnDim);
        this.attnInOut = new HalfFloatArray(batchSize * (queryGateDim + 2 * kvDim));
        this.ssmOutHalf = new HalfFloatArray(batchSize * valueDim);
        this.ssmInOut = new HalfFloatArray(batchSize * (convDim + valueDim + 2 * valueHeads));
        this.attention =
                attentionLayer < 0 || !MLX_ATTENTION_HEADS.contains(config.numberOfHeadsKey())
                        ? null
                        : new MlxPrefillAttention(
                                state,
                                context,
                                batchSize,
                                config.contextLength(),
                                config.numberOfHeads(),
                                config.numberOfKeyValueHeads(),
                                config.numberOfHeadsKey(),
                                (float) (1.0 / Math.sqrt(config.numberOfHeadsKey())));
    }

    /** The head widths MLX's fused prefill attention is built for. */
    private static final java.util.Set<Integer> MLX_ATTENTION_HEADS = java.util.Set.of(64, 80, 128);

    /** Whether the attention layers attend on MLX, or keep the Metal kernels. */
    public boolean runsAttention() {
        return attention != null;
    }

    /** The form the GEMMs read the weights in. */
    public MlxPrefill.Projections form() {
        return form;
    }

    /**
     * What one session's MLX prefill allocates besides the model's weights: the projections in
     * MLX's form (resident FP16 copies or the affine repack), the FP16 operands and products, and
     * the attention's gathered keys and values.
     */
    public static org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace workspace(
            DataType weights, Qwen35Configuration config, int batchSize) {
        long rows = batchSize;
        long halves =
                rows
                        * (2L * config.dim()
                                + 3L * config.hiddenDim()
                                + config.attentionOutputInputDim()
                                + config.queryGateDim()
                                + 2L * config.kvDim()
                                + 2L * config.deltaNetValueDim()
                                + config.deltaNetConvDim()
                                + 2L * config.numberOfValueHeads());
        long attentionQ = 2L * rows * config.numberOfHeads() * config.numberOfHeadsKey();
        long attentionKv = 2L * ((long) config.contextLength() + rows) * config.kvDim();
        long fixed = Short.BYTES * (halves + attentionQ + attentionKv) + 2L * Float.BYTES * rows;
        long projections = config.projectionWeights();
        return new org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace(
                projections,
                fixed + FP16Projections.residentBytes(projections),
                fixed + MlxPrefill.affineBytes(weights, projections));
    }

    /** Refuses, by name, a projection MLX cannot take in the form it was resolved for. */
    private static TornadoTensor[] require(MlxPrefill mlx, String name, TornadoTensor... tensors) {
        for (TornadoTensor t : tensors) {
            if (t == null) {
                throw new IllegalStateException("qwen35 MLX batched prefill: " + name + " missing");
            }
            if (t.isPacked()) {
                throw PackedWeights.noPackedKernel("The qwen35 MLX batched prefill (" + name + ")");
            }
            if (t.dataType() != mlx.weights()) {
                throw new IllegalStateException(
                        "the qwen35 MLX batched prefill was resolved for "
                                + mlx.weights()
                                + " weights, but "
                                + name
                                + " holds a "
                                + t.dataType()
                                + " tensor");
            }
        }
        return tensors;
    }

    /** Declares the buffers and this layer's weights in layer graph {@code layerIndex}. */
    public void declare(TaskGraph layer, int layerIndex) {
        bind(layer, layerIndex, firstLayer, normedHalf, inScale, dimOut, gateUpOut, hbHalf);
        bind(layer, layerIndex, firstLayer, downScale);
        if (config.isRecurrentLayer(layerIndex)) {
            bind(layer, layerIndex, firstRecurrent, ssmOutHalf, ssmInOut);
        } else {
            bind(layer, layerIndex, firstAttention, xbHalf, attnInOut);
            if (attention != null) {
                attention.declare(
                        layer, layerIndex == firstAttention, "batchLayer_" + firstAttention);
            }
        }
        List<Object> arrays = new ArrayList<>();
        for (MlxPrefillProjections.Matrix m :
                new MlxPrefillProjections.Matrix[] {
                    inStack[layerIndex],
                    outProjection[layerIndex],
                    gateUp[layerIndex],
                    down[layerIndex]
                }) {
            arrays.addAll(List.of(m.arrays()));
        }
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, arrays.toArray());
    }

    private static void bind(TaskGraph layer, int layerIndex, int first, Object... buffers) {
        if (layerIndex == first) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffers);
        } else {
            layer.consumeFromDevice("batchLayer_" + first, buffers);
        }
    }

    /**
     * Query/gate, key and value of the normed chunk into {@code wrapQGateBatch}, {@code wrapKBatch}
     * and {@code wrapVBatch}.
     */
    public void attentionProjections(TaskGraph layer, int layerIndex) {
        narrow(layer, "mlx_attn_in_half");
        int width = queryGateDim + 2 * kvDim;
        MlxPrefillProjections.gemm(
                layer,
                "mlx_attn_qkv",
                normedHalf,
                inStack[layerIndex],
                attnInOut,
                batchSize,
                dim,
                width);
        split(
                layer,
                "mlx_attn_q_split",
                attnInOut,
                state.workspace.wrapQGateBatch,
                width,
                0,
                queryGateDim);
        split(
                layer,
                "mlx_attn_k_split",
                attnInOut,
                state.workspace.wrapKBatch,
                width,
                queryGateDim,
                kvDim);
        split(
                layer,
                "mlx_attn_v_split",
                attnInOut,
                state.workspace.wrapVBatch,
                width,
                queryGateDim + kvDim,
                kvDim);
    }

    /**
     * The attention of attention layer {@code layerIndex}, from the rotated queries in {@code
     * wrapAttnQBatch} and the paged cache into {@code wrapXbBatch}.
     */
    public void attention(TaskGraph layer, int layerIndex) {
        attention.attention(
                layer,
                "mlx_attention",
                config.keyValueLayerIndex(layerIndex),
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapXbBatch);
    }

    /** The output projection of the gated attention in {@code wrapXbBatch}, into the residual. */
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        stage(layer, "mlx_attn_out_in", state.workspace.wrapXbBatch, xbHalf, attnDim);
        MlxPrefillProjections.gemm(
                layer,
                "mlx_attn_output",
                xbHalf,
                outProjection[layerIndex],
                dimOut,
                batchSize,
                attnDim,
                dim);
        residual(layer, "mlx_attn_out_residual", inScale);
    }

    /**
     * The convolution input, the output gate, beta and alpha of the normed chunk into {@code
     * wrapSsmQkvBatch}, {@code wrapSsmZBatch}, {@code wrapSsmBetaBatch} and {@code
     * wrapSsmAlphaBatch}.
     */
    public void deltaNetProjections(TaskGraph layer, int layerIndex) {
        narrow(layer, "mlx_ssm_in_half");
        int width = convDim + valueDim + 2 * valueHeads;
        MlxPrefillProjections.gemm(
                layer,
                "mlx_ssm_in_proj",
                normedHalf,
                inStack[layerIndex],
                ssmInOut,
                batchSize,
                dim,
                width);
        split(
                layer,
                "mlx_ssm_qkv_split",
                ssmInOut,
                state.workspace.wrapSsmQkvBatch,
                width,
                0,
                convDim);
        split(
                layer,
                "mlx_ssm_gate_split",
                ssmInOut,
                state.workspace.wrapSsmZBatch,
                width,
                convDim,
                valueDim);
        split(
                layer,
                "mlx_ssm_beta_split",
                ssmInOut,
                state.workspace.wrapSsmBetaBatch,
                width,
                convDim + valueDim,
                valueHeads);
        split(
                layer,
                "mlx_ssm_alpha_split",
                ssmInOut,
                state.workspace.wrapSsmAlphaBatch,
                width,
                convDim + valueDim + valueHeads,
                valueHeads);
    }

    /**
     * {@code ssm_out} of the gated delta-net readout in {@code wrapSsmOutBatch}, into the residual.
     */
    public void deltaNetOutput(TaskGraph layer, int layerIndex) {
        stage(layer, "mlx_ssm_out_in", state.workspace.wrapSsmOutBatch, ssmOutHalf, valueDim);
        MlxPrefillProjections.gemm(
                layer,
                "mlx_ssm_out",
                ssmOutHalf,
                outProjection[layerIndex],
                dimOut,
                batchSize,
                valueDim,
                dim);
        residual(layer, "mlx_ssm_out_residual", inScale);
    }

    /** The dense SwiGLU feed-forward of the normed chunk, into the residual. */
    public void feedForward(TaskGraph layer, int layerIndex) {
        narrow(layer, "mlx_ffn_in_half");
        MlxPrefillProjections.gemm(
                layer,
                "mlx_ffn_gate_up",
                normedHalf,
                gateUp[layerIndex],
                gateUpOut,
                batchSize,
                dim,
                2 * hidden);
        layer.task(
                "mlx_ffn_row_scale",
                MlxPrefillKernels::swigluRowScale,
                context,
                gateUpOut,
                downScale,
                hidden,
                batchSize,
                ROW_THREADS);
        layer.task(
                "mlx_ffn_swiglu",
                MlxPrefillKernels::swigluScaled,
                context,
                gateUpOut,
                hbHalf,
                downScale,
                hidden,
                batchSize * hidden);
        MlxPrefillProjections.gemm(
                layer, "mlx_ffn_down", hbHalf, down[layerIndex], dimOut, batchSize, hidden, dim);
        residual(layer, "mlx_ffn_residual", downScale);
    }

    /** The normed chunk narrowed to FP16 for the GEMMs that read it. */
    private void narrow(TaskGraph layer, String task) {
        layer.task(
                task,
                MlxPrefillKernels::toHalf,
                context,
                state.workspace.wrapNormedBatch,
                normedHalf,
                batchSize * dim);
    }

    /** {@code half = FP16(x * rowScale)}, with the row factor that keeps it inside FP16. */
    private void stage(TaskGraph layer, String task, FloatArray x, HalfFloatArray half, int width) {
        layer.task(
                task + "_scale",
                MlxPrefillKernels::rowScale,
                context,
                x,
                inScale,
                width,
                batchSize,
                ROW_THREADS);
        layer.task(
                task + "_half",
                MlxPrefillKernels::toHalfScaled,
                context,
                x,
                half,
                inScale,
                width,
                batchSize * width);
    }

    private void split(
            TaskGraph layer,
            String task,
            HalfFloatArray src,
            FloatArray out,
            int srcWidth,
            int offset,
            int width) {
        layer.task(
                task,
                MlxPrefillKernels::splitRows,
                context,
                src,
                out,
                srcWidth,
                offset,
                width,
                batchSize * width);
    }

    private void residual(TaskGraph layer, String task, FloatArray rowScale) {
        layer.task(
                task,
                MlxPrefillKernels::addRowScaledResidual,
                context,
                state.workspace.wrapXBatch,
                dimOut,
                rowScale,
                1.0f,
                dim,
                batchSize * dim);
    }

    /**
     * The grids of layer {@code layerIndex}'s glue kernels, under graph-qualified prefix {@code p}.
     */
    public void addGrids(GridScheduler scheduler, int layerIndex, String p) {
        WorkerGrid rows = WorkerGridFactory.genericWorker(batchSize * ROW_THREADS, ROW_THREADS);
        scheduler.addWorkerGrid(p + "mlx_ffn_in_half", elementwise(batchSize * dim));
        scheduler.addWorkerGrid(p + "mlx_ffn_row_scale", rows);
        scheduler.addWorkerGrid(p + "mlx_ffn_swiglu", elementwise(batchSize * hidden));
        scheduler.addWorkerGrid(p + "mlx_ffn_residual", elementwise(batchSize * dim));
        if (config.isRecurrentLayer(layerIndex)) {
            scheduler.addWorkerGrid(p + "mlx_ssm_in_half", elementwise(batchSize * dim));
            scheduler.addWorkerGrid(p + "mlx_ssm_qkv_split", elementwise(batchSize * convDim));
            scheduler.addWorkerGrid(p + "mlx_ssm_gate_split", elementwise(batchSize * valueDim));
            scheduler.addWorkerGrid(p + "mlx_ssm_beta_split", elementwise(batchSize * valueHeads));
            scheduler.addWorkerGrid(p + "mlx_ssm_alpha_split", elementwise(batchSize * valueHeads));
            addStage(scheduler, p + "mlx_ssm_out_in", valueDim, rows);
            scheduler.addWorkerGrid(p + "mlx_ssm_out_residual", elementwise(batchSize * dim));
        } else {
            scheduler.addWorkerGrid(p + "mlx_attn_in_half", elementwise(batchSize * dim));
            scheduler.addWorkerGrid(p + "mlx_attn_q_split", elementwise(batchSize * queryGateDim));
            scheduler.addWorkerGrid(p + "mlx_attn_k_split", elementwise(batchSize * kvDim));
            scheduler.addWorkerGrid(p + "mlx_attn_v_split", elementwise(batchSize * kvDim));
            if (attention != null) {
                attention.addGrids(scheduler, p, "mlx_attention");
            }
            addStage(scheduler, p + "mlx_attn_out_in", attnDim, rows);
            scheduler.addWorkerGrid(p + "mlx_attn_out_residual", elementwise(batchSize * dim));
        }
    }

    private void addStage(GridScheduler scheduler, String task, int width, WorkerGrid rows) {
        scheduler.addWorkerGrid(task + "_scale", rows);
        scheduler.addWorkerGrid(task + "_half", elementwise(batchSize * width));
    }

    private static WorkerGrid elementwise(int total) {
        return WorkerGridFactory.genericWorker((total + 255) / 256 * 256, 256);
    }
}
