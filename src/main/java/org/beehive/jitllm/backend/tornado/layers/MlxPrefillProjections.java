package org.beehive.jitllm.backend.tornado.layers;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.Mlx;

// @formatter:off
/**
 * The four batched-prefill projections of a Llama-style layer as MLX GEMMs, for any family whose
 * layer reads {@code wrapXBatch}, projects into {@code wrapQBatch}/{@code wrapKBatch}/{@code
 * wrapVBatch}, attends into {@code wrapXbBatch} and adds the output and down projections back into
 * {@code wrapXBatch}.
 *
 * <p>Q, K and V, and gate and up, are stacked so each pair or triple is one GEMM. Which form of the
 * weights the GEMMs read is {@link MlxPrefill#projections()}: F16 weights through {@link
 * Mlx#matmulTransposed}; Q8_0 and Q4_0 weights either decoded once into FP16 copies, or repacked
 * once, on the host, into MLX's affine format with groups of 32 for {@link Mlx#quantizedMatmul}.
 * The repack is exact: Q8_0's {@code s * q} is {@code s * (q + 128) - 128 * s}, and Q4_0's {@code s
 * * (q - 8)} is {@code s * q - 8 * s}, with the bias {@code -128 * s} or {@code -8 * s} exact in
 * FP16.
 *
 * <p>Built from an {@link MlxPrefill} the caller resolved, and only from it: a tensor that does not
 * match it — another type, or packed tiles — is refused, never handed back to the JIT kernels, so
 * the decode graphs that ask the same {@link MlxPrefill} which weights these graphs bind ({@link
 * #notBound}) are always right.
 *
 * <p>The family keeps its RMS reductions ({@code batch_attn_rms}, {@code batch_ffn_rms}), RoPE and
 * attention; this replaces the RMS applies and the projection tasks around them. The FP16 operands
 * and products are shared by every layer graph: the first declares them, each later one binds them
 * from its predecessor.
 */
// @formatter:on
public final class MlxPrefillProjections {

    private static final int GROUP = MlxPrefill.GROUP;

    /** Threads per row of the row-scale reductions. */
    private static final int ROW_THREADS = 256;

    /** One projection's weights in MLX's form: FP16, or affine-quantized. */
    record Matrix(
            HalfFloatArray fp16,
            IntArray q,
            HalfFloatArray scales,
            HalfFloatArray biases,
            int bits) {

        Object[] arrays() {
            return fp16 != null ? new Object[] {fp16} : new Object[] {q, scales, biases};
        }
    }

    private final State state;
    private final TornadoWeights weights;
    private final KernelContext context;
    private final int batchSize;
    private final int dim;
    private final int qDim;
    private final int kvDim;
    private final int hidden;
    private final float residualScale;
    private final Matrix[] qkv;
    private final Matrix[] wo;
    private final Matrix[] gateUp;
    private final Matrix[] w2;

    private final HalfFloatArray xHalf; // B × dim
    private final HalfFloatArray qkvHalf; // B × (qDim + 2·kvDim)
    private final HalfFloatArray attnHalf; // B × qDim
    private final HalfFloatArray outHalf; // B × dim
    private final HalfFloatArray gateUpHalf; // B × 2·hidden
    private final HalfFloatArray hbHalf; // B × hidden
    private final FloatArray attnRowScale; // B
    private final FloatArray downRowScale; // B

    /**
     * One layer's projection weights: Q, K and V (one fused tensor, or three stacked in that
     * order), the output projection, gate and up (fused, or stacked in that order) and the down
     * projection.
     */
    public record LayerWeights(
            TornadoTensor[] qkv, TornadoTensor wo, TornadoTensor[] gateUp, TornadoTensor w2) {

        /** The Llama-style layer: separate Q, K, V, gate and up tensors. */
        public static LayerWeights of(TornadoWeights weights, int i) {
            return new LayerWeights(
                    new TornadoTensor[] {
                        weights.wqLayered[i], weights.wkLayered[i], weights.wvLayered[i]
                    },
                    weights.woLayered[i],
                    new TornadoTensor[] {weights.w1Layered[i], weights.w3Layered[i]},
                    weights.w2Layered[i]);
        }

        List<TornadoTensor> all() {
            List<TornadoTensor> all = new ArrayList<>(List.of(qkv));
            all.add(wo);
            all.addAll(List.of(gateUp));
            all.add(w2);
            return all;
        }
    }

    // @formatter:off
    /**
     * The model weights of {@code layer} that MLX prefill graphs built for {@code mlx} do not bind,
     * each as its tensor and as its native array, so a graph that would bind them from those graphs
     * must upload them instead.
     *
     * <ul>
     *   <li>Q8_0 and Q4_0: every projection, since the GEMMs read the FP16 copies or the affine
     *       repack;
     *   <li>F16: the tensors stacked into one copy — split Q/K/V and gate/up. A fused tensor and
     *       the output and down projections are multiplied as they are, so the prefill graph binds
     *       them.
     * </ul>
     */
    // @formatter:on
    public static Object[] notBound(MlxPrefill mlx, LayerWeights layer) {
        List<TornadoTensor> copied = new ArrayList<>();
        if (mlx.projections() == MlxPrefill.Projections.FP16) {
            if (layer.qkv().length > 1) {
                copied.addAll(List.of(layer.qkv()));
            }
            if (layer.gateUp().length > 1) {
                copied.addAll(List.of(layer.gateUp()));
            }
        } else {
            copied.addAll(layer.all());
        }
        List<Object> notBound = new ArrayList<>(2 * copied.size());
        for (TornadoTensor t : copied) {
            notBound.add(t);
            notBound.add(t.dataType() == DataType.F16 ? t.asHalfFloatArray() : t.asByteArray());
        }
        return notBound.toArray();
    }

    /**
     * {@link #of(MlxPrefill, State, TornadoWeights, KernelContext, int, int, int, int, int, int)}
     * when the batched prefill runs on MLX, or null when {@code mlx} is empty and the family's JIT
     * projections run.
     */
    public static MlxPrefillProjections ifPresent(
            java.util.Optional<MlxPrefill> mlx,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int batchSize,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden) {
        return mlx.map(
                        m ->
                                of(
                                        m, state, weights, context, batchSize, layers, dim, qDim,
                                        kvDim, hidden))
                .orElse(null);
    }

    /** {@link #ifPresent} for a family with its own layer weights and residual scale. */
    public static MlxPrefillProjections ifPresent(
            java.util.Optional<MlxPrefill> mlx,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int batchSize,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden,
            float residualScale,
            IntFunction<LayerWeights> layerWeights) {
        return mlx.map(
                        m ->
                                of(
                                        m,
                                        state,
                                        weights,
                                        context,
                                        batchSize,
                                        layers,
                                        dim,
                                        qDim,
                                        kvDim,
                                        hidden,
                                        residualScale,
                                        layerWeights))
                .orElse(null);
    }

    /**
     * The projections of a Llama-style layer, for a prefill that runs on MLX as {@code mlx}
     * describes.
     */
    public static MlxPrefillProjections of(
            MlxPrefill mlx,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int batchSize,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden) {
        return of(
                mlx,
                state,
                weights,
                context,
                batchSize,
                layers,
                dim,
                qDim,
                kvDim,
                hidden,
                1.0f,
                i -> LayerWeights.of(weights, i));
    }

    /**
     * {@link #of(MlxPrefill, State, TornadoWeights, KernelContext, int, int, int, int, int, int)}
     * for a family whose layer weights {@code layerWeights} supplies, and whose projections are
     * added into the residual stream scaled by {@code residualScale}.
     *
     * @throws UnsupportedOperationException if a projection is held in packed tiles, which MLX does
     *     not read
     * @throws IllegalStateException if a projection is not of {@code mlx}'s weight type
     */
    public static MlxPrefillProjections of(
            MlxPrefill mlx,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int batchSize,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden,
            float residualScale,
            IntFunction<LayerWeights> layerWeights) {
        LayerWeights[] perLayer = new LayerWeights[layers];
        for (int i = 0; i < layers; i++) {
            perLayer[i] = layerWeights.apply(i);
            for (TornadoTensor t : perLayer[i].all()) {
                if (state.workspace.tileLayout.tiled(t)) {
                    throw PackedWeights.noPackedKernel("The MLX batched prefill");
                }
                if (t.dataType() != mlx.weights()) {
                    throw new IllegalStateException(
                            "the MLX batched prefill was resolved for "
                                    + mlx.weights()
                                    + " weights, but layer "
                                    + i
                                    + " holds a "
                                    + t.dataType()
                                    + " projection");
                }
            }
        }
        return new MlxPrefillProjections(
                mlx,
                state,
                weights,
                context,
                batchSize,
                dim,
                qDim,
                kvDim,
                hidden,
                residualScale,
                perLayer);
    }

    private MlxPrefillProjections(
            MlxPrefill mlx,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int batchSize,
            int dim,
            int qDim,
            int kvDim,
            int hidden,
            float residualScale,
            LayerWeights[] perLayer) {
        this.state = state;
        this.weights = weights;
        this.context = context;
        this.batchSize = batchSize;
        this.dim = dim;
        this.qDim = qDim;
        this.kvDim = kvDim;
        this.hidden = hidden;
        this.residualScale = residualScale;
        int layers = perLayer.length;
        this.qkv = new Matrix[layers];
        this.wo = new Matrix[layers];
        this.gateUp = new Matrix[layers];
        this.w2 = new Matrix[layers];
        MlxPrefill.Projections form = mlx.projections();
        for (int i = 0; i < layers; i++) {
            qkv[i] = matrix(form, perLayer[i].qkv());
            wo[i] = matrix(form, perLayer[i].wo());
            gateUp[i] = matrix(form, perLayer[i].gateUp());
            w2[i] = matrix(form, perLayer[i].w2());
        }
        this.xHalf = new HalfFloatArray(batchSize * dim);
        this.qkvHalf = new HalfFloatArray(batchSize * (qDim + 2 * kvDim));
        this.attnHalf = new HalfFloatArray(batchSize * qDim);
        this.outHalf = new HalfFloatArray(batchSize * dim);
        this.gateUpHalf = new HalfFloatArray(batchSize * 2 * hidden);
        this.hbHalf = new HalfFloatArray(batchSize * hidden);
        this.attnRowScale = new FloatArray(batchSize);
        this.downRowScale = new FloatArray(batchSize);
    }

    /** The tensors, rows stacked in order, as one matrix in {@code form}. */
    static Matrix matrix(MlxPrefill.Projections form, TornadoTensor... tensors) {
        return switch (form) {
            case FP16 -> {
                HalfFloatArray[] parts = new HalfFloatArray[tensors.length];
                for (int i = 0; i < parts.length; i++) {
                    parts[i] = tensors[i].asHalfFloatArray();
                }
                yield new Matrix(
                        parts.length == 1 ? parts[0] : HalfFloatArray.concat(parts),
                        null,
                        null,
                        null,
                        16);
            }
            case FP16_DECODED -> new Matrix(decode(tensors), null, null, null, 16);
            case AFFINE -> affine(tensors);
        };
    }

    /**
     * Q8_0 or Q4_0 tensors, rows stacked in order, repacked into MLX's affine format: {@code bits}
     * -bit unsigned values packed little-end first into 32-bit words, and per group of 32 an FP16
     * scale and bias with {@code w = scale * q + bias}.
     */
    static Matrix affine(TornadoTensor... tensors) {
        boolean q8 = tensors[0].dataType() == DataType.Q8_0;
        int bits = q8 ? 8 : 4;
        int blockBytes = q8 ? 34 : 18;
        long blocks = 0;
        for (TornadoTensor t : tensors) {
            blocks += t.asByteArray().getSize() / blockBytes;
        }
        IntArray q = new IntArray(Math.toIntExact(blocks * GROUP * bits / 32));
        HalfFloatArray scales = new HalfFloatArray(Math.toIntExact(blocks));
        HalfFloatArray biases = new HalfFloatArray(Math.toIntExact(blocks));
        int word = 0;
        int group = 0;
        int[] values = new int[GROUP];
        for (TornadoTensor t : tensors) {
            MemorySegment src = t.asByteArray().getSegment();
            long n = t.asByteArray().getSize() / blockBytes;
            for (long b = 0; b < n; b++) {
                long base = b * blockBytes;
                float scale = Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, base));
                if (q8) {
                    for (int j = 0; j < GROUP; j++) {
                        values[j] = src.get(ValueLayout.JAVA_BYTE, base + 2 + j) + 128;
                    }
                } else {
                    for (int j = 0; j < GROUP / 2; j++) {
                        int packed = src.get(ValueLayout.JAVA_BYTE, base + 2 + j) & 0xFF;
                        values[j] = packed & 0xF;
                        values[j + GROUP / 2] = packed >>> 4;
                    }
                }
                int perWord = 32 / bits;
                for (int w = 0; w < GROUP / perWord; w++) {
                    int packed = 0;
                    for (int e = 0; e < perWord; e++) {
                        packed |= values[w * perWord + e] << (e * bits);
                    }
                    q.set(word++, packed);
                }
                scales.set(group, new HalfFloat(scale));
                biases.set(group, new HalfFloat(-(q8 ? 128 : 8) * scale));
                group++;
            }
        }
        return new Matrix(null, q, scales, biases, bits);
    }

    /** Q8_0 or Q4_0 tensors, rows stacked in order, decoded to FP16. */
    static HalfFloatArray decode(TornadoTensor... tensors) {
        boolean q8 = tensors[0].dataType() == DataType.Q8_0;
        int blockBytes = q8 ? 34 : 18;
        long blocks = 0;
        for (TornadoTensor t : tensors) {
            blocks += t.asByteArray().getSize() / blockBytes;
        }
        HalfFloatArray out = new HalfFloatArray(Math.toIntExact(blocks * GROUP));
        MemorySegment dst = out.getSegment();
        long e = 0;
        for (TornadoTensor t : tensors) {
            MemorySegment src = t.asByteArray().getSegment();
            long n = t.asByteArray().getSize() / blockBytes;
            for (long b = 0; b < n; b++, e += GROUP) {
                long base = b * blockBytes;
                float scale = Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, base));
                for (int j = 0; j < GROUP; j++) {
                    int q;
                    if (q8) {
                        q = src.get(ValueLayout.JAVA_BYTE, base + 2 + j);
                    } else {
                        int packed = src.get(ValueLayout.JAVA_BYTE, base + 2 + (j % 16)) & 0xFF;
                        q = (j < 16 ? packed & 0xF : packed >>> 4) - 8;
                    }
                    dst.set(
                            ValueLayout.JAVA_SHORT_UNALIGNED,
                            (e + j) * 2,
                            Float.floatToFloat16(scale * q));
                }
            }
        }
        return out;
    }

    /**
     * Declares the shared FP16 buffers and this layer's weights in layer graph {@code layerIndex}.
     */
    public void declare(TaskGraph layer, int layerIndex, String predecessor) {
        Object[] buffers = {
            xHalf, qkvHalf, attnHalf, outHalf, gateUpHalf, hbHalf, attnRowScale, downRowScale
        };
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffers);
        } else {
            layer.consumeFromDevice(predecessor, buffers);
        }
        List<Object> layerWeights = new ArrayList<>();
        for (Matrix m :
                new Matrix[] {
                    qkv[layerIndex], wo[layerIndex], gateUp[layerIndex], w2[layerIndex]
                }) {
            layerWeights.addAll(List.of(m.arrays()));
        }
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights.toArray());
    }

    /**
     * The attention input: RMS apply (after the family's {@code batch_attn_rms}) into FP16, the
     * stacked QKV GEMM, and its split into {@code wrapQBatch}, {@code wrapKBatch}, {@code
     * wrapVBatch}.
     */
    public void qkv(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                xHalf,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);
        gemm(layer, "batch_qkv", xHalf, qkv[layerIndex], qkvHalf, dim, qDim + 2 * kvDim);
        layer.task(
                "batch_qkv_split",
                MlxPrefillKernels::splitQkv,
                context,
                qkvHalf,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                qDim,
                kvDim,
                batchSize * (qDim + 2 * kvDim));
    }

    /**
     * The output projection of {@code wrapXbBatch}, added into {@code wrapXBatch}. A row that would
     * overflow FP16 in the GEMM is scaled into range first, and the scale is undone in the FP32
     * residual add.
     */
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_attn_row_scale",
                MlxPrefillKernels::rowScale,
                context,
                state.workspace.wrapXbBatch,
                attnRowScale,
                qDim,
                batchSize,
                ROW_THREADS);
        layer.task(
                "batch_attn_half",
                MlxPrefillKernels::toHalfScaled,
                context,
                state.workspace.wrapXbBatch,
                attnHalf,
                attnRowScale,
                qDim,
                batchSize * qDim);
        gemm(layer, "batch_attn_out", attnHalf, wo[layerIndex], outHalf, qDim, dim);
        layer.task(
                "batch_attn_residual",
                MlxPrefillKernels::addRowScaledResidual,
                context,
                state.workspace.wrapXBatch,
                outHalf,
                attnRowScale,
                residualScale,
                dim,
                batchSize * dim);
    }

    /**
     * The feed-forward block after the family's {@code batch_ffn_rms}: RMS apply into FP16, the
     * stacked gate/up GEMM, SwiGLU, and the down GEMM added into {@code wrapXBatch}. A row whose
     * SwiGLU output would overflow FP16 in the down GEMM is scaled into range first, and the scale
     * is undone in the FP32 residual add.
     */
    public void ffn(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                xHalf,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        gemm(layer, "batch_ffn_gate_up", xHalf, gateUp[layerIndex], gateUpHalf, dim, 2 * hidden);
        layer.task(
                "batch_ffn_row_scale",
                MlxPrefillKernels::swigluRowScale,
                context,
                gateUpHalf,
                downRowScale,
                hidden,
                batchSize,
                ROW_THREADS);
        layer.task(
                "batch_ffn_swiglu",
                MlxPrefillKernels::swigluScaled,
                context,
                gateUpHalf,
                hbHalf,
                downRowScale,
                hidden,
                batchSize * hidden);
        gemm(layer, "batch_ffn_down", hbHalf, w2[layerIndex], outHalf, hidden, dim);
        layer.task(
                "batch_ffn_residual",
                MlxPrefillKernels::addRowScaledResidual,
                context,
                state.workspace.wrapXBatch,
                outHalf,
                downRowScale,
                residualScale,
                dim,
                batchSize * dim);
    }

    private void gemm(
            TaskGraph layer,
            String task,
            HalfFloatArray x,
            Matrix w,
            HalfFloatArray y,
            int k,
            int n) {
        gemm(layer, task, x, w, y, batchSize, k, n);
    }

    /**
     * {@code y[m][n] = x[m][k] . w[n][k]} as MLX library task {@code task}, over {@code w} in
     * whichever form it holds: FP16, or affine-quantized.
     */
    static void gemm(
            TaskGraph layer,
            String task,
            HalfFloatArray x,
            Matrix w,
            HalfFloatArray y,
            int m,
            int k,
            int n) {
        if (w.fp16() != null) {
            layer.libraryTask(task, Mlx::matmulTransposed, x, w.fp16(), y, m, k, n);
        } else {
            layer.libraryTask(
                    task,
                    Mlx::quantizedMatmul,
                    x,
                    w.q(),
                    w.scales(),
                    w.biases(),
                    y,
                    m,
                    k,
                    n,
                    GROUP,
                    w.bits());
        }
    }

    /** Keeps the shared buffers on the device for the next layer graph. */
    public void persist(TaskGraph layer) {
        layer.persistOnDevice(
                xHalf, qkvHalf, attnHalf, outHalf, gateUpHalf, hbHalf, attnRowScale, downRowScale);
    }

    /** The grids of this helper's kernels, under graph-qualified prefix {@code p}. */
    public void addGrids(GridScheduler scheduler, String p) {
        scheduler.addWorkerGrid(p + "batch_attn_rms_apply", elementwise(batchSize * dim));
        scheduler.addWorkerGrid(p + "batch_qkv_split", elementwise(batchSize * (qDim + 2 * kvDim)));
        scheduler.addWorkerGrid(p + "batch_attn_row_scale", rowWorker());
        scheduler.addWorkerGrid(p + "batch_attn_half", elementwise(batchSize * qDim));
        scheduler.addWorkerGrid(p + "batch_attn_residual", elementwise(batchSize * dim));
        scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", elementwise(batchSize * dim));
        scheduler.addWorkerGrid(p + "batch_ffn_row_scale", rowWorker());
        scheduler.addWorkerGrid(p + "batch_ffn_swiglu", elementwise(batchSize * hidden));
        scheduler.addWorkerGrid(p + "batch_ffn_residual", elementwise(batchSize * dim));
    }

    /** One workgroup of {@link #ROW_THREADS} per token row. */
    private WorkerGrid rowWorker() {
        return WorkerGridFactory.genericWorker(batchSize * ROW_THREADS, ROW_THREADS);
    }

    private static WorkerGrid elementwise(int total) {
        return WorkerGridFactory.genericWorker((total + 255) / 256 * 256, 256);
    }
}
