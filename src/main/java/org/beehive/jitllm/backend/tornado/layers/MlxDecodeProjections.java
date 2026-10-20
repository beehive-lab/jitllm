package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.MlxPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The single-token projections of a Llama-style decode layer as MLX GEMMs over the weights the MLX
 * batched prefill already holds on the device, so a model whose prefill runs on MLX keeps one copy
 * of its projections rather than two.
 *
 * <p><b>Why.</b> A large quantized model that prefills on MLX would otherwise hold its projections
 * twice: MLX's affine form for the prefill and the file's own blocks for the decode kernels. For a
 * 24B Q4_K model that is 15 GB beside 14 GB, which does not fit a 48 GB machine. Here the decode
 * reads MLX's form too, bound from the prefill graph of the same layer, and the file's projection
 * tensors are never uploaded.
 *
 * <p><b>Shape.</b> The same four GEMMs as {@link MlxPrefillProjections} at one row: the stacked QKV
 * split into {@code wrapQ}, {@code wrapK} and {@code wrapV}; the output projection of {@code
 * wrapXb} added into {@code wrapX}; the stacked gate/up, SwiGLU, and the down projection added into
 * {@code wrapX}. The inputs of the two projections that add into the residual stream are scaled
 * into FP16's range first and the scale undone in FP32, as in the prefill. The family keeps its
 * norms, rotation and attention.
 *
 * <p>The weights must be the same objects the prefill graphs uploaded — {@link MlxWeightCache}
 * hands out one per matrix — because a graph binds another's buffer by identity.
 */
// @formatter:on
public final class MlxDecodeProjections {

    private static final int ROW_THREADS = 256;

    private final State state;
    private final KernelContext context;
    private final int dim;
    private final int qDim;
    private final int kvDim;
    private final int hidden;
    private final MlxPrefillProjections.Matrix[] qkv;
    private final MlxPrefillProjections.Matrix[] wo;
    private final MlxPrefillProjections.Matrix[] gateUp;
    private final MlxPrefillProjections.Matrix[] w2;

    private final HalfFloatArray xHalf; // dim
    private final HalfFloatArray qkvHalf; // qDim + 2·kvDim
    private final HalfFloatArray attnHalf; // qDim
    private final HalfFloatArray outHalf; // dim
    private final HalfFloatArray gateUpHalf; // 2·hidden
    private final HalfFloatArray hbHalf; // hidden
    private final FloatArray attnScale; // 1
    private final FloatArray downScale; // 1

    /**
     * The decode projections of a model whose MLX prefill keeps its affine repacks in {@code
     * cache}; null unless {@code mlx} is affine, the only form the cache holds.
     */
    public static MlxDecodeProjections ifAffine(
            MlxPrefill mlx,
            MlxWeightCache cache,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden) {
        if (mlx.projections() != MlxPrefill.Projections.AFFINE) {
            return null;
        }
        return new MlxDecodeProjections(
                cache, state, weights, context, layers, dim, qDim, kvDim, hidden);
    }

    private MlxDecodeProjections(
            MlxWeightCache cache,
            State state,
            TornadoWeights weights,
            KernelContext context,
            int layers,
            int dim,
            int qDim,
            int kvDim,
            int hidden) {
        this.state = state;
        this.context = context;
        this.dim = dim;
        this.qDim = qDim;
        this.kvDim = kvDim;
        this.hidden = hidden;
        this.qkv = new MlxPrefillProjections.Matrix[layers];
        this.wo = new MlxPrefillProjections.Matrix[layers];
        this.gateUp = new MlxPrefillProjections.Matrix[layers];
        this.w2 = new MlxPrefillProjections.Matrix[layers];
        for (int i = 0; i < layers; i++) {
            MlxPrefillProjections.LayerWeights l =
                    MlxPrefillProjections.LayerWeights.of(weights, i);
            qkv[i] = cache.affine(l.qkv());
            wo[i] = cache.affine(l.wo());
            gateUp[i] = cache.affine(l.gateUp());
            w2[i] = cache.affine(l.w2());
        }
        this.xHalf = new HalfFloatArray(dim);
        this.qkvHalf = new HalfFloatArray(qDim + 2 * kvDim);
        this.attnHalf = new HalfFloatArray(qDim);
        this.outHalf = new HalfFloatArray(dim);
        this.gateUpHalf = new HalfFloatArray(2 * hidden);
        this.hbHalf = new HalfFloatArray(hidden);
        this.attnScale = new FloatArray(1);
        this.downScale = new FloatArray(1);
    }

    /**
     * Declares the buffers — uploaded by the first decode graph, bound from {@code predecessor}
     * after — and binds layer {@code layerIndex}'s weights from the prefill graph {@code
     * weightSource}, which uploaded them.
     */
    public void declare(
            TaskGraph layer,
            int layerIndex,
            boolean first,
            String predecessor,
            String weightSource) {
        Object[] buffers = {
            xHalf, qkvHalf, attnHalf, outHalf, gateUpHalf, hbHalf, attnScale, downScale
        };
        if (first) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffers);
        } else {
            layer.consumeFromDevice(predecessor, buffers);
        }
        List<Object> arrays = new ArrayList<>();
        for (MlxPrefillProjections.Matrix m :
                new MlxPrefillProjections.Matrix[] {
                    qkv[layerIndex], wo[layerIndex], gateUp[layerIndex], w2[layerIndex]
                }) {
            arrays.addAll(List.of(m.arrays()));
        }
        layer.consumeFromDevice(weightSource, arrays.toArray());
    }

    /** Keeps the buffers on the device for the next decode graph. */
    public void persist(TaskGraph layer) {
        layer.persistOnDevice(
                xHalf, qkvHalf, attnHalf, outHalf, gateUpHalf, hbHalf, attnScale, downScale);
    }

    /**
     * Q, K and V of the normed {@code wrapXb}, into {@code wrapQ}, {@code wrapK} and {@code wrapV}.
     */
    public void qkv(TaskGraph layer, int layerIndex) {
        layer.task(
                "mlx_qkv_half",
                MlxPrefillKernels::toHalf,
                context,
                state.workspace.wrapXb,
                xHalf,
                dim);
        MlxPrefillProjections.gemm(
                layer, "mlx_qkv", xHalf, qkv[layerIndex], qkvHalf, 1, dim, qDim + 2 * kvDim);
        layer.task(
                "mlx_qkv_split",
                MlxPrefillKernels::splitQkv,
                context,
                qkvHalf,
                state.workspace.wrapQ,
                state.workspace.wrapK,
                state.workspace.wrapV,
                qDim,
                kvDim,
                qDim + 2 * kvDim);
    }

    /** The output projection of the attention in {@code wrapXb}, added into {@code wrapX}. */
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        layer.task(
                "mlx_attn_row_scale",
                MlxPrefillKernels::rowScale,
                context,
                state.workspace.wrapXb,
                attnScale,
                qDim,
                1,
                ROW_THREADS);
        layer.task(
                "mlx_attn_half",
                MlxPrefillKernels::toHalfScaled,
                context,
                state.workspace.wrapXb,
                attnHalf,
                attnScale,
                qDim,
                qDim);
        MlxPrefillProjections.gemm(
                layer, "mlx_attn_out", attnHalf, wo[layerIndex], outHalf, 1, qDim, dim);
        layer.task(
                "mlx_attn_residual",
                MlxPrefillKernels::addRowScaledResidual,
                context,
                state.workspace.wrapX,
                outHalf,
                attnScale,
                1.0f,
                dim,
                dim);
    }

    /**
     * The feed-forward of the normed {@code wrapXb}: gate/up, SwiGLU and down, added into {@code
     * wrapX}.
     */
    public void ffn(TaskGraph layer, int layerIndex) {
        layer.task(
                "mlx_ffn_half",
                MlxPrefillKernels::toHalf,
                context,
                state.workspace.wrapXb,
                xHalf,
                dim);
        MlxPrefillProjections.gemm(
                layer,
                "mlx_ffn_gate_up",
                xHalf,
                gateUp[layerIndex],
                gateUpHalf,
                1,
                dim,
                2 * hidden);
        layer.task(
                "mlx_ffn_row_scale",
                MlxPrefillKernels::swigluRowScale,
                context,
                gateUpHalf,
                downScale,
                hidden,
                1,
                ROW_THREADS);
        layer.task(
                "mlx_ffn_swiglu",
                MlxPrefillKernels::swigluScaled,
                context,
                gateUpHalf,
                hbHalf,
                downScale,
                hidden,
                hidden);
        MlxPrefillProjections.gemm(
                layer, "mlx_ffn_down", hbHalf, w2[layerIndex], outHalf, 1, hidden, dim);
        layer.task(
                "mlx_ffn_residual",
                MlxPrefillKernels::addRowScaledResidual,
                context,
                state.workspace.wrapX,
                outHalf,
                downScale,
                1.0f,
                dim,
                dim);
    }

    /** The grids of the glue kernels, under graph-qualified prefix {@code p}. */
    public void addGrids(GridScheduler scheduler, String p) {
        WorkerGrid row = WorkerGridFactory.genericWorker(ROW_THREADS, ROW_THREADS);
        scheduler.addWorkerGrid(p + "mlx_qkv_half", elementwise(dim));
        scheduler.addWorkerGrid(p + "mlx_qkv_split", elementwise(qDim + 2 * kvDim));
        scheduler.addWorkerGrid(p + "mlx_attn_row_scale", row);
        scheduler.addWorkerGrid(p + "mlx_attn_half", elementwise(qDim));
        scheduler.addWorkerGrid(p + "mlx_attn_residual", elementwise(dim));
        scheduler.addWorkerGrid(p + "mlx_ffn_half", elementwise(dim));
        scheduler.addWorkerGrid(p + "mlx_ffn_row_scale", row);
        scheduler.addWorkerGrid(p + "mlx_ffn_swiglu", elementwise(hidden));
        scheduler.addWorkerGrid(p + "mlx_ffn_residual", elementwise(dim));
    }

    private static WorkerGrid elementwise(int total) {
        return WorkerGridFactory.genericWorker((total + 255) / 256 * 256, 256);
    }
}
