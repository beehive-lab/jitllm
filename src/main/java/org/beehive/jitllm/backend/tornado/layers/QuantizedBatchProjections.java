package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The JIT batched-prefill projections of a layer whose weights stay in the file's own
 * representations: each weight's kernel and grid chosen from its type, on SIMD-group matrices
 * (Metal) or as tiled batch projections elsewhere. The batched counterpart of {@link
 * QuantizedProjections}; the tensor-core prefill decodes to FP16 instead ({@link FP16Projections}).
 *
 * <p>A Q8_0 weight in such a layer runs the family's own Q8_0 kernels: SIMD-group, the scalar tiled
 * GEMMs where the family has them ({@code tiledQ8}), or per-row matvecs. Only the residual form
 * exists for Q8_0 outside the fused QKV and gate/up tasks.
 *
 * <p>Gate and up run fused where both share a type with a fused kernel on the path that runs (every
 * type on SIMD-group matrices; Q8_0, Q4_0 and Q4_K tiled), and otherwise as the norm, gate into
 * {@code wrapHbBatch}, up into {@code ffnUpResult}, and the SwiGLU, so a layer that splits them
 * needs {@code ffnUpResult} bound.
 */
// @formatter:on
public final class QuantizedBatchProjections {

    /** Lanes of a tiled batch projection's workgroup. */
    private static final int TILED_LOCAL = 128;

    private final KernelContext context;
    private final TornadoWorkspace ws;
    private final int batchSize;
    private final boolean simdgroup;
    private final boolean tiledQ8;
    private final int matVecLocal;

    /**
     * @param simdgroup whether the projections run on SIMD-group matrices
     * @param tiledQ8 whether a Q8_0 weight runs the scalar tiled GEMMs where not on SIMD-group
     *     matrices, rather than per-row matvecs
     * @param matVecLocal the lanes per row of the Q8_0 per-row matvecs
     */
    public QuantizedBatchProjections(
            KernelContext context,
            TornadoWorkspace ws,
            int batchSize,
            boolean simdgroup,
            boolean tiledQ8,
            int matVecLocal) {
        this.context = context;
        this.ws = ws;
        this.batchSize = batchSize;
        this.simdgroup = simdgroup;
        this.tiledQ8 = tiledQ8;
        this.matVecLocal = matVecLocal;
    }

    /** {@code out = w·x} over the batch, or {@code out += w·x} with {@code residual}. */
    public void projection(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean residual) {
        DataType type = w.dataType();
        if (type == DataType.Q8_0) {
            if (!residual) {
                throw new IllegalStateException(
                        "no batched Q8_0 projection that overwrites its output, for task " + name);
            }
            layer.task(
                    name,
                    simdgroup
                            ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual
                            : tiledQ8
                                    ? TransformerBatchPrefillKernels::batchedGemmQ8WithResidual
                                    : TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    simdgroup || tiledQ8 ? batchSize : matVecLocal);
        } else if (simdgroup) {
            layer.task(
                    name,
                    residual
                            ? switch (type) {
                                case Q4_0 ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmQ4_0WithResidual;
                                case Q4_1 ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmQ4_1WithResidual;
                                case Q4_K ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmQ4_KWithResidual;
                                case Q5_K ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmQ5_KWithResidual;
                                default ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmQ6_KWithResidual;
                            }
                            : switch (type) {
                                case Q4_0 ->
                                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0;
                                case Q4_1 ->
                                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_1;
                                case Q4_K ->
                                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_K;
                                case Q5_K ->
                                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQ5_K;
                                default -> TransformerBatchPrefillSimdgroupKernels::batchedGemmQ6_K;
                            },
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    batchSize);
        } else {
            layer.task(
                    name,
                    residual
                            ? switch (type) {
                                case Q4_0 ->
                                        TransformerComputeKernelsQ4_0
                                                ::matrixVectorTiledBatchWithResidualQ4_0;
                                case Q4_1 ->
                                        TransformerComputeKernelsQ4_1
                                                ::matrixVectorTiledBatchWithResidualQ4_1;
                                case Q4_K ->
                                        TransformerComputeKernelsQ4_K
                                                ::matrixVectorTiledBatchWithResidualQ4_K;
                                case Q5_K ->
                                        TransformerComputeKernelsQ5_K
                                                ::matrixVectorTiledBatchWithResidualQ5_K;
                                default ->
                                        TransformerComputeKernelsQ6_K
                                                ::matrixVectorTiledBatchWithResidualQ6_K;
                            }
                            : switch (type) {
                                case Q4_0 ->
                                        TransformerComputeKernelsQ4_0::matrixVectorTiledBatchQ4_0;
                                case Q4_1 ->
                                        TransformerComputeKernelsQ4_1::matrixVectorTiledBatchQ4_1;
                                case Q4_K ->
                                        TransformerComputeKernelsQ4_K::matrixVectorTiledBatchQ4_K;
                                case Q5_K ->
                                        TransformerComputeKernelsQ5_K::matrixVectorTiledBatchQ5_K;
                                default ->
                                        TransformerComputeKernelsQ6_K::matrixVectorTiledBatchQ6_K;
                            },
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    batchSize,
                    TILED_LOCAL);
        }
    }

    /**
     * Whether a layer's gate and up run as one fused task: both the same type, with a fused kernel
     * on the path that runs (every type on SIMD-group matrices; Q8_0, Q4_0 and Q4_K tiled).
     */
    public boolean fusedGateUp(TornadoTensor w1, TornadoTensor w3) {
        DataType type = w1.dataType();
        if (type != w3.dataType()) {
            return false;
        }
        return simdgroup || type == DataType.Q8_0 || type == DataType.Q4_0 || type == DataType.Q4_K;
    }

    /**
     * {@code hb = SiLU(w1·x) * (w3·x)} with {@code x} the FFN-normalized activations: one fused
     * task where {@link #fusedGateUp} holds, the norm applied as the SIMD-group kernel stages the
     * activations or by its own task before the tiled one; otherwise the norm, gate into {@code
     * wrapHbBatch}, up into {@code ffnUpResult}, and the SwiGLU elementwise.
     */
    public void gateUp(
            TaskGraph layer,
            TornadoTensor w1,
            TornadoTensor w3,
            FloatArray rmsWeights,
            int dim,
            int hidDim) {
        DataType type = w1.dataType();
        boolean fused = fusedGateUp(w1, w3);
        if (fused && (simdgroup || type == DataType.Q8_0)) {
            layer.task(
                    "batch_ffn_gate_up",
                    type == DataType.Q8_0
                            ? (simdgroup
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmRmsNormFFNGateUpQ8
                                    : tiledQ8
                                            ? TransformerBatchPrefillKernels
                                                    ::batchedGemmRmsNormFFNGateUpQ8
                                            : TransformerBatchPrefillKernels
                                                    ::batchedFusedRmsNormFFNGateUpQ8)
                            : switch (type) {
                                case Q4_0 ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmRmsNormFFNGateUpQ4_0;
                                case Q4_1 ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmRmsNormFFNGateUpQ4_1;
                                case Q4_K ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmRmsNormFFNGateUpQ4_K;
                                case Q5_K ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmRmsNormFFNGateUpQ5_K;
                                default ->
                                        TransformerBatchPrefillSimdgroupKernels
                                                ::batchedGemmRmsNormFFNGateUpQ6_K;
                            },
                    context,
                    ws.wrapXBatch,
                    ws.wrapHbBatch,
                    rmsWeights,
                    ws.ffnScaleBatch,
                    w1.asByteArray(),
                    w3.asByteArray(),
                    dim,
                    hidDim,
                    type == DataType.Q8_0 && !simdgroup && !tiledQ8 ? matVecLocal : batchSize);
            return;
        }
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                ws.wrapXbBatch,
                ws.wrapXBatch,
                rmsWeights,
                ws.ffnScaleBatch,
                dim);
        if (fused) {
            layer.task(
                    "batch_ffn_gate_up",
                    type == DataType.Q4_0
                            ? TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUTiledBatchQ4_0
                            : TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUTiledBatchQ4_K,
                    context,
                    ws.wrapXbBatch,
                    ws.wrapHbBatch,
                    w1.asByteArray(),
                    w3.asByteArray(),
                    dim,
                    hidDim,
                    batchSize,
                    TILED_LOCAL);
            return;
        }
        projection(layer, "batch_ffn_gate", ws.wrapXbBatch, ws.wrapHbBatch, w1, dim, hidDim, false);
        projection(layer, "batch_ffn_up", ws.wrapXbBatch, ws.ffnUpResult, w3, dim, hidDim, false);
        layer.task(
                "batch_ffn_gate_up",
                TransformerComputeKernelsQ6_K::siluAndMultiply,
                context,
                ws.wrapHbBatch,
                ws.ffnUpResult,
                batchSize * hidDim);
    }

    /** The grid of {@code w}'s projection to {@code rows} outputs, SIMD-group or tiled. */
    public WorkerGrid worker(TornadoTensor w, int rows) {
        int batchTiles = (batchSize + 63) / 64;
        DataType type = w.dataType();
        if (type == DataType.Q8_0) {
            int gemmLocal = simdgroup ? TransformerBatchPrefillSimdgroupKernels.THREADS : 256;
            return simdgroup || tiledQ8
                    ? WorkerGridFactory.genericWorker(
                            ((rows + 63) / 64) * batchTiles * gemmLocal, gemmLocal)
                    : WorkerGridFactory.genericWorker(batchSize * rows * matVecLocal, matVecLocal);
        }
        if (simdgroup) {
            int threads = TransformerBatchPrefillSimdgroupKernels.THREADS;
            return WorkerGridFactory.genericWorker(
                    ((rows + 63) / 64) * batchTiles * threads, threads);
        }
        int rowTile =
                switch (type) {
                    case Q4_0 -> TransformerComputeKernelsQ4_0.rowTile();
                    case Q4_1 -> TransformerComputeKernelsQ4_1.rowTile();
                    case Q4_K -> TransformerComputeKernelsQ4_K.rowTile();
                    case Q5_K -> TransformerComputeKernelsQ5_K.rowTile();
                    default -> TransformerComputeKernelsQ6_K.rowTile();
                };
        int colTile =
                switch (type) {
                    case Q4_0 -> TransformerComputeKernelsQ4_0.colTile();
                    case Q4_1 -> TransformerComputeKernelsQ4_1.colTile();
                    case Q4_K -> TransformerComputeKernelsQ4_K.colTile();
                    case Q5_K -> TransformerComputeKernelsQ5_K.colTile();
                    default -> TransformerComputeKernelsQ6_K.colTile();
                };
        int rowGroups = (batchSize + rowTile - 1) / rowTile;
        int colGroups = (rows + colTile - 1) / colTile;
        return WorkerGridFactory.genericWorker(rowGroups * colGroups * TILED_LOCAL, TILED_LOCAL);
    }
}
