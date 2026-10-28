package org.beehive.jitllm.backend.tornado.layers;

import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MMAKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The projections of a tensor-core batched prefill as FP16 GEMMs over one buffer per GEMM: each
 * weight decoded (Q8_0) or copied (F16) into it after the one before, then one GEMM over all of
 * them. Copying is what lets the F16 Q, K and V, or gate and up, go through cuBLAS as one call.
 *
 * <p>The Q8_0 GEMM decodes every weight again for every 128-row tile of the batch, inside its inner
 * loop. Decoding once per chunk costs one pass over the weights and runs the GEMM at the FP16 rate.
 * The scratch is the size of the widest projection the caller multiplies.
 *
 * <p>A projection wide enough to fill the device is decoded in the tile order {@link
 * Qwen35MMAKernels#gemmMMATiledB} stages, whose B tiles then arrive by {@code cp.async} from
 * contiguous words. A narrower one is decoded row-major for {@link SplitKGemm}, which splits its
 * depth.
 *
 * <p>When the decoded weights of the whole model fit in {@link #RESIDENT_BYTES}, each projection
 * keeps its own FP16 copy instead, decoded on the first chunk only: a small model's decode costs a
 * sixth of its prefill, and its FP16 weights are a few gigabytes.
 */
// @formatter:on
public final class FP16Projections {

    /**
     * The FP16 bytes up to which the backend keeps every projection decoded when the policy leaves
     * it the choice. A load that knows its budget resolves the choice against it first ({@code
     * TornadoMemoryModel.resolveDecodedProjections}).
     */
    public static final long RESIDENT_BYTES = 6L << 30;

    /**
     * Whether a batched prefill under {@code policy} keeps resident decoded copies of layers that
     * hold {@code weights} weights.
     */
    public static boolean resident(ExecutionPolicy policy, long weights) {
        return switch (policy.decodedProjections()) {
            case RESIDENT -> true;
            case PER_CHUNK -> false;
            case BACKEND -> residentBytes(weights) <= RESIDENT_BYTES;
        };
    }

    /** The device bytes of resident FP16 copies of {@code weights} weights. */
    public static long residentBytes(long weights) {
        return 2 * weights;
    }

    /**
     * The device bytes a {@link FP16Projections} allocates besides any resident copies: the
     * per-chunk scratch (when not resident), the decode flags and the split-K partial products.
     */
    public static long workspaceBytes(
            boolean resident,
            long widest,
            int rows,
            int widestSplit,
            int projectionCalls,
            boolean nativeGemm) {
        return 2 * (resident ? 1 : widest)
                + (long) (projectionCalls + 1) * Integer.BYTES
                + SplitKGemm.bytes(rows, widestSplit, nativeGemm);
    }

    private final KernelContext context;
    private final WeightPrepacking prepacking;
    private final HalfFloatArray scratch;
    private final SplitKGemm gemm;
    private final Map<String, Integer> decodeTasks = new LinkedHashMap<>();
    private final java.util.List<String> markTasks = new java.util.ArrayList<>();
    private final boolean resident;
    private final IntArray decoded;
    private final int neverDecoded;
    private int nextFlag;

    /**
     * @param prepacking the plan's weight prepacking: a prepacked weight has no FP16 decode here
     * @param widest the most weights one projection call multiplies, all its tensors together
     * @param rows the padded batch, the GEMMs' row count
     * @param widestSplit the widest output a split projection writes
     */
    public FP16Projections(
            KernelContext context,
            WeightPrepacking prepacking,
            long widest,
            int rows,
            int widestSplit,
            boolean resident,
            int projectionCalls,
            boolean nativeGemm) {
        this.context = context;
        this.prepacking = prepacking;
        this.resident = resident;
        // One flag per resident projection, and a last one never set for the per-chunk decodes.
        this.decoded = new IntArray(projectionCalls + 1);
        this.neverDecoded = projectionCalls;
        this.scratch = new HalfFloatArray(resident ? 1 : Math.toIntExact(widest));
        this.gemm = new SplitKGemm(context, rows, widestSplit, nativeGemm);
    }

    /** The buffers, which every layer graph transfers or consumes with its intermediates. */
    public Object[] buffers() {
        return new Object[] {scratch, gemm.partial(), decoded};
    }

    // @formatter:off
    /**
     * {@code c[m][N] = a[m][k] . [W_0 | W_1 | ...]^T} as task {@code task}: each Q8_0 weight
     * ({@code [N_i][k]}, row-major) decoded into the scratch after the one before it, then one FP16
     * GEMM over their rows, so the output columns are the weights' rows in order. The GEMM's grid
     * is the caller's, under {@code task}; {@link #addGrids} registers the decode tasks'.
     */
    // @formatter:on
    public void project(
            TaskGraph layer,
            String task,
            HalfFloatArray a,
            FloatArray c,
            int m,
            int k,
            TornadoTensor... ws) {
        HalfFloatArray b = scratch;
        int flag = 0;
        if (resident) {
            long total = 0;
            for (TornadoTensor w : ws) {
                total += elements(w);
            }
            b = new HalfFloatArray(Math.toIntExact(total));
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, b);
            flag = nextFlag++;
        }
        long total = 0;
        boolean wholeRowBlocks = k % 32 == 0;
        for (TornadoTensor w : ws) {
            if (prepacking.prepacked(w)) {
                throw WeightPrepacking.noPrepackedKernel("Q8_0 batch-prefill task '" + task + "'");
            }
            long elements = elements(w);
            total += elements;
            wholeRowBlocks &= (elements / k) % 128 == 0;
        }
        int columns = Math.toIntExact(total / k);
        boolean tileDecodable = true;
        for (TornadoTensor w : ws) {
            tileDecodable &= w.dataType() != DataType.F16;
        }
        if (tileDecodable
                && !gemm.nativeGemm()
                && wholeRowBlocks
                && m % 128 == 0
                && gemm.slices(m, columns, k) == 1) {
            projectTiled(layer, task, a, c, m, k, b, resident ? flag : neverDecoded, ws);
            return;
        }
        int offset = 0;
        for (int i = 0; i < ws.length; i++) {
            if (prepacking.prepacked(ws[i])) {
                throw WeightPrepacking.noPrepackedKernel("Q8_0 batch-prefill task '" + task + "'");
            }
            int n = (int) elements(ws[i]);
            String decode = task + "Decode" + i;
            if (ws[i].dataType() == DataType.F16) {
                layer.task(
                        decode,
                        TransformerBatchPrefillKernels::copyFP16Once,
                        context,
                        ws[i].asHalfFloatArray(),
                        b,
                        offset,
                        n,
                        decoded,
                        resident ? flag : neverDecoded);
            } else if (decodesOnceOnly(ws[i])) {
                // Q4_0, Q4_1 and Q5_K have the decode-once form only; a per-chunk decode passes it
                // the flag that is never set.
                layer.task(
                        decode,
                        switch (ws[i].dataType()) {
                            case Q4_0 -> TransformerComputeKernelsQ4_0::dequantizeQ4_0ToFP16Once;
                            case Q4_1 -> TransformerComputeKernelsQ4_1::dequantizeQ4_1ToFP16Once;
                            default -> TransformerComputeKernelsQ5_K::dequantizeQ5_KToFP16Once;
                        },
                        context,
                        ws[i].asByteArray(),
                        b,
                        offset,
                        n,
                        decoded,
                        resident ? flag : neverDecoded);
            } else if (ws[i].dataType() == DataType.Q4_K || ws[i].dataType() == DataType.Q6_K) {
                boolean q4k = ws[i].dataType() == DataType.Q4_K;
                if (resident) {
                    layer.task(
                            decode,
                            q4k
                                    ? TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16Once
                                    : TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16Once,
                            context,
                            ws[i].asByteArray(),
                            b,
                            offset,
                            n,
                            decoded,
                            flag);
                } else {
                    layer.task(
                            decode,
                            q4k
                                    ? TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16
                                    : TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                            context,
                            ws[i].asByteArray(),
                            b,
                            offset,
                            n);
                }
            } else if (resident) {
                layer.task(
                        decode,
                        TransformerBatchPrefillKernels::dequantizeQ8_0ToFP16Once,
                        context,
                        ws[i].asByteArray(),
                        b,
                        offset,
                        n,
                        decoded,
                        flag);
            } else {
                layer.task(
                        decode,
                        TransformerBatchPrefillKernels::dequantizeQ8_0ToFP16,
                        context,
                        ws[i].asByteArray(),
                        b,
                        offset,
                        n);
            }
            boolean striding =
                    resident || ws[i].dataType() == DataType.F16 || decodesOnceOnly(ws[i]);
            decodeTasks.put(
                    layer.getTaskGraphName() + "." + decode,
                    striding ? Math.min(n, TransformerBatchPrefillKernels.DECODE_ONCE_LANES) : n);
            offset += n;
        }
        if (resident) {
            String mark = task + "Decoded";
            layer.task(mark, TransformerBatchPrefillKernels::markDecoded, context, decoded, flag);
            markTasks.add(layer.getTaskGraphName() + "." + mark);
        }
        gemm.gemm(layer, task, a, b, c, m, offset / k, k);
    }

    /** Whether {@code w}'s type has only the decode-once FP16 decoder. */
    private static boolean decodesOnceOnly(TornadoTensor w) {
        DataType type = w.dataType();
        return type == DataType.Q4_0 || type == DataType.Q4_1 || type == DataType.Q5_K;
    }

    /**
     * Weights in {@code w}: FP16; Q8_0, Q4_0 and Q4_1 at 34, 18 and 20 bytes to 32 weights; Q4_K,
     * Q5_K and Q6_K at 144, 176 and 210 bytes to 256.
     */
    private static long elements(TornadoTensor w) {
        return switch (w.dataType()) {
            case F16 -> w.asHalfFloatArray().getSize();
            case Q4_0 -> (long) w.asByteArray().getSize() / 18 * 32;
            case Q4_1 -> (long) w.asByteArray().getSize() / 20 * 32;
            case Q4_K -> (long) w.asByteArray().getSize() / 144 * 256;
            case Q5_K -> (long) w.asByteArray().getSize() / 176 * 256;
            case Q6_K -> (long) w.asByteArray().getSize() / 210 * 256;
            default -> (long) w.asByteArray().getSize() / 34 * 32;
        };
    }

    /**
     * The tile-order decodes of {@code ws} (any quantized type) into {@code b}, then {@code
     * gemmMMATiledB}.
     */
    private void projectTiled(
            TaskGraph layer,
            String task,
            HalfFloatArray a,
            FloatArray c,
            int m,
            int k,
            HalfFloatArray b,
            int flag,
            TornadoTensor... ws) {
        int offset = 0;
        for (int i = 0; i < ws.length; i++) {
            int n = (int) elements(ws[i]);
            String decode = task + "Decode" + i;
            layer.task(
                    decode,
                    switch (ws[i].dataType()) {
                        case Q4_0 -> Qwen35MMAKernels::dequantizeQ4_0ToFP16Tiled;
                        case Q4_1 -> Qwen35MMAKernels::dequantizeQ4_1ToFP16Tiled;
                        case Q4_K -> Qwen35MMAKernels::dequantizeQ4_KToFP16Tiled;
                        case Q5_K -> Qwen35MMAKernels::dequantizeQ5_KToFP16Tiled;
                        case Q6_K -> Qwen35MMAKernels::dequantizeQ6_KToFP16Tiled;
                        default -> Qwen35MMAKernels::dequantizeQ8_0ToFP16Tiled;
                    },
                    context,
                    ws[i].asByteArray(),
                    b,
                    offset,
                    n,
                    k,
                    decoded,
                    flag);
            decodeTasks.put(
                    layer.getTaskGraphName() + "." + decode,
                    Math.min(n, Qwen35MMAKernels.TILED_DECODE_LANES));
            offset += n;
        }
        if (resident) {
            String mark = task + "Decoded";
            layer.task(mark, TransformerBatchPrefillKernels::markDecoded, context, decoded, flag);
            markTasks.add(layer.getTaskGraphName() + "." + mark);
        }
        layer.task(task, Qwen35MMAKernels::gemmMMATiledB, context, a, b, c, m, offset / k, k);
    }

    /**
     * Registers the grids of the decode tasks and the split projections {@link #project} added,
     * replacing the caller's grid for a split projection. Call after the caller's grids.
     */
    public void addGrids(GridScheduler scheduler) {
        decodeTasks.forEach(
                (name, n) ->
                        scheduler.addWorkerGrid(
                                name, WorkerGridFactory.genericWorker((n + 255) / 256 * 256, 256)));
        markTasks.forEach(
                name -> scheduler.addWorkerGrid(name, WorkerGridFactory.genericWorker(1, 1)));
        gemm.addGrids(scheduler);
    }

    /** Whether the projections run as cuBLAS GEMMs rather than jitllm's kernels. */
    public boolean nativeGemm() {
        return gemm.nativeGemm();
    }
}
