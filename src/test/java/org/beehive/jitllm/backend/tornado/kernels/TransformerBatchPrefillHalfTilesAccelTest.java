package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The Q8_0 {@code ...HalfTiles} GEMMs of {@link TransformerBatchPrefillSimdgroupKernels} against
 * the arithmetic they implement, beside the FP32-tile GEMMs on the same data.
 *
 * <p>The weights are Q8_0 values whose products need more than FP16's 11 significant bits. Example:
 * scale 1427/1024 times value 29 is 40.4140625, which FP16 rounds to 40.40625. The activations are
 * -1, 0 or 1, so every sum is exact in {@code float} in any order. Each kernel must then equal its
 * own reference bit for bit: an FP16-tile kernel the sum over the weights rounded to FP16 (round to
 * nearest even, subnormals kept), an FP32-tile kernel the sum over the exact weights. Rounding the
 * staged weights is then the only difference between the two, and an accumulation, masking or index
 * error is a mismatch. Whole tiles and tiles that end inside the batch or the rows are both
 * covered.
 */
// @formatter:on
public class TransformerBatchPrefillHalfTilesAccelTest {

    /** The contraction length: four 32-wide steps of the k-loop. */
    private static final int N = 128;

    private static void assumeMetal() {
        assumeTrue("SIMD-group matrices are Metal only", BatchPrefillGemmPolicy.tiled());
    }

    /**
     * Q8_0 test weights. Normal scales lie in [1, 2) with 11 significant bits. Subnormal scales are
     * {@code m * 2^-24} for m below 1024, so a weight below 2^-14 is an FP16 subnormal, which an
     * FP16 conversion that flushes to zero would lose.
     */
    private record Weights(boolean subnormal) {

        float scale(int row, int block) {
            int m = (row * 37 + block * 101 + 403) % 1023;
            return subnormal ? (1 + m) * 0x1p-24f : (1024 + m) / 1024.0f;
        }

        static int value(int row, int k) {
            return ((row * 13 + k * 7) % 63) - 31;
        }

        /** The exact Q8_0 weight, which {@code float} holds exactly. */
        float exact(int row, int k) {
            return value(row, k) * scale(row, k / 32);
        }

        /** The weight rounded to FP16, as the {@code ...HalfTiles} kernels stage it. */
        float half(int row, int k) {
            return Float.float16ToFloat(Float.floatToFloat16(exact(row, k)));
        }

        Weight reference(boolean halfTiles) {
            return halfTiles ? this::half : this::exact;
        }

        ByteArray bytes(int rows, int firstRow) {
            int blocks = N / 32;
            ByteArray w = new ByteArray(rows * blocks * 34);
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < blocks; b++) {
                    int base = (r * blocks + b) * 34;
                    short bits = Float.floatToFloat16(scale(firstRow + r, b));
                    w.set(base, (byte) bits);
                    w.set(base + 1, (byte) (bits >> 8));
                    for (int i = 0; i < 32; i++) {
                        w.set(base + 2 + i, (byte) value(firstRow + r, b * 32 + i));
                    }
                }
            }
            return w;
        }
    }

    private static final Weights NORMAL = new Weights(false);

    private interface Weight {
        float at(int row, int k);
    }

    /**
     * -1, 0 or 1, from a hash of the token and k, so that it varies along k as well as across
     * tokens and a sum over the weights in the wrong k order is a different sum.
     */
    private static int activation(int token, int k) {
        int h = (token * 73 + k * 151 + 17) * 0x9E3779B1;
        return (h >>> 8) % 3 - 1;
    }

    /**
     * {@code w} read in a wrong k order: one column further on, or reversed within each 8-wide
     * fragment.
     */
    private static List<Weight> wrongKOrders(Weight w) {
        return List.of((row, k) -> w.at(row, (k + 1) % N), (row, k) -> w.at(row, k ^ 7));
    }

    private static FloatArray activations(int batch) {
        FloatArray x = new FloatArray(batch * N);
        for (int t = 0; t < batch; t++) {
            for (int k = 0; k < N; k++) {
                x.set(t * N + k, activation(t, k));
            }
        }
        return x;
    }

    /**
     * Exact in {@code float}: at most 128 terms of at most 62, in steps of 2^-10 for normal scales
     * and of 2^-24 for subnormal ones.
     */
    private static float dot(int token, int row, Weight w) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += activation(token, k) * w.at(row, k);
        }
        return sum;
    }

    private static void run(TaskGraph graph, String name, int groups, boolean halfTiles) {
        int threads = TransformerBatchPrefillSimdgroupKernels.threads(halfTiles);
        WorkerGrid1D worker = new WorkerGrid1D(groups * threads);
        worker.setLocalWork(threads, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".t", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /** The outputs, {@code rows} by {@code batch} tokens, where two weights' sums differ. */
    private static int differing(int batch, int rows, Weight w, Weight other) {
        int differing = 0;
        for (int t = 0; t < batch; t++) {
            for (int r = 0; r < rows; r++) {
                if (dot(t, r, w) != dot(t, r, other)) {
                    differing++;
                }
            }
        }
        return differing;
    }

    /**
     * The data must tell the reference from FP16 against FP32 and from a wrong k order, or a bit
     * for bit match proves nothing. More than half the outputs must differ.
     */
    private static void assertDataDiscriminates(
            String name, Weights weights, Weight w, int batch, int rows) {
        int total = batch * rows;
        int rounding = differing(batch, rows, weights::half, weights::exact);
        assertTrue(
                name + ": only " + rounding + " of " + total + " outputs tell FP16 from FP32",
                rounding * 2 > total);
        for (Weight wrong : wrongKOrders(w)) {
            int misordered = differing(batch, rows, w, wrong);
            assertTrue(
                    name + ": only " + misordered + " of " + total + " tell a wrong k order",
                    misordered * 2 > total);
        }
    }

    private static void checkResidual(boolean halfTiles, int d, int batch) {
        assumeMetal();
        FloatArray x = activations(batch);
        FloatArray out = new FloatArray(batch * d);
        Weight w = NORMAL.reference(halfTiles);
        float[] expected = new float[batch * d];
        for (int t = 0; t < batch; t++) {
            for (int r = 0; r < d; r++) {
                float start = (t * 3 + r) % 11 - 5;
                out.set(t * d + r, start);
                expected[t * d + r] = start + dot(t, r, w);
            }
        }
        String name = "halfTilesResidual" + (halfTiles ? "FP16" : "FP32") + d + "x" + batch;
        assertDataDiscriminates(name, NORMAL, w, batch, d);
        ByteArray wq = NORMAL.bytes(d, 0);
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, wq);
        graph.task(
                "t",
                halfTiles
                        ? TransformerBatchPrefillSimdgroupKernels
                                ::batchedGemmQ8WithResidualHalfTiles
                        : TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                new KernelContext(),
                x,
                out,
                wq,
                N,
                d,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, name, ((d + 63) / 64) * ((batch + 63) / 64), halfTiles);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(name + " element " + i, expected[i], out.get(i), 0.0f);
        }
    }

    private static void checkQkv(
            Weights weights, boolean halfTiles, int qDim, int kvDim, int batch) {
        assumeMetal();
        FloatArray x = activations(batch);
        FloatArray q = new FloatArray(batch * qDim);
        FloatArray k = new FloatArray(batch * kvDim);
        FloatArray v = new FloatArray(batch * kvDim);
        // Each projection its own rows of one generator, so a swapped projection is a mismatch.
        ByteArray wq = weights.bytes(qDim, 0);
        ByteArray wk = weights.bytes(kvDim, qDim);
        ByteArray wv = weights.bytes(kvDim, qDim + kvDim);
        String name =
                "halfTilesQkv"
                        + (halfTiles ? "FP16" : "FP32")
                        + (weights.subnormal() ? "Subnormal" : "")
                        + batch;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, q, k, v)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, wq, wk, wv);
        graph.task(
                "t",
                halfTiles
                        ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8HalfTiles
                        : TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
                new KernelContext(),
                x,
                q,
                k,
                v,
                wq,
                wk,
                wv,
                N,
                qDim,
                kvDim,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, q, k, v);
        run(graph, name, ((qDim + 2 * kvDim) / 64) * ((batch + 63) / 64), halfTiles);
        Weight w = weights.reference(halfTiles);
        int rows = qDim + 2 * kvDim;
        for (int t = 0; t < batch; t++) {
            for (int r = 0; r < rows; r++) {
                float got =
                        r < qDim
                                ? q.get(t * qDim + r)
                                : r < qDim + kvDim
                                        ? k.get(t * kvDim + r - qDim)
                                        : v.get(t * kvDim + r - qDim - kvDim);
                assertEquals(name + " token " + t + " row " + r, dot(t, r, w), got, 0.0f);
            }
        }
        assertDataDiscriminates(name, weights, w, batch, rows);
    }

    private static void checkGateUp(boolean halfTiles, int hiddenDim, int batch) {
        assumeMetal();
        FloatArray x = activations(batch);
        FloatArray rms = new FloatArray(N);
        rms.init(1.0f);
        FloatArray norm = new FloatArray(batch);
        norm.init(1.0f);
        FloatArray hb = new FloatArray(batch * hiddenDim);
        ByteArray w1 = NORMAL.bytes(hiddenDim, 0);
        ByteArray w3 = NORMAL.bytes(hiddenDim, hiddenDim);
        String name = "halfTilesGateUp" + (halfTiles ? "FP16" : "FP32") + hiddenDim + "x" + batch;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, rms, norm, hb)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w1, w3);
        graph.task(
                "t",
                halfTiles
                        ? TransformerBatchPrefillSimdgroupKernels
                                ::batchedGemmRmsNormFFNGateUpQ8HalfTiles
                        : TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ8,
                new KernelContext(),
                x,
                hb,
                rms,
                norm,
                w1,
                w3,
                N,
                hiddenDim,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, name, ((hiddenDim + 31) / 32) * ((batch + 63) / 64), halfTiles);
        Weight w = NORMAL.reference(halfTiles);
        for (int t = 0; t < batch; t++) {
            for (int r = 0; r < hiddenDim; r++) {
                float expected = gateUp(t, r, hiddenDim, w);
                assertEquals(
                        name + " token " + t + " row " + r,
                        expected,
                        hb.get(t * hiddenDim + r),
                        tolerance(expected));
            }
        }
        // A negative gate saturates the SiLU towards zero on both sides, so only about half the
        // outputs can show a difference; require a quarter.
        List<Weight> others = new ArrayList<>(wrongKOrders(w));
        others.add(NORMAL.reference(!halfTiles));
        for (Weight other : others) {
            int differing = 0;
            for (int t = 0; t < batch; t++) {
                for (int r = 0; r < hiddenDim; r++) {
                    float expected = gateUp(t, r, hiddenDim, w);
                    if (Math.abs(gateUp(t, r, hiddenDim, other) - expected) > tolerance(expected)) {
                        differing++;
                    }
                }
            }
            assertTrue(
                    name + ": only " + differing + " outputs tell another reference",
                    differing * 4 > batch * hiddenDim);
        }
    }

    private static float gateUp(int token, int row, int hiddenDim, Weight w) {
        return swiglu(dot(token, row, w), dot(token, row + hiddenDim, w));
    }

    /**
     * The gate and up sums are exact; only the SiLU's exp differs between the host and the kernel,
     * by a few units in the last place.
     */
    private static float tolerance(float expected) {
        return 1e-5f * Math.max(1.0f, Math.abs(expected));
    }

    private static float swiglu(float gate, float up) {
        return (gate / (1.0f + (float) Math.exp(-gate))) * up;
    }

    @Test
    public void halfTilesResidualRoundsWeightsToFp16() {
        checkResidual(true, 128, 128);
        checkResidual(true, 72, 70);
    }

    @Test
    public void fp32TilesResidualKeepsExactWeights() {
        checkResidual(false, 128, 128);
        checkResidual(false, 72, 70);
    }

    @Test
    public void halfTilesQkvRoundsWeightsToFp16() {
        checkQkv(NORMAL, true, 128, 64, 128);
        checkQkv(NORMAL, true, 128, 64, 70);
    }

    @Test
    public void fp32TilesQkvKeepsExactWeights() {
        checkQkv(NORMAL, false, 128, 64, 128);
        checkQkv(NORMAL, false, 128, 64, 70);
    }

    /** Weights below 2^-14 stay FP16 subnormals rather than flushing to zero. */
    @Test
    public void halfTilesQkvKeepsFp16Subnormals() {
        checkQkv(new Weights(true), true, 128, 64, 70);
        checkQkv(new Weights(true), false, 128, 64, 70);
    }

    @Test
    public void halfTilesGateUpRoundsWeightsToFp16() {
        checkGateUp(true, 64, 128);
        checkGateUp(true, 40, 70);
    }

    @Test
    public void fp32TilesGateUpKeepsExactWeights() {
        checkGateUp(false, 64, 128);
        checkGateUp(false, 40, 70);
    }
}
