package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * {@link TransformerBatchPrefillSimdgroupKernels} against host references, for whole tiles and for
 * batches and row counts that end inside a tile (the fragment-at-a-time write-back).
 *
 * <p>Activations and weights are small multiples of a quarter and Q8_0 scales are powers of two, so
 * every product and partial sum is exact in {@code float}: the projections must match their
 * reference bit for bit, which catches an index or transposition error that a tolerance could hide.
 * Only the gate/up kernels' {@code SiLU} is compared with a tolerance.
 */
// @formatter:on
public class TransformerBatchPrefillSimdgroupKernelsAccelTest {

    private static final float UNTOUCHED = -12345.5f;

    private static float value(int a, int b) {
        return (((a * 7 + b * 3) % 9) - 4) * 0.25f;
    }

    private static float weight(int row, int k) {
        return (((row * 5 + k * 11) % 7) - 3) * 0.25f;
    }

    private static HalfFloatArray fp16Weights(int rows, int n) {
        HalfFloatArray w = new HalfFloatArray(rows * n);
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < n; k++) {
                w.set(r * n + k, new HalfFloat(weight(r, k)));
            }
        }
        return w;
    }

    /** Q8_0 rows: per 32 columns a half scale of {@code 0.25 * 2^(block % 3)} and int8 values. */
    private static ByteArray q8Weights(int rows, int n) {
        int blocks = n / 32;
        ByteArray w = new ByteArray(rows * blocks * 34);
        for (int r = 0; r < rows; r++) {
            for (int b = 0; b < blocks; b++) {
                int base = (r * blocks + b) * 34;
                short bits = Float.floatToFloat16(q8Scale(b));
                w.set(base, (byte) bits);
                w.set(base + 1, (byte) (bits >> 8));
                for (int i = 0; i < 32; i++) {
                    w.set(base + 2 + i, (byte) q8Value(r, b * 32 + i));
                }
            }
        }
        return w;
    }

    private static float q8Scale(int block) {
        return 0.25f * (1 << (block % 3));
    }

    private static int q8Value(int row, int k) {
        return ((row * 13 + k * 5) % 23) - 11;
    }

    private static float q8Weight(int row, int k) {
        return q8Value(row, k) * q8Scale(k / 32);
    }

    private interface Weights {
        float at(int row, int k);
    }

    private static FloatArray activations(int batch, int n) {
        FloatArray x = new FloatArray(batch * n);
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < n; k++) {
                x.set(b * n + k, value(b, k));
            }
        }
        return x;
    }

    private static float dot(FloatArray x, int b, int n, Weights w, int row) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += x.get(b * n + k) * w.at(row, k);
        }
        return sum;
    }

    private static void run(TaskGraph graph, String name, int groups) {
        WorkerGrid1D worker =
                new WorkerGrid1D(groups * TransformerBatchPrefillSimdgroupKernels.THREADS);
        worker.setLocalWork(TransformerBatchPrefillSimdgroupKernels.THREADS, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".t", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void assumeMetal() {
        assumeTrue("SIMD-group matrices are Metal only", BatchPrefillGemmPolicy.tiled());
    }

    private static void checkResidual(boolean q8, int n, int d, int batch) {
        assumeMetal();
        FloatArray x = activations(batch, n);
        FloatArray out = new FloatArray(batch * d);
        float[] expected = new float[batch * d];
        Weights w =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < d; r++) {
                float start = value(r, b + 1);
                out.set(b * d + r, start);
                expected[b * d + r] = start + dot(x, b, n, w, r);
            }
        }
        String name = "simdgroupResidual" + (q8 ? "Q8" : "FP16") + d + "x" + batch;
        TaskGraph graph =
                new TaskGraph(name).transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out);
        if (q8) {
            ByteArray wq = q8Weights(d, n);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, wq);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                    new KernelContext(),
                    x,
                    out,
                    wq,
                    n,
                    d,
                    batch);
        } else {
            HalfFloatArray wh = fp16Weights(d, n);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, wh);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmFP16WithResidual,
                    new KernelContext(),
                    x,
                    out,
                    wh,
                    n,
                    d,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, name, ((d + 63) / 64) * ((batch + 63) / 64));
        for (int i = 0; i < expected.length; i++) {
            assertEquals(name + " element " + i, expected[i], out.get(i), 0.0f);
        }
    }

    private static void checkGateUp(boolean q8, int dim, int hiddenDim, int batch) {
        assumeMetal();
        FloatArray x = activations(batch, dim);
        FloatArray rms = new FloatArray(dim);
        rms.init(1.0f);
        FloatArray scale = new FloatArray(batch);
        scale.init(1.0f);
        FloatArray hb = new FloatArray(batch * hiddenDim);
        hb.init(UNTOUCHED);
        Weights w1 =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        Weights w3 = q8 ? (r, k) -> q8Weight(r + 1, k) : (r, k) -> weight(r + 1, k);
        String name = "simdgroupGateUp" + (q8 ? "Q8" : "FP16") + hiddenDim + "x" + batch;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, rms, scale, hb);
        if (q8) {
            ByteArray g = q8Weights(hiddenDim + 1, dim);
            ByteArray g1 = new ByteArray(hiddenDim * (dim / 32) * 34);
            ByteArray g3 = new ByteArray(hiddenDim * (dim / 32) * 34);
            int rowBytes = (dim / 32) * 34;
            for (int i = 0; i < hiddenDim * rowBytes; i++) {
                g1.set(i, g.get(i));
                g3.set(i, g.get(i + rowBytes));
            }
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, g1, g3);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ8,
                    new KernelContext(),
                    x,
                    hb,
                    rms,
                    scale,
                    g1,
                    g3,
                    dim,
                    hiddenDim,
                    batch);
        } else {
            HalfFloatArray g = fp16Weights(hiddenDim + 1, dim);
            HalfFloatArray g1 = new HalfFloatArray(hiddenDim * dim);
            HalfFloatArray g3 = new HalfFloatArray(hiddenDim * dim);
            for (int i = 0; i < hiddenDim * dim; i++) {
                g1.set(i, g.get(i));
                g3.set(i, g.get(i + dim));
            }
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, g1, g3);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpFP16,
                    new KernelContext(),
                    x,
                    hb,
                    rms,
                    scale,
                    g1,
                    g3,
                    dim,
                    hiddenDim,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, name, ((hiddenDim + 31) / 32) * ((batch + 63) / 64));
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < hiddenDim; r++) {
                float gate = dot(x, b, dim, w1, r);
                float up = dot(x, b, dim, w3, r);
                float expected = (gate / (1.0f + (float) Math.exp(-gate))) * up;
                assertEquals(
                        name + " token " + b + " row " + r,
                        expected,
                        hb.get(b * hiddenDim + r),
                        1e-4f * Math.max(1.0f, Math.abs(expected)));
            }
        }
    }

    private static void checkQkv(boolean q8, int dim, int qDim, int kvDim, int batch) {
        assumeMetal();
        FloatArray xf = activations(batch, dim);
        FloatArray q = new FloatArray(batch * qDim);
        FloatArray k = new FloatArray(batch * kvDim);
        FloatArray v = new FloatArray(batch * kvDim);
        q.init(UNTOUCHED);
        k.init(UNTOUCHED);
        v.init(UNTOUCHED);
        String name = "simdgroupQkv" + (q8 ? "Q8" : "FP16") + batch;
        TaskGraph graph = new TaskGraph(name);
        if (q8) {
            ByteArray wq = q8Weights(qDim, dim);
            ByteArray wk = q8Weights(kvDim, dim);
            ByteArray wv = q8Weights(kvDim, dim);
            graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, xf, q, k, v);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, wq, wk, wv);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
                    new KernelContext(),
                    xf,
                    q,
                    k,
                    v,
                    wq,
                    wk,
                    wv,
                    dim,
                    qDim,
                    kvDim,
                    batch);
        } else {
            HalfFloatArray xh = new HalfFloatArray(batch * dim);
            for (int i = 0; i < batch * dim; i++) {
                xh.set(i, new HalfFloat(xf.get(i)));
            }
            HalfFloatArray wq = fp16Weights(qDim, dim);
            HalfFloatArray wk = fp16Weights(kvDim, dim);
            HalfFloatArray wv = fp16Weights(kvDim, dim);
            graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, xh, q, k, v);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, wq, wk, wv);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFP16,
                    new KernelContext(),
                    xh,
                    q,
                    k,
                    v,
                    wq,
                    wk,
                    wv,
                    dim,
                    qDim,
                    kvDim,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, q, k, v);
        run(graph, name, ((qDim + 2 * kvDim) / 64) * ((batch + 63) / 64));
        Weights w =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < qDim; r++) {
                assertEquals(
                        name + " q " + b + "," + r,
                        dot(xf, b, dim, w, r),
                        q.get(b * qDim + r),
                        0.0f);
            }
            for (int r = 0; r < kvDim; r++) {
                assertEquals(
                        name + " k " + b + "," + r,
                        dot(xf, b, dim, w, r),
                        k.get(b * kvDim + r),
                        0.0f);
                assertEquals(
                        name + " v " + b + "," + r,
                        dot(xf, b, dim, w, r),
                        v.get(b * kvDim + r),
                        0.0f);
            }
        }
    }

    @Test
    public void fp16ResidualWholeTiles() {
        checkResidual(false, 128, 128, 128);
    }

    @Test
    public void fp16ResidualPartialTiles() {
        checkResidual(false, 128, 72, 70);
    }

    @Test
    public void q8ResidualWholeTiles() {
        checkResidual(true, 128, 128, 128);
    }

    @Test
    public void q8ResidualPartialTiles() {
        checkResidual(true, 128, 72, 70);
    }

    @Test
    public void fp16GateUp() {
        checkGateUp(false, 64, 40, 70);
    }

    @Test
    public void q8GateUp() {
        checkGateUp(true, 64, 40, 70);
    }

    @Test
    public void fp16QkvWholeAndPartialTiles() {
        checkQkv(false, 64, 128, 64, 128);
        checkQkv(false, 64, 128, 64, 70);
    }

    @Test
    public void q8QkvWholeAndPartialTiles() {
        checkQkv(true, 64, 128, 64, 128);
        checkQkv(true, 64, 128, 64, 70);
    }

    /**
     * The Q/K/V kernel as a single stored Q8_0 projection, the way {@code qwen35} batched prefill
     * calls it on Metal: key/value width zero, so every 64-row tile lies in the first output, and
     * that one buffer and weight bound to all three slots.
     */
    private static void checkQ8Store(int n, int d, int batch) {
        assumeMetal();
        FloatArray x = activations(batch, n);
        FloatArray out = new FloatArray(batch * d);
        out.init(UNTOUCHED);
        ByteArray w = q8Weights(d, n);
        String name = "simdgroupQ8Store" + d + "x" + batch;
        TaskGraph graph = new TaskGraph(name);
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        graph.task(
                "t",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
                new KernelContext(),
                x,
                out,
                out,
                out,
                w,
                w,
                w,
                n,
                d,
                0,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, name, (d / 64) * ((batch + 63) / 64));
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < d; r++) {
                assertEquals(
                        name + " " + b + "," + r,
                        dot(x, b, n, TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight, r),
                        out.get(b * d + r),
                        0.0f);
            }
        }
    }

    @Test
    public void q8StoreWholeAndPartialTiles() {
        checkQ8Store(128, 192, 128);
        checkQ8Store(128, 192, 70);
    }

    /** Fused Q/K/V rows (Phi-3's {@code wqkv}): rows {@code [0, qDim)} Q, then K, then V. */
    private static void checkQkvFused(boolean q8, int dim, int kvDim, int batch) {
        assumeMetal();
        int rows = dim + 2 * kvDim;
        FloatArray x = activations(batch, dim);
        FloatArray q = new FloatArray(batch * dim);
        FloatArray k = new FloatArray(batch * kvDim);
        FloatArray v = new FloatArray(batch * kvDim);
        q.init(UNTOUCHED);
        k.init(UNTOUCHED);
        v.init(UNTOUCHED);
        String name = "simdgroupQkvFused" + (q8 ? "Q8" : "FP16") + batch;
        TaskGraph graph =
                new TaskGraph(name).transferToDevice(DataTransferMode.EVERY_EXECUTION, x, q, k, v);
        if (q8) {
            ByteArray w = q8Weights(rows, dim);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFusedQ8,
                    new KernelContext(),
                    x,
                    q,
                    k,
                    v,
                    w,
                    dim,
                    dim,
                    kvDim,
                    batch);
        } else {
            HalfFloatArray w = fp16Weights(rows, dim);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFusedFP16,
                    new KernelContext(),
                    x,
                    q,
                    k,
                    v,
                    w,
                    dim,
                    dim,
                    kvDim,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, q, k, v);
        run(graph, name, (rows / 64) * ((batch + 63) / 64));
        Weights w =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < dim; r++) {
                assertEquals(
                        name + " q " + b + "," + r, dot(x, b, dim, w, r), q.get(b * dim + r), 0.0f);
            }
            for (int r = 0; r < kvDim; r++) {
                assertEquals(
                        name + " k " + b + "," + r,
                        dot(x, b, dim, w, dim + r),
                        k.get(b * kvDim + r),
                        0.0f);
                assertEquals(
                        name + " v " + b + "," + r,
                        dot(x, b, dim, w, dim + kvDim + r),
                        v.get(b * kvDim + r),
                        0.0f);
            }
        }
    }

    /** Fused gate and up rows (Phi-3's {@code wUp}): gate rows first, then up rows. */
    private static void checkGateUpFused(boolean q8, int dim, int hiddenDim, int batch) {
        assumeMetal();
        FloatArray x = activations(batch, dim);
        FloatArray rms = new FloatArray(dim);
        rms.init(1.0f);
        FloatArray scale = new FloatArray(batch);
        scale.init(1.0f);
        FloatArray hb = new FloatArray(batch * hiddenDim);
        hb.init(UNTOUCHED);
        String name = "simdgroupGateUpFused" + (q8 ? "Q8" : "FP16") + hiddenDim + "x" + batch;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, rms, scale, hb);
        if (q8) {
            ByteArray w = q8Weights(2 * hiddenDim, dim);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpFusedQ8,
                    new KernelContext(),
                    x,
                    hb,
                    rms,
                    scale,
                    w,
                    dim,
                    hiddenDim,
                    batch);
        } else {
            HalfFloatArray w = fp16Weights(2 * hiddenDim, dim);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpFusedFP16,
                    new KernelContext(),
                    x,
                    hb,
                    rms,
                    scale,
                    w,
                    dim,
                    hiddenDim,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, name, ((hiddenDim + 31) / 32) * ((batch + 63) / 64));
        Weights w =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < hiddenDim; r++) {
                float gate = dot(x, b, dim, w, r);
                float up = dot(x, b, dim, w, hiddenDim + r);
                float expected = (gate / (1.0f + (float) Math.exp(-gate))) * up;
                assertEquals(
                        name + " token " + b + " row " + r,
                        expected,
                        hb.get(b * hiddenDim + r),
                        1e-4f * Math.max(1.0f, Math.abs(expected)));
            }
        }
    }

    @Test
    public void fusedQkvWholeAndPartialTiles() {
        checkQkvFused(false, 64, 64, 128);
        checkQkvFused(false, 64, 64, 70);
        checkQkvFused(true, 64, 64, 70);
    }

    @Test
    public void fusedGateUp() {
        checkGateUpFused(false, 64, 40, 70);
        checkGateUpFused(true, 64, 40, 70);
    }

    /** Q4_0 rows: per 32 columns a half scale of {@code 0.25 * 2^(block % 3)} and nibbles 0..15. */
    private static ByteArray q4Weights(int rows, int n) {
        int blocks = n / 32;
        ByteArray w = new ByteArray(rows * blocks * 18);
        for (int r = 0; r < rows; r++) {
            for (int b = 0; b < blocks; b++) {
                int base = (r * blocks + b) * 18;
                short bits = Float.floatToFloat16(q8Scale(b));
                w.set(base, (byte) bits);
                w.set(base + 1, (byte) (bits >> 8));
                for (int i = 0; i < 16; i++) {
                    int lo = q4Nibble(r, b * 32 + i);
                    int hi = q4Nibble(r, b * 32 + 16 + i);
                    w.set(base + 2 + i, (byte) (lo | (hi << 4)));
                }
            }
        }
        return w;
    }

    private static int q4Nibble(int row, int k) {
        return (row * 7 + k * 3) % 16;
    }

    private static float q4Weight(int row, int k) {
        return (q4Nibble(row, k) - 8) * q8Scale(k / 32);
    }

    @Test
    public void q4ResidualQkvAndGateUp() {
        assumeMetal();
        int n = 128;
        int d = 72;
        int batch = 70;
        FloatArray x = activations(batch, n);
        FloatArray out = new FloatArray(batch * d);
        float[] expected = new float[batch * d];
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < d; r++) {
                out.set(b * d + r, value(r, b + 1));
                expected[b * d + r] =
                        value(r, b + 1)
                                + dot(
                                        x,
                                        b,
                                        n,
                                        TransformerBatchPrefillSimdgroupKernelsAccelTest::q4Weight,
                                        r);
            }
        }
        ByteArray w = q4Weights(d, n);
        TaskGraph graph =
                new TaskGraph("simdgroupQ4Residual")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        graph.task(
                "t",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0WithResidual,
                new KernelContext(),
                x,
                out,
                w,
                n,
                d,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, "simdgroupQ4Residual", ((d + 63) / 64) * ((batch + 63) / 64));
        for (int i = 0; i < expected.length; i++) {
            assertEquals("q4 residual element " + i, expected[i], out.get(i), 0.0f);
        }
    }

    /** The plain Q4_0 projection overwrites every element, partial tiles included. */
    @Test
    public void q4Projection() {
        assumeMetal();
        int n = 128;
        int d = 72;
        int batch = 70;
        FloatArray x = activations(batch, n);
        FloatArray out = new FloatArray(batch * d);
        out.init(UNTOUCHED);
        ByteArray w = q4Weights(d, n);
        TaskGraph graph =
                new TaskGraph("simdgroupQ4Projection")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        graph.task(
                "t",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0,
                new KernelContext(),
                x,
                out,
                w,
                n,
                d,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, "simdgroupQ4Projection", ((d + 63) / 64) * ((batch + 63) / 64));
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < d; r++) {
                float expected =
                        dot(x, b, n, TransformerBatchPrefillSimdgroupKernelsAccelTest::q4Weight, r);
                assertEquals(
                        "q4 projection token " + b + " row " + r,
                        expected,
                        out.get(b * d + r),
                        1e-4f * Math.max(1.0f, Math.abs(expected)));
            }
        }
    }

    /**
     * The strided projections write exactly their columns of a wider packed row from FP16
     * activations, partial tiles included, and leave every other column alone.
     */
    private static void checkHalfStrided(boolean q8) {
        assumeMetal();
        int n = 128;
        int d = 72;
        int batch = 70;
        int off = 40;
        int ldo = off + d + 24;
        FloatArray xf = activations(batch, n);
        HalfFloatArray x = new HalfFloatArray(batch * n);
        for (int i = 0; i < batch * n; i++) {
            x.set(i, new HalfFloat(xf.get(i)));
        }
        FloatArray out = new FloatArray(batch * ldo);
        out.init(UNTOUCHED);
        String name = q8 ? "simdgroupHalfQ8Strided" : "simdgroupHalfFP16Strided";
        TaskGraph graph =
                new TaskGraph(name).transferToDevice(DataTransferMode.EVERY_EXECUTION, x, out);
        if (q8) {
            ByteArray w = q8Weights(d, n);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmHalfQ8Strided,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch,
                    ldo,
                    off);
        } else {
            HalfFloatArray w = fp16Weights(d, n);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmHalfFP16Strided,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch,
                    ldo,
                    off);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, name, ((d + 63) / 64) * ((batch + 63) / 64));
        Weights w =
                q8
                        ? TransformerBatchPrefillSimdgroupKernelsAccelTest::q8Weight
                        : TransformerBatchPrefillSimdgroupKernelsAccelTest::weight;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < ldo; c++) {
                float got = out.get(b * ldo + c);
                if (c < off || c >= off + d) {
                    assertEquals(
                            name + " column " + c + " is not this projection's",
                            UNTOUCHED,
                            got,
                            0.0f);
                } else {
                    float expected = dot(xf, b, n, w, c - off);
                    assertEquals(
                            name + " token " + b + " row " + (c - off),
                            expected,
                            got,
                            1e-4f * Math.max(1.0f, Math.abs(expected)));
                }
            }
        }
    }

    @Test
    public void halfActivationsQ8Strided() {
        checkHalfStrided(true);
    }

    @Test
    public void halfActivationsFP16Strided() {
        checkHalfStrided(false);
    }

    /**
     * K-quant rows of {@code n} columns: random bytes, with each super-block's fp16 scales set to
     * about 0.06 so that every weight is finite.
     */
    private static ByteArray kQuantWeights(boolean q4k, int rows, int n, long seed) {
        int blockBytes = q4k ? 144 : 210;
        int blocks = rows * (n / 256);
        byte[] raw = new byte[blocks * blockBytes];
        new Random(seed).nextBytes(raw);
        int[] scaleOffsets = q4k ? new int[] {0, 2} : new int[] {208};
        for (int b = 0; b < blocks; b++) {
            for (int offset : scaleOffsets) {
                raw[b * blockBytes + offset] = (byte) (b & 0xFF);
                raw[b * blockBytes + offset + 1] = 0x2C;
            }
        }
        return ByteArray.fromArray(raw);
    }

    /** A K-quant weight, through the decode the device kernels are held against. */
    private static Weights kQuantWeight(boolean q4k, ByteArray w, int n) {
        return (row, k) -> {
            int block = row * (n / 256) + k / 256;
            return q4k
                    ? TransformerComputeKernelsQ4_K.decode(w, block * 144, k % 256)
                    : TransformerComputeKernelsQ6_K.decode(w, block * 210, k % 256);
        };
    }

    /** {@code sum |x * w|}, the scale of the rounding a reordered sum can differ by. */
    private static float absDot(FloatArray x, int b, int n, Weights w, int row) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += Math.abs(x.get(b * n + k) * w.at(row, k));
        }
        return sum;
    }

    /**
     * The plain and residual K-quant projections, partial tiles included: the plain one overwrites
     * every element, the residual one adds to what is there.
     */
    private static void checkKQuantProjection(boolean q4k, boolean residual) {
        assumeMetal();
        int n = 512;
        int d = 72;
        int batch = 70;
        FloatArray x = activations(batch, n);
        FloatArray out = new FloatArray(batch * d);
        for (int i = 0; i < batch * d; i++) {
            out.set(i, residual ? value(i % d, i / d + 1) : UNTOUCHED);
        }
        ByteArray w = kQuantWeights(q4k, d, n, q4k ? 41 : 61);
        String name = "simdgroup" + (q4k ? "Q4_K" : "Q6_K") + (residual ? "Residual" : "");
        TaskGraph graph = new TaskGraph(name).transferToDevice(DataTransferMode.EVERY_EXECUTION, x);
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, out);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        if (q4k && residual) {
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_KWithResidual,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch);
        } else if (q4k) {
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_K,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch);
        } else if (residual) {
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ6_KWithResidual,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch);
        } else {
            graph.task(
                    "t",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ6_K,
                    new KernelContext(),
                    x,
                    out,
                    w,
                    n,
                    d,
                    batch);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, name, ((d + 63) / 64) * ((batch + 63) / 64));
        Weights weights = kQuantWeight(q4k, w, n);
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < d; r++) {
                float expected = (residual ? value(r, b + 1) : 0.0f) + dot(x, b, n, weights, r);
                float tolerance = 1e-5f * absDot(x, b, n, weights, r) + 1e-4f;
                assertEquals(
                        name + " token " + b + " row " + r,
                        expected,
                        out.get(b * d + r),
                        tolerance);
            }
        }
    }

    @Test
    public void q4kProjection() {
        checkKQuantProjection(true, false);
    }

    @Test
    public void q4kResidual() {
        checkKQuantProjection(true, true);
    }

    @Test
    public void q6kProjection() {
        checkKQuantProjection(false, false);
    }

    @Test
    public void q6kResidual() {
        checkKQuantProjection(false, true);
    }

    /**
     * The Q4_K gate/up projection with the RMS weights and per-token scales applied as the
     * activations are staged, partial tiles included.
     */
    @Test
    public void q4kGateUp() {
        assumeMetal();
        int dim = 512;
        int hiddenDim = 40;
        int batch = 70;
        FloatArray x = activations(batch, dim);
        FloatArray rms = new FloatArray(dim);
        for (int k = 0; k < dim; k++) {
            rms.set(k, 0.5f + (k % 5) * 0.25f);
        }
        FloatArray scale = new FloatArray(batch);
        for (int b = 0; b < batch; b++) {
            scale.set(b, 0.75f + (b % 3) * 0.25f);
        }
        FloatArray normed = new FloatArray(batch * dim);
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < dim; k++) {
                normed.set(b * dim + k, rms.get(k) * scale.get(b) * x.get(b * dim + k));
            }
        }
        FloatArray hb = new FloatArray(batch * hiddenDim);
        hb.init(UNTOUCHED);
        ByteArray w1 = kQuantWeights(true, hiddenDim, dim, 71);
        ByteArray w3 = kQuantWeights(true, hiddenDim, dim, 73);
        String name = "simdgroupGateUpQ4_K";
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, rms, scale, hb);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, w1, w3);
        graph.task(
                "t",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ4_K,
                new KernelContext(),
                x,
                hb,
                rms,
                scale,
                w1,
                w3,
                dim,
                hiddenDim,
                batch);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, name, ((hiddenDim + 31) / 32) * ((batch + 63) / 64));
        Weights gateWeights = kQuantWeight(true, w1, dim);
        Weights upWeights = kQuantWeight(true, w3, dim);
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < hiddenDim; r++) {
                float gate = dot(normed, b, dim, gateWeights, r);
                float up = dot(normed, b, dim, upWeights, r);
                float expected = (gate / (1.0f + (float) Math.exp(-gate))) * up;
                float gateError = 1e-5f * absDot(normed, b, dim, gateWeights, r) + 1e-4f;
                float upError = 1e-5f * absDot(normed, b, dim, upWeights, r) + 1e-4f;
                float tolerance =
                        Math.abs(up) * gateError * 1.1f
                                + Math.abs(gate) * upError
                                + 1e-4f * Math.max(1.0f, Math.abs(expected));
                assertEquals(
                        name + " token " + b + " row " + r,
                        expected,
                        hb.get(b * hiddenDim + r),
                        tolerance);
            }
        }
    }
}
