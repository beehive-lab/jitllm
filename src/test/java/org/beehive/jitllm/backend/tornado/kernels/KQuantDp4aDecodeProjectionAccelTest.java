package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The packed-integer K-quant projections a decode layer runs: the Q4_K projection, its residual
 * form, the fused Q4_K gate/up with SwiGLU, and the residual Q6_K projection, each against a
 * reference that quantizes the activation the same way and decodes the weights with the host tensor
 * that defines the format.
 *
 * <p>Random super-blocks (every scale and minimum bit pattern, both nibbles), activations whose
 * blocks span three orders of magnitude and one all-zero block, and the 64-lane rows the decode
 * layers dispatch, over enough super-blocks that every warp of the row carries work.
 */
// @formatter:on
public class KQuantDp4aDecodeProjectionAccelTest {

    private static final int QK_K = 256;
    private static final int QK = 32;
    private static final int Q4_K_BYTES = 144;
    private static final int Q6_K_BYTES = 210;
    private static final int LOCAL = 64;
    private static final int N = 20 * QK_K;
    private static final int ROWS = 48;

    @Test
    public void q4_KProjectionMatchesTheReference() throws Exception {
        byte[] raw = q4_KWeights(new Random(11L));
        float[] reference = reference(raw, true);
        float[] got = run("plain", raw, null, true);
        assertClose("Q4_K", reference, got);
    }

    @Test
    public void q4_KResidualProjectionAddsToTheResidual() throws Exception {
        byte[] raw = q4_KWeights(new Random(12L));
        float[] reference = reference(raw, true);
        float[] residual = residual();
        float[] got = run("residual", raw, residual, true);
        for (int r = 0; r < ROWS; r++) {
            reference[r] += residual[r];
        }
        assertClose("Q4_K residual", reference, got);
    }

    @Test
    public void q6_KResidualProjectionAddsToTheResidual() throws Exception {
        byte[] raw = q6_KWeights(new Random(13L));
        float[] reference = reference(raw, false);
        float[] residual = residual();
        float[] got = run("residual", raw, residual, false);
        for (int r = 0; r < ROWS; r++) {
            reference[r] += residual[r];
        }
        assertClose("Q6_K residual", reference, got);
    }

    @Test
    public void fusedQ4_KGateUpMatchesSwiGLUOfTheTwoProjections() throws Exception {
        byte[] gateRaw = q4_KWeights(new Random(14L));
        byte[] upRaw = q4_KWeights(new Random(15L));
        float[] gate = reference(gateRaw, true);
        float[] up = reference(upRaw, true);
        float[] expected = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            double g = gate[r];
            expected[r] = (float) (g / (1.0 + Math.exp(-g)) * up[r]);
        }

        Activation a = activation();
        ByteArray w1 = bytes(gateRaw);
        ByteArray w3 = bytes(upRaw);
        FloatArray out = new FloatArray(ROWS);
        out.init(0.0f);
        TaskGraph graph =
                new TaskGraph("gateup")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                a.x,
                                w1,
                                w3,
                                a.quants,
                                a.scales,
                                a.sums,
                                out)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                a.x,
                                a.quants,
                                a.scales,
                                a.sums)
                        .task(
                                "matvec",
                                TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_KDP4A,
                                new KernelContext(),
                                a.quants,
                                a.scales,
                                a.sums,
                                out,
                                w1,
                                w3,
                                N,
                                ROWS,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        execute(graph, "gateup");
        assertClose("fused Q4_K gate/up", expected, out.toHeapArray());
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private record Activation(
            float[] host, FloatArray x, IntArray quants, FloatArray scales, IntArray sums) {}

    private static Activation activation() {
        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            int block = i / QK;
            if (block == 7) {
                host[i] = 0.0f;
            } else if (block % 3 == 0) {
                host[i] = (float) (Math.sin(0.013 * i) * 1e-2);
            } else if (block % 3 == 1) {
                host[i] = (float) (Math.cos(0.021 * i) * 4.0);
            } else {
                host[i] = (i % 2 == 0 ? 1 : -1) * (0.05f + (i % 11) * 0.04f);
            }
        }
        FloatArray x = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            x.set(i, host[i]);
        }
        IntArray quants = new IntArray(N / 4);
        FloatArray scales = new FloatArray(N / QK);
        IntArray sums = new IntArray(N / QK);
        quants.init(0);
        scales.init(0.0f);
        sums.init(0);
        return new Activation(host, x, quants, scales, sums);
    }

    private static float[] residual() {
        float[] r = new float[ROWS];
        for (int i = 0; i < ROWS; i++) {
            r[i] = (i % 2 == 0 ? 1 : -1) * (0.25f + 0.1f * i);
        }
        return r;
    }

    /** Random Q4_K super-blocks with a small positive {@code d} and {@code dmin}. */
    private static byte[] q4_KWeights(Random random) {
        int blocks = ROWS * (N / QK_K);
        byte[] raw = new byte[blocks * Q4_K_BYTES];
        random.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * Q4_K_BYTES;
            // d about 0.002, dmin about 0.001: halves 0x1818 and 0x1418.
            raw[base] = 0x18;
            raw[base + 1] = 0x18;
            raw[base + 2] = 0x18;
            raw[base + 3] = 0x14;
        }
        return raw;
    }

    /** Random Q6_K super-blocks with a small positive {@code d} and random signed scales. */
    private static byte[] q6_KWeights(Random random) {
        int blocks = ROWS * (N / QK_K);
        byte[] raw = new byte[blocks * Q6_K_BYTES];
        random.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * Q6_K_BYTES;
            raw[base + 208] = 0x18;
            raw[base + 209] = 0x18;
        }
        return raw;
    }

    /** The host decode of each row against the activation quantized as the device does. */
    private static float[] reference(byte[] raw, boolean q4) {
        float[] host = activation().host;
        int[] quants = new int[N];
        float[] scales = new float[N / QK];
        for (int block = 0; block < N / QK; block++) {
            int base = block * QK;
            float maxAbs = 0;
            for (int i = 0; i < QK; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(host[base + i]));
            }
            scales[block] = maxAbs / 127.0f;
            float inverse = maxAbs > 0 ? 127.0f / maxAbs : 0.0f;
            for (int i = 0; i < QK; i++) {
                float v = host[base + i] * inverse;
                int q = (int) (v + (v >= 0 ? 0.5f : -0.5f));
                quants[base + i] = Math.min(127, Math.max(-127, q));
            }
        }
        float[] out = new float[ROWS];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            Q4_KFloatTensor q4Tensor = q4 ? new Q4_KFloatTensor(ROWS * N, segment) : null;
            Q6_KFloatTensor q6Tensor = q4 ? null : new Q6_KFloatTensor(ROWS * N, segment);
            for (int row = 0; row < ROWS; row++) {
                double sum = 0;
                for (int i = 0; i < N; i++) {
                    float w = q4 ? q4Tensor.getFloat(row * N + i) : q6Tensor.getFloat(row * N + i);
                    sum += w * scales[i / QK] * quants[i];
                }
                out[row] = (float) sum;
            }
        }
        return out;
    }

    /** The plain or residual projection on the device; {@code residual} null for the plain one. */
    private static float[] run(String name, byte[] raw, float[] residual, boolean q4)
            throws Exception {
        Activation a = activation();
        ByteArray w = bytes(raw);
        FloatArray out = new FloatArray(ROWS);
        for (int r = 0; r < ROWS; r++) {
            out.set(r, residual == null ? 0.0f : residual[r]);
        }
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                a.x,
                                w,
                                a.quants,
                                a.scales,
                                a.sums,
                                out)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                a.x,
                                a.quants,
                                a.scales,
                                a.sums);
        if (residual == null) {
            graph.task(
                    "matvec",
                    TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KDP4A,
                    new KernelContext(),
                    a.quants,
                    a.scales,
                    a.sums,
                    out,
                    w,
                    N,
                    ROWS,
                    LOCAL);
        } else if (q4) {
            graph.task(
                    "matvec",
                    TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_KDP4A,
                    new KernelContext(),
                    a.quants,
                    a.scales,
                    a.sums,
                    out,
                    w,
                    N,
                    ROWS,
                    LOCAL);
        } else {
            graph.task(
                    "matvec",
                    TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_KDP4A,
                    new KernelContext(),
                    a.quants,
                    a.scales,
                    a.sums,
                    out,
                    w,
                    N,
                    ROWS,
                    LOCAL);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        execute(graph, name);
        return out.toHeapArray();
    }

    private static void execute(TaskGraph graph, String name) throws Exception {
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(N);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid(name + ".quantize", blocks);
        WorkerGrid1D rows = new WorkerGrid1D(ROWS * LOCAL);
        rows.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid(name + ".matvec", rows);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    private static ByteArray bytes(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static void assertClose(String what, float[] expected, float[] got) {
        double largest = 0;
        double worst = 0;
        for (int r = 0; r < ROWS; r++) {
            assertTrue(what + " row " + r + " is " + got[r], Float.isFinite(got[r]));
            largest = Math.max(largest, Math.abs(expected[r]));
            worst = Math.max(worst, Math.abs(expected[r] - got[r]));
        }
        System.out.printf(
                "[KQ-DP4A] %s: worst %.6g, largest |out| %.6g over %d rows of %d%n",
                what, worst, largest, ROWS, N);
        assertTrue(
                what + ": worst " + worst + " against largest " + largest,
                worst <= 1e-4 * largest + 1e-5);
    }
}
