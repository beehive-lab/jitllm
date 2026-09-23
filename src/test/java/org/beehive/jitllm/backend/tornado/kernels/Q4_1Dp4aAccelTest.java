package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.tensor.standard.Q4_1FloatTensor;
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
 * The packed-integer {@code Q4_1} feed-forward projection against a reference that quantizes the
 * activation the same way.
 *
 * <p>What needs proving is the <b>minimum term</b>. A Q4_1 weight is {@code d * q + m} with no
 * recentring, so the block contribution decomposes as {@code sx * (d * Σ q·xq + m * Σ xq)} and the
 * second half is carried entirely by the activation's sum of quants. Getting it wrong — dropping
 * it, scaling it by the wrong factor, or applying Q4_0's {@code -8} recentring to it — changes the
 * answer by an amount that grows with the minimum, so the weights below carry minimums of both
 * signs and of a magnitude comparable to the scale.
 *
 * <p>The activation covers the cases that make the term observable: a block that is entirely zero
 * (its scale and its sum must both be zero, so the minimum must contribute nothing), neighbouring
 * values of opposite sign that cancel, and magnitudes three orders apart. The destination is seeded
 * so the residual add is checked rather than assumed.
 */
// @formatter:on
public class Q4_1Dp4aAccelTest {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 20;
    private static final int LOCAL = 128;

    private static void putHalf(byte[] raw, int offset, float value) {
        int bits = halfBits(value);
        raw[offset] = (byte) (bits & 0xFF);
        raw[offset + 1] = (byte) ((bits >> 8) & 0xFF);
    }

    /** Round-to-nearest fp32 -> fp16 bits, enough for the modest values used here. */
    private static int halfBits(float value) {
        int f = Float.floatToIntBits(value);
        int sign = (f >>> 16) & 0x8000;
        int val = (f & 0x7FFFFFFF) + 0x1000;
        if (val >= 0x47800000) {
            return sign | 0x7C00;
        }
        if (val < 0x38800000) {
            return sign;
        }
        return sign | ((val - 0x38000000) >>> 13);
    }

    @Test
    public void thePackedProjectionMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        assumeTrue("no tensor-core-capable device", TensorCoreSupport.isTensorCoreCapableBackend());

        int n = 8 * QK; // 256 inputs, eight blocks
        int d = 96; // output rows
        Random random = new Random(20260914L);

        byte[] raw = new byte[d * (n / QK) * BLOCK_BYTES];
        for (int block = 0; block < d * (n / QK); block++) {
            int base = block * BLOCK_BYTES;
            // A scale that varies per block, and a minimum of both signs and comparable size.
            putHalf(raw, base, 0.03f + 0.01f * (block % 5));
            putHalf(raw, base + 2, ((block % 3) - 1) * (0.05f + 0.02f * (block % 4)));
            for (int i = 0; i < QK / 2; i++) {
                raw[base + 4 + i] = (byte) random.nextInt(256);
            }
        }

        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            int block = i / QK;
            if (block == 3) {
                host[i] = 0.0f; // the zero block: scale and sum must both be zero
            } else if (block % 3 == 0) {
                host[i] = (float) (Math.sin(0.013 * i) * 1e-3);
            } else if (block % 3 == 1) {
                host[i] = (float) (Math.cos(0.021 * i) * 25.0);
            } else {
                host[i] = (i % 2 == 0 ? 1 : -1) * (0.5f + (i % 11) * 0.4f); // cancelling neighbours
            }
        }

        FloatArray x = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            x.set(i, host[i]);
        }
        IntArray quants = new IntArray(n / 4);
        FloatArray scales = new FloatArray(n / QK);
        IntArray sums = new IntArray(n / QK);
        quants.init(0);
        scales.init(0.0f);
        sums.init(0);
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        // Seeded, so the residual add is part of what is compared.
        float[] seed = new float[d];
        FloatArray out = new FloatArray(d);
        for (int row = 0; row < d; row++) {
            seed[row] = (row % 2 == 0 ? 1 : -1) * (0.25f + 0.5f * row);
            out.set(row, seed[row]);
        }

        TaskGraph graph =
                new TaskGraph("q41dp4a")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, quants, scales, sums, out)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                x,
                                quants,
                                scales,
                                sums)
                        .task(
                                "matvec",
                                TransformerComputeKernelsQ4_1
                                        ::matrixVectorGenericWithResidualQ4_1DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                out,
                                w,
                                n,
                                d,
                                LOCAL)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION, out, quants, scales, sums);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(n);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("q41dp4a.quantize", blocks);
        WorkerGrid1D rows = new WorkerGrid1D(d * LOCAL);
        rows.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("q41dp4a.matvec", rows);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        assertEquals("the zero block's scale on the device", 0.0f, scales.get(3), 0.0f);
        assertEquals("the zero block's sum on the device", 0, sums.get(3));

        // The reference: the same block quantization, the host tensor's own Q4_1 decode, summed in
        // double, plus the same seed.
        int[] hostQuants = new int[n / 4];
        float[] hostScales = new float[n / QK];
        for (int block = 0; block < n / QK; block++) {
            int base = block * QK;
            float maxAbs = 0;
            for (int i = 0; i < QK; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(host[base + i]));
            }
            hostScales[block] = maxAbs / 127.0f;
            float inverse = maxAbs > 0 ? 127.0f / maxAbs : 0.0f;
            for (int g = 0; g < QK / 4; g++) {
                int packed = 0;
                for (int lane = 0; lane < 4; lane++) {
                    float v = host[base + g * 4 + lane] * inverse;
                    int q = (int) (v + (v >= 0 ? 0.5f : -0.5f));
                    q = Math.min(127, Math.max(-127, q));
                    packed |= (q & 0xFF) << (lane * 8);
                }
                hostQuants[block * (QK / 4) + g] = packed;
            }
        }

        double largest = 0;
        double worst = 0;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            Q4_1FloatTensor reference = new Q4_1FloatTensor(d * n, segment);
            for (int row = 0; row < d; row++) {
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    int block = i / QK;
                    int quant = (byte) ((hostQuants[i / 4] >> ((i % 4) * 8)) & 0xFF);
                    sum += reference.getFloat(row * n + i) * hostScales[block] * quant;
                }
                sum += seed[row];
                float got = out.get(row);
                assertTrue("row " + row + " is " + got, Float.isFinite(got));
                largest = Math.max(largest, Math.abs(sum));
                worst = Math.max(worst, Math.abs(sum - got));
            }
        }
        System.out.printf(
                "[Q41-DP4A] against the same-quantization reference: worst %.6g, largest |out|"
                        + " %.6g over %d rows of %d%n",
                worst, largest, d, n);
        assertTrue("worst " + worst + " against largest " + largest, worst <= 1e-4 * largest);
    }
}
