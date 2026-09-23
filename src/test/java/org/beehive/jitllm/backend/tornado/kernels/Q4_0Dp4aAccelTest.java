package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The packed-integer {@code Q4_0} matrix-vector, at the 27B's own reduction width.
 *
 * <p>Three cases, deliberately separate, because they answer different questions and only the first
 * can be exact:
 *
 * <ol>
 *   <li><b>The integer dot product itself.</b> Block scales of exactly one and activations that
 *       quantize without rounding, so the kernel's output is an integer and is compared with {@code
 *       0.0f} tolerance. This is what says the nibble unpacking, the low/high pairing and the
 *       {@code -8 * sum} correction are right.
 *   <li><b>Against a reference that quantizes the activations the same way.</b> The only difference
 *       left is the order of summation, so the bound is tight.
 *   <li><b>Against the existing FP32 kernel.</b> That difference <b>is</b> the cost of quantizing
 *       the activations, and it is reported rather than hidden inside a tolerance: this case exists
 *       to measure a precision change, not to bless one.
 * </ol>
 */
// @formatter:on
public class Q4_0Dp4aAccelTest {

    /** The 27B's embedding width: a real reduction length, 160 blocks per row. */
    private static final int N = 5120;

    /** Enough rows to cover a workgroup grid without making the test slow. */
    private static final int D = 64;

    private static final int LOCAL = 128;
    private static final int QK = 32;
    private static final int BLOCK_BYTES = 18;

    /** Weights whose block scale is exactly one, so the integer dot product survives to the end. */
    private static byte[] unitScaleWeights(long seed) {
        Random random = new Random(seed);
        int blocks = D * (N / QK);
        byte[] raw = new byte[blocks * BLOCK_BYTES];
        for (int b = 0; b < blocks; b++) {
            int base = b * BLOCK_BYTES;
            raw[base] = 0x00;
            raw[base + 1] = 0x3C; // 1.0 in fp16
            for (int i = 0; i < 16; i++) {
                raw[base + 2 + i] = (byte) random.nextInt(256);
            }
        }
        return raw;
    }

    private static byte[] realisticWeights(long seed) {
        Random random = new Random(seed);
        int blocks = D * (N / QK);
        byte[] raw = new byte[blocks * BLOCK_BYTES];
        for (int b = 0; b < blocks; b++) {
            int base = b * BLOCK_BYTES;
            // A small positive scale that differs per block, and a negative one every third block.
            int bits = 0x2C00 | (b & 0xFF);
            if (b % 3 == 0) {
                bits |= 0x8000;
            }
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
            for (int i = 0; i < 16; i++) {
                raw[base + 2 + i] = (byte) random.nextInt(256);
            }
        }
        return raw;
    }

    // @formatter:off
    /**
     * The fused gate/up form: both projections and the SiLU that combines them.
     *
     * <p>Against a reference quantizing the activation identically, so what is being checked is the
     * kernel rather than the representation. Three things it can get wrong that the single
     * projection cannot: the two weight matrices sharing one activation's scale and correction, the
     * two halves of the reduction tree, and the SiLU being applied to the gate before the multiply
     * rather than after.
     *
     * <p>The activation carries a <b>zero block</b> — thirty-two exact zeros, whose scale is zero
     * and whose contribution must be zero rather than a division by it — alongside blocks of mixed
     * sign and very different magnitudes, so the per-block scales are nonuniform by construction.
     */
    // @formatter:on
    @Test
    public void theFusedGateUpMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        byte[] rawGate = realisticWeights(31337L);
        byte[] rawUp = realisticWeights(90210L);

        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            int block = i / QK;
            if (block == 2) {
                host[i] = 0.0f; // a zero block: scale zero, contribution zero
            } else if (block % 3 == 0) {
                host[i] = (float) (Math.sin(0.013 * i) * 1e-3); // tiny magnitudes
            } else if (block % 3 == 1) {
                host[i] = (float) (Math.cos(0.021 * i) * 40.0); // large, both signs
            } else {
                host[i] = (i % 2 == 0 ? 1 : -1) * (0.25f + (i % 11) * 0.5f);
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
        FloatArray hb = new FloatArray(D);
        hb.init(0.0f);
        ByteArray w1 = toDevice(rawGate);
        ByteArray w3 = toDevice(rawUp);

        TaskGraph graph =
                new TaskGraph("fused")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                x,
                                w1,
                                w3,
                                quants,
                                scales,
                                sums,
                                hb)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                x,
                                quants,
                                scales,
                                sums)
                        .task(
                                "gateUp",
                                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                hb,
                                w1,
                                w3,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb, quants, scales, sums);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(N);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("fused.quantize", blocks);
        WorkerGrid1D rows = new WorkerGrid1D(D * LOCAL);
        rows.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("fused.gateUp", rows);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        Quantized reference = new Quantized(host);
        // The zero block must have come out as a zero scale, not a NaN.
        assertEquals("the zero block's scale", 0.0f, reference.scales[2], 0.0f);
        assertEquals("the zero block's scale on the device", 0.0f, scales.get(2), 0.0f);
        assertEquals("the zero block's sum on the device", 0, sums.get(2));

        double largest = 0;
        double worst = 0;
        int worstRow = -1;
        for (int row = 0; row < D; row++) {
            double gate = 0;
            double up = 0;
            for (int block = 0; block < N / QK; block++) {
                int gateDot = 0;
                int upDot = 0;
                for (int i = 0; i < QK; i++) {
                    int packedQuant = reference.quants[block * (QK / 4) + i / 4];
                    int quant = (byte) ((packedQuant >> ((i % 4) * 8)) & 0xFF);
                    gateDot += (nibbleOf(rawGate, row, block * QK + i) - 8) * quant;
                    upDot += (nibbleOf(rawUp, row, block * QK + i) - 8) * quant;
                }
                gate += (double) scaleOf(rawGate, row, block) * reference.scales[block] * gateDot;
                up += (double) scaleOf(rawUp, row, block) * reference.scales[block] * upDot;
            }
            double silu = gate / (1.0 + Math.exp(-gate));
            double expected = silu * up;
            float got = hb.get(row);
            assertTrue("row " + row + " is " + got, Float.isFinite(got));
            largest = Math.max(largest, Math.abs(expected));
            if (Math.abs(expected - got) > worst) {
                worst = Math.abs(expected - got);
                worstRow = row;
            }
        }
        System.out.printf(
                "[FUSED] against the same-quantization reference: worst %.6g at row %d,"
                        + " largest |out| %.6g%n",
                worst, worstRow, largest);
        assertTrue(
                "worst " + worst + " at row " + worstRow + " against largest " + largest,
                worst <= 1e-4 * largest);
    }

    // @formatter:off
    /**
     * The residual form: {@code hb[row] += w[row]·x}, against the same-quantization reference.
     *
     * <p>What it adds over the plain case is the accumulate-into-destination, so the destination is
     * seeded with a value of the same order as the projection's own output — a residual that
     * vanished, or was applied twice, would otherwise hide inside the sum.
     *
     * <p>The activation carries a zero block and blocks of mixed sign and very different
     * magnitudes, so the per-block scales are nonuniform and the packing signed.
     */
    // @formatter:on
    @Test
    public void theResidualFormMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        byte[] raw = realisticWeights(5150L);
        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            int block = i / QK;
            if (block == 5) {
                host[i] = 0.0f;
            } else if (block % 4 == 0) {
                host[i] = (float) (Math.sin(0.011 * i) * 1e-3);
            } else if (block % 4 == 1) {
                host[i] = (float) (Math.cos(0.023 * i) * 30.0);
            } else {
                host[i] = (i % 2 == 0 ? 1 : -1) * (0.5f + (i % 13) * 0.25f);
            }
        }
        float[] seed = new float[D];
        for (int i = 0; i < D; i++) {
            seed[i] = (float) Math.cos(0.07 * i) * 20.0f;
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
        FloatArray hb = new FloatArray(D);
        for (int i = 0; i < D; i++) {
            hb.set(i, seed[i]);
        }
        ByteArray w = toDevice(raw);

        TaskGraph graph =
                new TaskGraph("residual")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, quants, scales, sums, hb)
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
                                TransformerComputeKernelsQ4_0
                                        ::matrixVectorGenericWithResidualQ4_0DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                hb,
                                w,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb, scales, sums);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(N);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("residual.quantize", blocks);
        WorkerGrid1D rows = new WorkerGrid1D(D * LOCAL);
        rows.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("residual.matvec", rows);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        Quantized reference = new Quantized(host);
        assertEquals("the zero block's scale", 0.0f, reference.scales[5], 0.0f);
        assertEquals("the zero block's scale on the device", 0.0f, scales.get(5), 0.0f);
        assertEquals("the zero block's sum on the device", 0, sums.get(5));

        double largest = 0;
        double worst = 0;
        for (int row = 0; row < D; row++) {
            double sum = 0;
            for (int block = 0; block < N / QK; block++) {
                int dot = 0;
                for (int i = 0; i < QK; i++) {
                    int packedQuant = reference.quants[block * (QK / 4) + i / 4];
                    int quant = (byte) ((packedQuant >> ((i % 4) * 8)) & 0xFF);
                    dot += (nibbleOf(raw, row, block * QK + i) - 8) * quant;
                }
                sum += (double) scaleOf(raw, row, block) * reference.scales[block] * dot;
            }
            double expected = seed[row] + sum;
            float got = hb.get(row);
            assertTrue("row " + row + " is " + got, Float.isFinite(got));
            largest = Math.max(largest, Math.abs(expected));
            worst = Math.max(worst, Math.abs(expected - got));
        }
        System.out.printf(
                "[RESIDUAL] against the same-quantization reference: worst %.6g,"
                        + " largest |out| %.6g%n",
                worst, largest);
        assertTrue("worst " + worst + " against largest " + largest, worst <= 1e-4 * largest);
    }

    // @formatter:off
    /**
     * A narrower consumer ignores whatever a wider one left in the tail of the shared scratch.
     *
     * <p>The three quantization arrays are sized for the widest activation any projection reads —
     * the feed-forward's hidden width — and the narrower projections use a prefix. That is only
     * safe if every consumer is governed by the <b>logical</b> activation length it was given
     * rather than by the capacity of the array, and this is the case that would catch it if one
     * were not: the tail is filled with values that would produce a grossly different answer if
     * they were read.
     *
     * <p>Both the quantization and the projection are run over the narrow length while the arrays
     * stay wide, and the result is compared against the same run on arrays with a clean tail.
     */
    // @formatter:on
    @Test
    public void aNarrowProjectionIgnoresTheWideScratchTail() throws Exception {
        byte[] raw = realisticWeights(8080L);
        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            host[i] = (float) Math.sin(0.017 * i) * (1.0f + (i % 5) * 0.3f);
        }
        float[] clean = runWithScratch(host, raw, 1, false);
        float[] poisoned = runWithScratch(host, raw, 4, true);
        for (int row = 0; row < D; row++) {
            assertTrue("row " + row + " is " + poisoned[row], Float.isFinite(poisoned[row]));
            assertEquals(
                    "row " + row + " changed when the scratch tail was poisoned",
                    clean[row],
                    poisoned[row],
                    0.0f);
        }
    }

    /** Runs the narrow projection with scratch {@code capacity} times longer than it needs. */
    private static float[] runWithScratch(float[] host, byte[] raw, int capacity, boolean poison)
            throws Exception {
        FloatArray x = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            x.set(i, host[i]);
        }
        IntArray quants = new IntArray(N / 4 * capacity);
        FloatArray scales = new FloatArray(N / QK * capacity);
        IntArray sums = new IntArray(N / QK * capacity);
        quants.init(0);
        scales.init(0.0f);
        sums.init(0);
        if (poison) {
            for (int i = N / 4; i < quants.getSize(); i++) {
                quants.set(i, 0x7F7F7F7F);
            }
            for (int i = N / QK; i < scales.getSize(); i++) {
                scales.set(i, 1.0e6f);
                sums.set(i, 4064);
            }
        }
        FloatArray out = new FloatArray(D);
        out.init(0.0f);
        ByteArray w = toDevice(raw);

        TaskGraph graph =
                new TaskGraph("tail")
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
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                out,
                                w,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        GridScheduler scheduler = new GridScheduler();
        // The grid is the logical length, not the capacity: the quantization writes the prefix.
        WorkerGrid1D blocks = new WorkerGrid1D(N);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("tail.quantize", blocks);
        WorkerGrid1D rows = new WorkerGrid1D(D * LOCAL);
        rows.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("tail.matvec", rows);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    private static int nibbleOf(byte[] raw, int row, int element) {
        int block = element / QK;
        int within = element - block * QK;
        int base = (row * (N / QK) + block) * BLOCK_BYTES;
        int packed = raw[base + 2 + (within & 15)] & 0xFF;
        return within < 16 ? (packed & 0xF) : ((packed >> 4) & 0xF);
    }

    private static float scaleOf(byte[] raw, int row, int block) {
        int base = (row * (N / QK) + block) * BLOCK_BYTES;
        int bits = (raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8);
        return new HalfFloat((short) bits).getFloat32();
    }

    /** The host's copy of the kernel's activation quantization, block by block. */
    private static final class Quantized {
        final int[] quants = new int[N / 4];
        final float[] scales = new float[N / QK];
        final int[] sums = new int[N / QK];

        Quantized(float[] x) {
            for (int block = 0; block < N / QK; block++) {
                int base = block * QK;
                float maxAbs = 0;
                for (int i = 0; i < QK; i++) {
                    maxAbs = Math.max(maxAbs, Math.abs(x[base + i]));
                }
                scales[block] = maxAbs / 127.0f;
                float inverse = maxAbs > 0 ? 127.0f / maxAbs : 0.0f;
                int sum = 0;
                for (int g = 0; g < QK / 4; g++) {
                    int packed = 0;
                    for (int lane = 0; lane < 4; lane++) {
                        float v = x[base + g * 4 + lane] * inverse;
                        int q = (int) (v + (v >= 0 ? 0.5f : -0.5f));
                        q = Math.min(127, Math.max(-127, q));
                        sum += q;
                        packed |= (q & 0xFF) << (lane * 8);
                    }
                    quants[block * (QK / 4) + g] = packed;
                }
                sums[block] = sum;
            }
        }
    }

    private static int nibble(byte[] raw, int row, int element) {
        int block = element / QK;
        int within = element - block * QK;
        int base = (row * (N / QK) + block) * BLOCK_BYTES;
        int packed = raw[base + 2 + (within & 15)] & 0xFF;
        return within < 16 ? (packed & 0xF) : ((packed >> 4) & 0xF);
    }

    private static float blockScale(byte[] raw, int row, int block) {
        int base = (row * (N / QK) + block) * BLOCK_BYTES;
        int bits = (raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8);
        return new HalfFloat((short) bits).getFloat32();
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    /** Runs the quantization and the projection, exactly as a layer would chain them. */
    private static float[] runDp4a(float[] host, byte[] raw) throws Exception {
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
        FloatArray out = new FloatArray(D);
        out.init(0.0f);
        ByteArray w = toDevice(raw);

        TaskGraph graph =
                new TaskGraph("dp4a")
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
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                out,
                                w,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D quantizeWorker = new WorkerGrid1D(N);
        quantizeWorker.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("dp4a.quantize", quantizeWorker);
        WorkerGrid1D matvecWorker = new WorkerGrid1D(D * LOCAL);
        matvecWorker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("dp4a.matvec", matvecWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    /** The existing FP32 kernel on the same inputs. */
    private static float[] runFp32(float[] host, byte[] raw) throws Exception {
        FloatArray x = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            x.set(i, host[i]);
        }
        FloatArray out = new FloatArray(D);
        out.init(0.0f);
        ByteArray w = toDevice(raw);
        TaskGraph graph =
                new TaskGraph("fp32")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, w, out)
                        .task(
                                "matvec",
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                                new KernelContext(),
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D worker = new WorkerGrid1D(D * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("fp32.matvec", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    /**
     * The integer dot product, exactly.
     *
     * <p>Block scales of one and activations that are whole multiples of the quantization step, so
     * nothing rounds and the answer is an integer that the kernel must reproduce bit for bit.
     */
    @Test
    public void thePackedIntegerDotProductIsExact() throws Exception {
        byte[] raw = unitScaleWeights(20260911L);
        Random random = new Random(7L);
        float[] host = new float[N];
        for (int block = 0; block < N / QK; block++) {
            // One element of each block is +127 and one is -127, so the block's scale is exactly
            // one and every other element is already an integer in range.
            for (int i = 0; i < QK; i++) {
                host[block * QK + i] = random.nextInt(255) - 127;
            }
            host[block * QK] = 127;
            host[block * QK + 1] = -127;
        }

        float[] got = runDp4a(host, raw);

        for (int row = 0; row < D; row++) {
            long expected = 0;
            for (int i = 0; i < N; i++) {
                expected += (long) (nibble(raw, row, i) - 8) * (long) host[i];
            }
            assertEquals("row " + row, (float) expected, got[row], 0.0f);
        }
    }

    /** Against a reference that quantizes the activations exactly as the kernel does. */
    @Test
    public void itMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        byte[] raw = realisticWeights(4242L);
        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            host[i] = (float) Math.sin(0.017 * i) * (1.0f + (i % 7) * 0.25f);
        }

        float[] got = runDp4a(host, raw);
        Quantized q = new Quantized(host);

        double largest = 0;
        double worst = 0;
        for (int row = 0; row < D; row++) {
            double expected = 0;
            for (int block = 0; block < N / QK; block++) {
                int dot = 0;
                for (int i = 0; i < QK; i++) {
                    int packed = q.quants[block * (QK / 4) + i / 4];
                    int qx = (byte) ((packed >> ((i % 4) * 8)) & 0xFF);
                    dot += (nibble(raw, row, block * QK + i) - 8) * qx;
                }
                expected += (double) blockScale(raw, row, block) * q.scales[block] * dot;
            }
            largest = Math.max(largest, Math.abs(expected));
            worst = Math.max(worst, Math.abs(expected - got[row]));
            assertTrue("row " + row + " is " + got[row], Float.isFinite(got[row]));
        }
        System.out.printf(
                "[DP4A] against the same-quantization reference: worst %.6g, largest |out| %.6g%n",
                worst, largest);
        assertTrue(
                "worst deviation " + worst + " against largest output " + largest,
                worst <= 1e-4 * largest);
    }

    /**
     * Against the existing FP32 kernel: this difference is the precision change itself.
     *
     * <p>Reported, and bounded only loosely. What the bound is <b>for</b> is catching a defect that
     * changes the answer by more than quantizing the activations can; deciding whether the change
     * is acceptable in a model is not a kernel test's business.
     */
    @Test
    public void theCostOfQuantizingTheActivationsIsMeasured() throws Exception {
        byte[] raw = realisticWeights(99L);
        float[] host = new float[N];
        for (int i = 0; i < N; i++) {
            host[i] = (float) Math.sin(0.017 * i) * (1.0f + (i % 7) * 0.25f);
        }

        float[] dp4a = runDp4a(host, raw);
        float[] fp32 = runFp32(host, raw);

        double largest = 0;
        double worst = 0;
        double sumSquaredDifference = 0;
        double sumSquaredReference = 0;
        for (int row = 0; row < D; row++) {
            assertTrue("row " + row + " is " + dp4a[row], Float.isFinite(dp4a[row]));
            largest = Math.max(largest, Math.abs(fp32[row]));
            worst = Math.max(worst, Math.abs(fp32[row] - dp4a[row]));
            sumSquaredDifference += (fp32[row] - dp4a[row]) * (double) (fp32[row] - dp4a[row]);
            sumSquaredReference += fp32[row] * (double) fp32[row];
        }
        double relL2 = Math.sqrt(sumSquaredDifference / sumSquaredReference);
        System.out.printf(
                "[DP4A] against the FP32 kernel: worst %.6g (%.4g%% of the largest output),"
                        + " relative L2 %.6g%n",
                worst, 100.0 * worst / largest, relL2);
        assertTrue(
                "relative L2 " + relL2 + " is larger than activation quantization explains",
                relL2 < 5e-3);
    }
}
