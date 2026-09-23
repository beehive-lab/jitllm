package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.List;
import java.util.Random;
import org.beehive.jllm.golden.ProgramIdentity;
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
 * The warp-shuffle reduction of the packed-integer {@code Q4_0} matrix-vector.
 *
 * <p>{@link TransformerComputeKernelsQ4_0#matrixVectorGenericQ4_0DP4A} differs from {@link
 * TransformerComputeKernelsQ4_0#matrixVectorGenericQ4_0DP4A} in its reduction tail alone, so these
 * cases are about the tail: that every lane's partial reaches the output, that the four warps are
 * all combined rather than three of them dropped, and that the lane mapping is the one the shuffle
 * widths assume.
 *
 * <p>Dropping warps is the failure this file is built to catch, and it does not announce itself: a
 * kernel that keeps only warp zero still returns plausible finite numbers. The exact integer cases
 * pin it — with unit block scales the answer is a specific integer, and a missing warp changes it.
 *
 * <p>Reduction order differs from the tree's, so sums of block products may round differently. The
 * exact cases are chosen to have no rounding at all; the approximate ones are bounded relatively.
 */
// @formatter:on
public class Q4_0Dp4aReductionAccelTest {

    private static final int LOCAL = 128;
    private static final int QK = 32;
    private static final int BLOCK_BYTES = 18;

    /** Reduction lengths that matter: fewer blocks than lanes, one round, and the 27B's width. */
    private static final int[] LENGTHS = {128, 2048, 4096, 5120};

    private static final int D = 32;

    // ---------------------------------------------------------------- fixtures

    /** Weights whose block scale is exactly one, so the integer dot product survives to the end. */
    private static byte[] unitScaleWeights(int n, long seed) {
        Random random = new Random(seed);
        int blocks = D * (n / QK);
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

    /**
     * Scales that differ per block by four orders of magnitude, alternate sign, and are exactly
     * zero every eleventh block.
     */
    private static byte[] nonuniformScaleWeights(int n, long seed) {
        Random random = new Random(seed);
        int blocks = D * (n / QK);
        byte[] raw = new byte[blocks * BLOCK_BYTES];
        for (int b = 0; b < blocks; b++) {
            int base = b * BLOCK_BYTES;
            short bits;
            if (b % 11 == 0) {
                bits = 0; // a zero block scale
            } else {
                // exponents from 2^-12 to 2^3, alternating sign
                float scale = (float) Math.pow(2.0, -12 + (b % 16)) * (1.0f + (b % 5) * 0.13f);
                if (b % 3 == 0) {
                    scale = -scale;
                }
                bits = new HalfFloat(scale).getHalfFloatValue();
            }
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) ((bits >> 8) & 0xFF);
            for (int i = 0; i < 16; i++) {
                raw[base + 2 + i] = (byte) random.nextInt(256);
            }
        }
        return raw;
    }

    private static int nibble(byte[] raw, int n, int row, int element) {
        int block = element / QK;
        int within = element - block * QK;
        int base = (row * (n / QK) + block) * BLOCK_BYTES;
        int packed = raw[base + 2 + (within & 15)] & 0xFF;
        return within < 16 ? (packed & 0xF) : ((packed >> 4) & 0xF);
    }

    private static float blockScale(byte[] raw, int n, int row, int block) {
        int base = (row * (n / QK) + block) * BLOCK_BYTES;
        int bits = (raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8);
        return new HalfFloat((short) bits).getFloat32();
    }

    /** The activation quantization the kernel chain performs, repeated on the host. */
    private static final class Quantized {
        final int[] quants;
        final float[] scales;
        final int[] sums;

        Quantized(float[] x, int n) {
            quants = new int[n / 4];
            scales = new float[n / QK];
            sums = new int[n / QK];
            for (int block = 0; block < n / QK; block++) {
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

    /** The same-quantization reference: the dot product the kernel must compute, in doubles. */
    private static double[] reference(float[] host, byte[] raw, int n) {
        Quantized q = new Quantized(host, n);
        double[] expected = new double[D];
        for (int row = 0; row < D; row++) {
            double total = 0;
            for (int block = 0; block < n / QK; block++) {
                int dot = 0;
                for (int i = 0; i < QK; i++) {
                    int packed = q.quants[block * (QK / 4) + i / 4];
                    int qx = (byte) ((packed >> ((i % 4) * 8)) & 0xFF);
                    dot += (nibble(raw, n, row, block * QK + i) - 8) * qx;
                }
                total += (double) blockScale(raw, n, row, block) * q.scales[block] * dot;
            }
            expected[row] = total;
        }
        return expected;
    }

    // ---------------------------------------------------------------- device

    private static float[] run(String name, float[] host, byte[] raw, int n) throws Exception {
        return run(name, host, raw, n, null);
    }

    private static float[] run(
            String name, float[] host, byte[] raw, int n, ProgramIdentity.SourceRecorder recorder)
            throws Exception {
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
        FloatArray out = new FloatArray(D);
        // Poisoned, so a row nothing writes is visible rather than plausible.
        out.init(Float.NaN);
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }

        TaskGraph graph =
                new TaskGraph(name)
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
                                n,
                                D,
                                LOCAL);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D quantizeWorker = new WorkerGrid1D(n);
        quantizeWorker.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid(name + ".quantize", quantizeWorker);
        WorkerGrid1D matvecWorker = new WorkerGrid1D(D * LOCAL);
        matvecWorker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid(name + ".matvec", matvecWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            if (recorder != null) {
                plan.withPrintKernel();
            }
            plan.withGridScheduler(scheduler).execute();
        }

        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    private static float[] reduce(float[] host, byte[] raw, int n) throws Exception {
        return run("reduce", host, raw, n);
    }

    // ---------------------------------------------------------------- cases

    /**
     * Every warp's contribution reaches the output, exactly.
     *
     * <p>Unit block scales and integer activations, so the answer is an integer. At 5120 the row
     * spans 160 blocks over 128 lanes and therefore all four warps; keeping only warp zero, or
     * combining the warps without the barrier, changes the integer.
     */
    @Test
    public void everyWarpsContributionReachesTheOutput() throws Exception {
        for (int n : LENGTHS) {
            byte[] raw = unitScaleWeights(n, 20260911L + n);
            Random random = new Random(7L);
            float[] host = new float[n];
            for (int block = 0; block < n / QK; block++) {
                for (int i = 0; i < QK; i++) {
                    host[block * QK + i] = random.nextInt(255) - 127;
                }
                host[block * QK] = 127;
                host[block * QK + 1] = -127;
            }

            float[] got = reduce(host, raw, n);

            for (int row = 0; row < D; row++) {
                long expected = 0;
                for (int i = 0; i < n; i++) {
                    expected += (long) (nibble(raw, n, row, i) - 8) * (long) host[i];
                }
                assertEquals("n " + n + " row " + row, (float) expected, got[row], 0.0f);
            }
        }
    }

    /**
     * Cancellation: rows whose terms are large and whose total is small.
     *
     * <p>Still exact — the terms are integers — so a reduction that loses or double-counts a warp
     * shows up against a total that is far smaller than any single warp's partial.
     */
    @Test
    public void largeTermsThatCancelStillLandOnTheirInteger() throws Exception {
        int n = 4096;
        byte[] raw = unitScaleWeights(n, 31337L);
        float[] host = new float[n];
        for (int block = 0; block < n / QK; block++) {
            for (int i = 0; i < QK; i++) {
                // Alternating ±127: each block's terms are as large as the format allows, and
                // adjacent blocks cancel each other.
                host[block * QK + i] = ((block + i) % 2 == 0) ? 127 : -127;
            }
        }

        float[] got = reduce(host, raw, n);

        for (int row = 0; row < D; row++) {
            long expected = 0;
            for (int i = 0; i < n; i++) {
                expected += (long) (nibble(raw, n, row, i) - 8) * (long) host[i];
            }
            assertEquals("row " + row, (float) expected, got[row], 0.0f);
        }
    }

    /** Zero activation blocks and zero weight-block scales contribute nothing, and nothing NaN. */
    @Test
    public void zeroBlocksContributeNothing() throws Exception {
        int n = 5120;
        byte[] raw = nonuniformScaleWeights(n, 555L);
        float[] host = new float[n];
        for (int block = 0; block < n / QK; block++) {
            boolean zeroBlock = block % 7 == 0;
            for (int i = 0; i < QK; i++) {
                host[block * QK + i] =
                        zeroBlock ? 0.0f : (float) Math.sin(0.019 * (block * QK + i));
            }
        }

        float[] got = reduce(host, raw, n);
        double[] expected = reference(host, raw, n);

        double largest = 0;
        double worst = 0;
        for (int row = 0; row < D; row++) {
            assertTrue("row " + row + " is " + got[row], Float.isFinite(got[row]));
            largest = Math.max(largest, Math.abs(expected[row]));
            worst = Math.max(worst, Math.abs(expected[row] - got[row]));
        }
        assertTrue("worst " + worst + " against largest |out| " + largest, worst <= 1e-4 * largest);
    }

    /** Mixed signs and scales spanning four orders of magnitude, at every reduction length. */
    @Test
    public void itMatchesTheSameQuantizationReferenceAtEveryLength() throws Exception {
        for (int n : LENGTHS) {
            byte[] raw = nonuniformScaleWeights(n, 4242L + n);
            float[] host = new float[n];
            for (int i = 0; i < n; i++) {
                host[i] = (float) Math.sin(0.017 * i) * (1.0f + (i % 7) * 0.25f);
                if (i % 5 == 0) {
                    host[i] = -host[i] * 3.0f;
                }
            }

            float[] got = reduce(host, raw, n);
            double[] expected = reference(host, raw, n);

            double largest = 0;
            double worst = 0;
            for (int row = 0; row < D; row++) {
                assertTrue("n " + n + " row " + row + " is " + got[row], Float.isFinite(got[row]));
                largest = Math.max(largest, Math.abs(expected[row]));
                worst = Math.max(worst, Math.abs(expected[row] - got[row]));
            }
            System.out.printf(
                    "[DP4A warp] n %d: worst %.6g, largest |out| %.6g%n", n, worst, largest);
            assertTrue(
                    "n " + n + " worst " + worst + " against largest |out| " + largest,
                    worst <= 1e-4 * largest);
        }
    }

    // @formatter:off
    /**
     * The generated code is the reduction this kernel claims to be.
     *
     * <p>What the source must show, and what a passing numerical test cannot: five shuffle
     * instructions rather than a shared-memory tree, and <b>one</b> workgroup barrier — the tree at
     * 128 lanes emits seven. Both are counted against the matvec module alone, which is the one
     * kernel source that mentions the shuffle.
     */
    // @formatter:on
    @Test
    public void theGeneratedCodeShufflesAndSynchronizesOnce() throws Exception {
        assumeTrue(
                "[SKIP] -Dtornado.print.kernel.dir sends the dump to a file, where the recorder"
                        + " cannot see it",
                !ProgramIdentity.kernelDumpRedirectedToFile());

        int n = 5120;
        byte[] raw = nonuniformScaleWeights(n, 1234L);
        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            host[i] = (float) Math.sin(0.011 * i);
        }

        ProgramIdentity.SourceRecorder recorder = ProgramIdentity.SourceRecorder.install();
        List<String> sources;
        try {
            run("codegen", host, raw, n, recorder);
            sources = recorder.sources();
        } finally {
            recorder.uninstall();
        }

        assertTrue("no kernel source was recorded", !sources.isEmpty());
        String matvec = null;
        for (String source : sources) {
            if (source.contains("shfl") || source.contains("sub_group_shuffle")) {
                matvec = source;
            }
        }
        assertTrue(
                "no recorded kernel contains a shuffle; recorded " + sources.size() + " sources",
                matvec != null);

        int shuffles = count(matvec, "shfl");
        int barriers = count(matvec, "__syncthreads") + count(matvec, "barrier(");
        System.out.printf(
                "[DP4A warp] generated code: %d shuffles, %d barriers%n", shuffles, barriers);
        assertEquals("one shuffle per butterfly step", 5, shuffles);
        assertEquals("one barrier, between the warp writes and the combine", 1, barriers);
    }

    private static int count(String text, String needle) {
        int seen = 0;
        int at = text.indexOf(needle);
        while (at >= 0) {
            seen++;
            at = text.indexOf(needle, at + needle.length());
        }
        return seen;
    }

    // ------------------------------------------------- the residual and fused forms

    /** Runs one of the residual kernels over a pre-filled destination. */
    private static float[] runResidual(
            String name, float[] host, byte[] raw, int n, float[] destination) throws Exception {
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
        FloatArray hb = new FloatArray(D);
        for (int i = 0; i < D; i++) {
            hb.set(i, destination[i]);
        }
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }

        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, quants, scales, sums, hb)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                x,
                                quants,
                                scales,
                                sums);
        graph.task(
                "matvec",
                TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0DP4A,
                new KernelContext(),
                quants,
                scales,
                sums,
                hb,
                w,
                n,
                D,
                LOCAL);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D quantizeWorker = new WorkerGrid1D(n);
        quantizeWorker.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid(name + ".quantize", quantizeWorker);
        WorkerGrid1D matvecWorker = new WorkerGrid1D(D * LOCAL);
        matvecWorker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid(name + ".matvec", matvecWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = hb.get(i);
        }
        return result;
    }

    /** Runs one of the fused gate/up kernels. */
    private static float[] runFused(String name, float[] host, byte[] gateRaw, byte[] upRaw, int n)
            throws Exception {
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
        FloatArray hb = new FloatArray(D);
        hb.init(Float.NaN);
        ByteArray w1 = new ByteArray(gateRaw.length);
        ByteArray w3 = new ByteArray(upRaw.length);
        for (int i = 0; i < gateRaw.length; i++) {
            w1.set(i, gateRaw[i]);
            w3.set(i, upRaw[i]);
        }

        TaskGraph graph =
                new TaskGraph(name)
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
                                sums);
        graph.task(
                "fused",
                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A,
                new KernelContext(),
                quants,
                scales,
                sums,
                hb,
                w1,
                w3,
                n,
                D,
                LOCAL);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, hb);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D quantizeWorker = new WorkerGrid1D(n);
        quantizeWorker.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid(name + ".quantize", quantizeWorker);
        WorkerGrid1D fusedWorker = new WorkerGrid1D(D * LOCAL);
        fusedWorker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid(name + ".fused", fusedWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            result[i] = hb.get(i);
        }
        return result;
    }

    /**
     * The residual form: every warp counted, and the destination accumulated exactly once.
     *
     * <p>Unit scales and integer activations again, so the dot product is exact; the destination
     * starts at a value far larger than the dot product, which is what makes a second accumulate —
     * one per warp, say — impossible to miss.
     */
    @Test
    public void theResidualFormAddsEveryWarpOnceToItsDestination() throws Exception {
        int n = 5120;
        byte[] raw = unitScaleWeights(n, 6060L);
        Random random = new Random(3L);
        float[] host = new float[n];
        for (int block = 0; block < n / QK; block++) {
            for (int i = 0; i < QK; i++) {
                host[block * QK + i] = random.nextInt(255) - 127;
            }
            host[block * QK] = 127;
            host[block * QK + 1] = -127;
        }
        float[] destination = new float[D];
        for (int row = 0; row < D; row++) {
            destination[row] = 1_000_000.0f + row;
        }

        float[] got = runResidual("residual", host, raw, n, destination);

        for (int row = 0; row < D; row++) {
            long dot = 0;
            for (int i = 0; i < n; i++) {
                dot += (long) (nibble(raw, n, row, i) - 8) * (long) host[i];
            }
            assertEquals("row " + row, destination[row] + (float) dot, got[row], 0.0f);
        }
    }

    /**
     * The fused form's gate is reduced across every warp before SiLU.
     *
     * <p>Exact: unit scales, integer activations, and an up matrix of ones-times-scale so the
     * product is {@code silu(gateDot)} times a known integer. The reference applies SiLU to the
     * whole gate sum; a per-warp SiLU, or a gate missing warps, disagrees by orders of magnitude
     * rather than by rounding.
     */
    @Test
    public void theFusedGateIsReducedBeforeSiLU() throws Exception {
        int n = 5120;
        byte[] gateRaw = unitScaleWeights(n, 1111L);
        byte[] upRaw = unitScaleWeights(n, 2222L);
        Random random = new Random(5L);
        float[] host = new float[n];
        for (int block = 0; block < n / QK; block++) {
            for (int i = 0; i < QK; i++) {
                host[block * QK + i] = random.nextInt(255) - 127;
            }
            host[block * QK] = 127;
            host[block * QK + 1] = -127;
        }

        float[] got = runFused("fused", host, gateRaw, upRaw, n);

        for (int row = 0; row < D; row++) {
            double gate = 0;
            double up = 0;
            for (int i = 0; i < n; i++) {
                gate += (double) (nibble(gateRaw, n, row, i) - 8) * host[i];
                up += (double) (nibble(upRaw, n, row, i) - 8) * host[i];
            }
            double silu = gate / (1.0 + Math.exp(-gate));
            double expected = silu * up;
            assertTrue("row " + row + " is " + got[row], Float.isFinite(got[row]));
            assertEquals(
                    "row " + row, expected, got[row], Math.max(1e-3, 1e-5 * Math.abs(expected)));
        }
    }
}
