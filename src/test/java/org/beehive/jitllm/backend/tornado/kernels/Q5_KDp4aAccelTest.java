package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Random;
import org.beehive.jitllm.golden.ProgramIdentity;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
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
 * The packed-integer {@code Q5_K} projection, which is {@code ssm_out}'s representation.
 *
 * <p>Its algebra is not Q4_0's and none of Q6_K's carries over. A Q5_K weight is {@code d * sc(sub)
 * * q - dmin * m(sub)} with {@code q} five bits wide and never recentred, so a sub-block's
 * contribution splits into {@code d*sc} times the integer dot product and {@code dmin*m} times the
 * <b>sum of the activation's quants</b>. The second term is the one this representation adds, it
 * reads the same scratch array Q4_0 multiplies by the constant 8, and getting it wrong produces
 * plausible finite numbers.
 *
 * <p>Three failures this file exists to catch, none of which announces itself: a fifth bit taken
 * from the wrong position of {@code qh} (sixteen quantization steps), the {@code subBlock >= 4}
 * branch of the six-bit scale packing where a value straddles two bytes, and a minimum term applied
 * to the wrong quantity.
 */
// @formatter:on
public class Q5_KDp4aAccelTest {

    private static final int QK_K = 256;
    private static final int BLOCK_BYTES = 176;
    private static final int SCALES_OFFSET = 4;
    private static final int QH_OFFSET = 16;
    private static final int QS_OFFSET = 48;
    private static final int QK = 32;
    private static final int LOCAL = 128;

    /** {@code ssm_out}'s own reduction length in the 27B: 6144 into 5120. */
    private static final int SSM_OUT_N = 6144;

    // ---------------------------------------------------------------- fixtures

    /**
     * Payload bytes that drive the addressing, with scales that stay in a sane range.
     *
     * <p>The payload patterns — all clear, all set, both alternations, and every fourth block left
     * random — are what drive the fifth-bit plane both ways at every position and the six-bit
     * scale/minimum packing across its whole range, including the straddling branch. The scale
     * fields are written last and deliberately are <b>not</b> the fp16 corner cases the decode
     * parity test uses: this test multiplies and accumulates, so an infinite block scale would
     * measure float overflow rather than addressing.
     */
    private static byte[] adversarialWeights(int rows, int n, long seed) {
        int blocks = rows * (n / QK_K);
        byte[] raw = new byte[blocks * BLOCK_BYTES];
        Random random = new Random(seed);
        random.nextBytes(raw);
        byte[] patterns = {0x00, (byte) 0xFF, (byte) 0xAA, 0x55};
        for (int b = 0; b < blocks; b++) {
            int base = b * BLOCK_BYTES;
            if (b % 4 != 3) {
                byte pattern = patterns[b % patterns.length];
                for (int i = 0; i < BLOCK_BYTES; i++) {
                    raw[base + i] = pattern;
                }
            }
            // d and dmin: both signs, four orders of magnitude apart, and zero every ninth block.
            float d = (b % 9 == 0) ? 0.0f : (float) Math.pow(2.0, -14 + (b % 12)) * 0.031f;
            float dmin = (b % 7 == 0) ? 0.0f : (float) Math.pow(2.0, -13 + (b % 10)) * 0.017f;
            if (b % 3 == 0) {
                d = -d;
            }
            if (b % 5 == 0) {
                dmin = -dmin;
            }
            writeHalf(raw, base, d);
            writeHalf(raw, base + 2, dmin);
        }
        return raw;
    }

    private static void writeHalf(byte[] raw, int at, float value) {
        short bits = new HalfFloat(value).getHalfFloatValue();
        raw[at] = (byte) (bits & 0xFF);
        raw[at + 1] = (byte) ((bits >> 8) & 0xFF);
    }

    // ---------------------------------------------------------------- the test's own decoder

    /** This test's independent reading of the five-bit quant at {@code withinBlock}. */
    private static int quant(byte[] raw, int blockBase, int withinBlock) {
        int pairIndex = withinBlock / 64;
        int posInPair = withinBlock - pairIndex * 64;
        int highNibble = posInPair / 32;
        int posInHalf = posInPair - highNibble * 32;
        int qs = raw[blockBase + QS_OFFSET + pairIndex * 32 + posInHalf] & 0xFF;
        int q = (highNibble == 0) ? (qs & 0xF) : ((qs >> 4) & 0xF);
        int qh = raw[blockBase + QH_OFFSET + posInHalf] & 0xFF;
        return q + ((qh >> (pairIndex * 2 + highNibble)) & 1) * 16;
    }

    /** The sub-block's six-bit scale, read independently of the kernel's packing. */
    private static int subScale(byte[] raw, int blockBase, int subBlock) {
        int base = blockBase + SCALES_OFFSET;
        if (subBlock < 4) {
            return raw[base + subBlock] & 63;
        }
        int low = raw[base + subBlock + 4] & 0xFF;
        int high = raw[base + subBlock - 4] & 0xFF;
        return (low & 0xF) | ((high >> 6) << 4);
    }

    /** The sub-block's six-bit minimum. */
    private static int subMin(byte[] raw, int blockBase, int subBlock) {
        int base = blockBase + SCALES_OFFSET;
        if (subBlock < 4) {
            return raw[base + subBlock + 4] & 63;
        }
        int low = raw[base + subBlock + 4] & 0xFF;
        return ((low >> 4) & 0xF) | (((raw[base + subBlock] & 0xFF) >> 6) << 4);
    }

    private static float blockHalf(byte[] raw, int at) {
        int bits = (raw[at] & 0xFF) | ((raw[at + 1] & 0xFF) << 8);
        return new HalfFloat((short) bits).getFloat32();
    }

    /** The activation quantization the kernel chain performs, repeated on the host. */
    private static final class Quantized {
        final float[] scales;
        final int[] sums;
        final int[][] quants;

        Quantized(float[] x, int n) {
            int blocks = n / QK;
            scales = new float[blocks];
            sums = new int[blocks];
            quants = new int[blocks][QK];
            for (int block = 0; block < blocks; block++) {
                int base = block * QK;
                float maxAbs = 0;
                for (int i = 0; i < QK; i++) {
                    maxAbs = Math.max(maxAbs, Math.abs(x[base + i]));
                }
                scales[block] = maxAbs / 127.0f;
                float inverse = maxAbs > 0 ? 127.0f / maxAbs : 0.0f;
                int sum = 0;
                for (int i = 0; i < QK; i++) {
                    float v = x[base + i] * inverse;
                    int q = (int) (v + (v >= 0 ? 0.5f : -0.5f));
                    q = Math.min(127, Math.max(-127, q));
                    quants[block][i] = q;
                    sum += q;
                }
                sums[block] = sum;
            }
        }
    }

    /**
     * The same-quantization reference, in doubles: the dot product the kernel must compute, with
     * the activation quantized exactly as the device quantizes it.
     */
    private static double[] reference(float[] host, byte[] raw, int n, int rows) {
        Quantized q = new Quantized(host, n);
        double[] expected = new double[rows];
        for (int row = 0; row < rows; row++) {
            double total = 0;
            for (int sb = 0; sb < n / QK; sb++) {
                int block = sb / 8;
                int subInBlock = sb - block * 8;
                int blockBase = (row * (n / QK_K) + block) * BLOCK_BYTES;
                double scale =
                        (double) blockHalf(raw, blockBase) * subScale(raw, blockBase, subInBlock);
                double minimum =
                        (double) blockHalf(raw, blockBase + 2) * subMin(raw, blockBase, subInBlock);
                long dot = 0;
                for (int t = 0; t < QK; t++) {
                    dot += (long) quant(raw, blockBase, subInBlock * 32 + t) * q.quants[sb][t];
                }
                total += q.scales[sb] * (scale * dot - minimum * q.sums[sb]);
            }
            expected[row] = total;
        }
        return expected;
    }

    // ---------------------------------------------------------------- device

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    /** Quantizes the activation and runs the packed projection, as a layer chains them. */
    private static float[] run(
            String name,
            float[] host,
            byte[] raw,
            int n,
            int rows,
            float[] destination,
            ProgramIdentity.SourceRecorder recorder)
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
        FloatArray hb = new FloatArray(rows);
        for (int i = 0; i < rows; i++) {
            hb.set(i, destination[i]);
        }
        ByteArray w = toDevice(raw);

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
                                sums)
                        .task(
                                "matvec",
                                TransformerComputeKernelsQ5_K
                                        ::matrixVectorGenericWithResidualQ5_KDP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                hb,
                                w,
                                n,
                                rows,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D quantizeWorker = new WorkerGrid1D(n);
        quantizeWorker.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid(name + ".quantize", quantizeWorker);
        WorkerGrid1D matvecWorker = new WorkerGrid1D(rows * LOCAL);
        matvecWorker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid(name + ".matvec", matvecWorker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            if (recorder != null) {
                plan.withPrintKernel();
            }
            plan.withGridScheduler(scheduler).execute();
        }

        float[] result = new float[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = hb.get(i);
        }
        return result;
    }

    private static float[] run(String name, float[] host, byte[] raw, int n, int rows)
            throws Exception {
        return run(name, host, raw, n, rows, new float[rows], null);
    }

    // ---------------------------------------------------------------- cases

    /**
     * One-hot: every weight of a super-block, decoded by the kernel, against the host tensor.
     *
     * <p>An activation that is 1.0 at one element and zero elsewhere quantizes to 127 in that
     * element's block and to a zero scale everywhere else, so the projection returns that one
     * weight — sub-block scale, minimum and fifth bit included. Running it for all 256 positions
     * covers both nibble halves, all eight sub-blocks (so both branches of the six-bit scale
     * packing), and every bit position of {@code qh}; running it over eight rows does that against
     * eight different super-blocks.
     */
    @Test
    public void oneHotDecodingMatchesTheHostTensor() throws Exception {
        int n = QK_K;
        int rows = 8;
        byte[] raw = adversarialWeights(rows, n, 20260911L);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            Q5_KFloatTensor hostTensor = new Q5_KFloatTensor(rows * n, segment);

            int compared = 0;
            for (int element = 0; element < n; element++) {
                float[] host = new float[n];
                host[element] = 1.0f;
                float[] got = run("oneHot", host, raw, n, rows);
                for (int row = 0; row < rows; row++) {
                    float expected = hostTensor.getFloat(row * n + element);
                    assertTrue(
                            "row " + row + " element " + element + " is " + got[row],
                            Float.isFinite(got[row]));
                    assertEquals(
                            "row " + row + " element " + element,
                            expected,
                            got[row],
                            1e-5f * Math.max(1.0f, Math.abs(expected)));
                    compared++;
                }
            }
            assertEquals(rows * n, compared);
        }
    }

    /**
     * Dense, at {@code ssm_out}'s own width, against a reference that quantizes the same way.
     *
     * <p>Mixed signs on both the block scales and the activations, sub-block scales and minima
     * spanning their whole six-bit range, zero {@code d} and zero {@code dmin} blocks, zero
     * activation blocks, and terms that cancel: the activation alternates sign every element, so
     * each row's total is far smaller than its largest term and a lost or doubled sub-block shows.
     */
    @Test
    public void denseMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        int n = SSM_OUT_N;
        int rows = 32;
        byte[] raw = adversarialWeights(rows, n, 4242L);
        float[] host = new float[n];
        for (int block = 0; block < n / QK; block++) {
            boolean zeroBlock = block % 11 == 0;
            for (int i = 0; i < QK; i++) {
                int index = block * QK + i;
                float value = (float) Math.sin(0.017 * index) * (1.0f + (index % 7) * 0.25f);
                host[index] = zeroBlock ? 0.0f : ((i % 2 == 0) ? value : -value);
            }
        }

        float[] got = run("dense", host, raw, n, rows);
        double[] expected = reference(host, raw, n, rows);

        double largest = 0;
        double worst = 0;
        for (int row = 0; row < rows; row++) {
            assertTrue("row " + row + " is " + got[row], Float.isFinite(got[row]));
            largest = Math.max(largest, Math.abs(expected[row]));
            worst = Math.max(worst, Math.abs(expected[row] - got[row]));
        }
        System.out.printf(
                "[Q5_K DP4A] dense n=%d: worst %.6g, largest |out| %.6g%n", n, worst, largest);
        assertTrue(
                "worst deviation " + worst + " against largest output " + largest,
                worst <= 1e-4 * largest);
    }

    /** A zero activation contributes nothing at all — not even through the minimum term. */
    @Test
    public void aZeroActivationContributesNothing() throws Exception {
        int n = SSM_OUT_N;
        int rows = 16;
        byte[] raw = adversarialWeights(rows, n, 777L);
        float[] host = new float[n];
        float[] destination = new float[rows];
        for (int row = 0; row < rows; row++) {
            destination[row] = 3.5f + row;
        }

        float[] got = run("zero", host, raw, n, rows, destination, null);

        for (int row = 0; row < rows; row++) {
            assertEquals("row " + row, destination[row], got[row], 0.0f);
        }
    }

    /** The residual is added once per row, by one thread, after the whole reduction. */
    @Test
    public void theResidualIsAddedOnce() throws Exception {
        int n = SSM_OUT_N;
        int rows = 32;
        byte[] raw = adversarialWeights(rows, n, 5150L);
        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            host[i] = (float) Math.cos(0.013 * i);
        }
        float[] destination = new float[rows];
        for (int row = 0; row < rows; row++) {
            destination[row] = 1000.0f * (row % 2 == 0 ? 1 : -1);
        }

        float[] withResidual = run("residual", host, raw, n, rows, destination, null);
        float[] withoutResidual = run("plain", host, raw, n, rows);

        for (int row = 0; row < rows; row++) {
            assertEquals(
                    "row " + row,
                    destination[row] + withoutResidual[row],
                    withResidual[row],
                    1e-3f * Math.max(1.0f, Math.abs(destination[row])));
        }
    }

    /**
     * The generated code issues the packed instruction, rather than falling back to its Java body.
     */
    @Test
    public void theGeneratedCodeUsesDp4a() throws Exception {
        assumeTrue(
                "[SKIP] -Dtornado.print.kernel.dir sends the dump to a file",
                !ProgramIdentity.kernelDumpRedirectedToFile());

        int n = QK_K * 4;
        int rows = 8;
        byte[] raw = adversarialWeights(rows, n, 31337L);
        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            host[i] = (float) Math.sin(0.05 * i);
        }

        ProgramIdentity.SourceRecorder recorder = ProgramIdentity.SourceRecorder.install();
        List<String> sources;
        try {
            run("codegen", host, raw, n, rows, new float[rows], recorder);
            sources = recorder.sources();
        } finally {
            recorder.uninstall();
        }

        String matvec = null;
        for (String source : sources) {
            if (source.contains("dp4a")) {
                matvec = source;
            }
        }
        assertTrue(
                "no recorded kernel issues dp4a; recorded " + sources.size() + " sources",
                matvec != null);
        int shuffles = 0;
        int at = matvec.indexOf("shfl");
        while (at >= 0) {
            shuffles++;
            at = matvec.indexOf("shfl", at + 4);
        }
        System.out.printf("[Q5_K DP4A] generated code: dp4a present, %d shuffles%n", shuffles);
        assertEquals("one shuffle per butterfly step", 5, shuffles);
    }
}
