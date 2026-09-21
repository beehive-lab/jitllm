package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The one-block-lookahead {@code Q4_0} tensor-core projection against the retained paired-load
 * kernel it was copied from, on the device, over identical input bytes.
 *
 * <p>A bit-parity test, unlike {@link Qwen35MMAProjectionAccelTest}: both kernels decode the same
 * nibbles to the same halves, feed the same {@code m16n8k16} instruction in the same K order and
 * accumulate in the same FP32 registers, so every output must carry the same raw bits. A tolerance
 * would hide exactly the defects this exists for — a word's bytes swapped, a nibble taken from the
 * wrong byte, a column left unwritten.
 *
 * <p>Both outputs are poisoned with NaN before the run, so a column that the candidate never stores
 * fails as a non-finite value rather than agreeing with a stale zero, and every output of both
 * kernels is required to be finite: the fixtures are bounded, so a non-finite result is a decode or
 * addressing error, not a numerical one.
 */
// @formatter:on
public class Qwen35MMAQ4_0PrefetchParityAccelTest {

    private static final int BLOCK_BYTES = 18;

    /**
     * Random Q4_0 bytes with a finite scale of either sign per block: the nibbles come from the
     * generator and cover every value, the scale's magnitude stays near 0.06 and its sign
     * alternates by block.
     */
    private static byte[] randomWeights(int n, int k, long seed) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * BLOCK_BYTES;
            int bits = 0x2C00 | (b & 0xFF);
            if ((b & 1) == 1) {
                bits |= 0x8000;
            }
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    /** Every nibble value in both halves of the block, against both signs of scale. */
    private static byte[] everyNibbleWeights(int n, int k) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        for (int c = 0; c < n; c++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int base = (c * blocksPerRow + blk) * BLOCK_BYTES;
                int bits = 0x2C00 | ((c + blk) & 0xFF);
                if (((c + blk) & 1) == 1) {
                    bits |= 0x8000;
                }
                raw[base] = (byte) (bits & 0xFF);
                raw[base + 1] = (byte) (bits >> 8);
                for (int b = 0; b < 16; b++) {
                    int lo = (b + c + blk) & 0xF;
                    int hi = (15 - b + c) & 0xF;
                    raw[base + 2 + b] = (byte) ((hi << 4) | lo);
                }
            }
        }
        return raw;
    }

    private static HalfFloatArray activations(int m, int k, long seed) {
        Random rng = new Random(seed);
        HalfFloatArray a = new HalfFloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
        }
        return a;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static WorkerGrid grid(int m, int n, int columnsPerWarp) {
        WorkerGrid worker =
                new WorkerGrid1D(
                        (m / Qwen35MMAKernels.BM) * (n / columnsPerWarp) * Qwen35MMAKernels.LOCAL);
        worker.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
        return worker;
    }

    /** Runs both kernels over the same device bytes and asserts raw-bit equality everywhere. */
    private static void assertBitParity(String what, int m, int n, int k, byte[] raw, long seed)
            throws Exception {
        HalfFloatArray a = activations(m, k, seed);
        ByteArray w = toDevice(raw);
        FloatArray original = new FloatArray(m * n);
        FloatArray candidate = new FloatArray(m * n);
        original.init(Float.NaN);
        candidate.init(Float.NaN);

        TaskGraph graph =
                new TaskGraph("parity")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, a, w, original, candidate)
                        .task(
                                "original",
                                Qwen35ReferenceKernels::projectionMMAQ4_0,
                                new KernelContext(),
                                a,
                                w,
                                original,
                                m,
                                n,
                                k)
                        .task(
                                "prefetch",
                                Qwen35MMAKernels::projectionMMAQ4_0Prefetch,
                                new KernelContext(),
                                a,
                                w,
                                candidate,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, original, candidate);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("parity.original", grid(m, n, Qwen35MMAKernels.BN));
        scheduler.addWorkerGrid("parity.prefetch", grid(m, n, Qwen35MMAKernels.BN));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int mismatches = 0;
        String first = null;
        for (int i = 0; i < m * n; i++) {
            float o = original.get(i);
            float c = candidate.get(i);
            assertTrue(
                    what + ": original is not finite at row " + (i / n) + " col " + (i % n),
                    Float.isFinite(o));
            assertTrue(
                    what + ": candidate is not finite at row " + (i / n) + " col " + (i % n),
                    Float.isFinite(c));
            if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(c)) {
                if (first == null) {
                    first =
                            "row "
                                    + (i / n)
                                    + " col "
                                    + (i % n)
                                    + ": original "
                                    + o
                                    + " candidate "
                                    + c;
                }
                mismatches++;
            }
        }
        assertEquals(
                what + ": " + mismatches + " outputs differ, first at " + first, 0, mismatches);
    }

    /** Two row tiles and many column groups, over every nibble and both scale signs. */
    @Test
    public void everyNibbleAndBothSignsAgreeBitForBit() throws Exception {
        assertBitParity("nibbles", 32, 256, 512, everyNibbleWeights(256, 512), 1L);
    }

    /**
     * One block (no lookahead at all), two (one lookahead, then the guarded last round), odd
     * counts, and the production depths.
     */
    @Test
    public void everyBlockCountAgreesBitForBit() throws Exception {
        for (int k : new int[] {32, 64, 96, 160, 1024, 5120, 6144, 17408}) {
            assertBitParity("k=" + k, 32, 256, k, randomWeights(256, k, 11L + k), 200L + k);
        }
    }

    /** One, two and four row tiles over the same bytes. */
    @Test
    public void everyRowTileCountAgreesBitForBit() throws Exception {
        byte[] raw = randomWeights(512, 1024, 7L);
        for (int m : new int[] {16, 32, 64}) {
            assertBitParity("m=" + m, m, 512, 1024, raw, 100L + m);
        }
    }

    /** The production shapes this model dispatches: their N and K, at the production width. */
    @Test
    public void theProductionShapesAgreeBitForBit() throws Exception {
        int[][] shapes = {
            {17408, 5120}, // ffn_gate, ffn_up
            {5120, 17408}, // ffn_down
            {5120, 6144}, // attn_output
            {10240, 5120}, // ssm_qkv
            {6144, 5120}, // ssm_gate
            {12288, 5120}, // attn_q + gate
            {1024, 5120}, // attn_k, attn_v
        };
        for (int[] shape : shapes) {
            int n = shape[0];
            int k = shape[1];
            assertBitParity("n=" + n + " k=" + k, 32, n, k, randomWeights(n, k, 1000L + n + k), n);
        }
    }
}
