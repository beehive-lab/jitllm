package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Reading four packed nibble bytes as two sixteen-bit words gives the same packed word the
 * byte-by-byte assembly did.
 *
 * <p>{@code getHalfFloatValue()} is used by the fused projection as a <b>bit-preserving load</b>:
 * the bytes it reads are quantized weights, never a number. So the patterns below are chosen to
 * break any implementation that treats them as floating point or lets a signed short sign-extend —
 * every half NaN and infinity encoding, the signalling and quiet mantissa edges, negative zero,
 * subnormals, and bytes with the high bit set in every position.
 *
 * <p>Both {@code Q4_0} alignment classes are covered. A block is eighteen bytes with its quants at
 * {@code base + 2}, so quant runs land on {@code byteIndex % 4 == 2} for even blocks and {@code ==
 * 0} for odd ones; the offsets walked here cross that boundary and the block boundary with it.
 */
// @formatter:on
public class Q4_0PairedWeightLoadAccelTest {

    /** The paired read exactly as the fused projection performs it. */
    public static void packPaired(KernelContext context, ByteArray w, IntArray out, int words) {
        int i = context.globalIdx;
        if (i >= words) {
            return;
        }
        int offset = i * 4;
        out.set(
                i,
                (w.getHalfFloat(offset).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(offset + 2).getHalfFloatValue() & 0xFFFF) << 16));
    }

    /** The byte-by-byte assembly it replaces. */
    public static void packBytes(KernelContext context, ByteArray w, IntArray out, int words) {
        int i = context.globalIdx;
        if (i >= words) {
            return;
        }
        int offset = i * 4;
        int b0 = w.get(offset) & 0xFF;
        int b1 = w.get(offset + 1) & 0xFF;
        int b2 = w.get(offset + 2) & 0xFF;
        int b3 = w.get(offset + 3) & 0xFF;
        out.set(i, b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }

    private static byte[] hostilePattern(int bytes) {
        // Half encodings that are NaN, infinity, negative zero and subnormal, plus high-bit bytes.
        int[] halves = {
            0x7C00, 0xFC00, // +inf, -inf
            0x7E00, 0xFE00, // quiet NaN, negative quiet NaN
            0x7C01, 0xFC01, // signalling NaN edges
            0x7DFF, 0xFDFF, // NaN mantissa edges
            0x8000, 0x0000, // negative zero, zero
            0x0001, 0x8001, // subnormals
            0xFFFF, 0x00FF, 0xFF00, 0x8080, // high-bit bytes in each position
        };
        byte[] raw = new byte[bytes];
        for (int i = 0; i + 1 < bytes; i += 2) {
            int h = halves[(i / 2) % halves.length];
            raw[i] = (byte) (h & 0xFF);
            raw[i + 1] = (byte) ((h >> 8) & 0xFF);
        }
        return raw;
    }

    @Test
    public void thePairedReadPacksTheSameWordAsTheByteAssembly() throws Exception {
        assumeTrue("no tensor-core-capable device", TensorCoreSupport.isTensorCoreCapableBackend());

        // Eighteen-byte blocks, so the run covers both alignment classes and block boundaries.
        int blocks = 64;
        int bytes = blocks * 18;
        int words = bytes / 4;
        byte[] raw = hostilePattern(bytes);

        ByteArray w = new ByteArray(bytes);
        for (int i = 0; i < bytes; i++) {
            w.set(i, raw[i]);
        }
        IntArray paired = new IntArray(words);
        IntArray bytewise = new IntArray(words);
        paired.init(0xDEADBEEF);
        bytewise.init(0xDEADBEEF);

        TaskGraph graph =
                new TaskGraph("pack")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, paired, bytewise)
                        .task(
                                "paired",
                                Q4_0PairedWeightLoadAccelTest::packPaired,
                                new KernelContext(),
                                w,
                                paired,
                                words)
                        .task(
                                "bytes",
                                Q4_0PairedWeightLoadAccelTest::packBytes,
                                new KernelContext(),
                                w,
                                bytewise,
                                words)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, paired, bytewise);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D grid = new WorkerGrid1D(words);
        grid.setLocalWork(32, 1, 1);
        scheduler.addWorkerGrid("pack.paired", grid);
        scheduler.addWorkerGrid("pack.bytes", grid);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int alignedAt2 = 0;
        int alignedAt0 = 0;
        for (int i = 0; i < words; i++) {
            int expected = 0;
            for (int b = 0; b < 4; b++) {
                expected |= (raw[i * 4 + b] & 0xFF) << (8 * b);
            }
            assertEquals("paired word " + i + " at byte " + (i * 4), expected, paired.get(i));
            assertEquals("bytewise word " + i, expected, bytewise.get(i));
            if ((i * 4) % 4 == 0) {
                alignedAt0++;
            }
        }
        // Every quant run of a Q4_0 block: even blocks land at %4 == 2, odd blocks at %4 == 0.
        for (int block = 0; block < blocks; block++) {
            int qs = block * 18 + 2;
            if (qs % 4 == 2) {
                alignedAt2++;
            }
        }
        assertTrue("the layout must exercise the two-byte aligned class", alignedAt2 > 0);
        assertTrue("and the four-byte aligned one", alignedAt0 > 0);
        System.out.printf(
                "[PAIRED] %d words identical; %d Q4_0 runs at 2-byte alignment%n",
                words, alignedAt2);
    }
}
