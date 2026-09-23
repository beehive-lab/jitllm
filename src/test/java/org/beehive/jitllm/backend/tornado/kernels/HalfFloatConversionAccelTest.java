package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
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

/**
 * What a device does when it rounds an FP32 value to FP16 and reads it back.
 *
 * <p>This is not an abstract question. The Llama-shaped FP16 layer graphs write the normalized
 * activation to an {@code HalfFloatArray} before the QKV projection ({@code
 * mapContextWithQuantize}), so every layer of every token passes through this conversion. Qwen3
 * does not — its RMS norm and QKV projection are fused and stay in FP32 — which is why a defect
 * here shows up as "Llama and Granite disagree with the CPU, Qwen3 agrees" and looks like a family
 * problem rather than a conversion one.
 *
 * <p>The reference is {@link Float#floatToFloat16}, which is IEEE 754 round-to-nearest-even. A
 * backend that truncates instead loses up to one ULP per conversion in a consistent direction,
 * which accumulates across layers rather than cancelling.
 */
public class HalfFloatConversionAccelTest {

    private static final int N = 4096;

    /**
     * Reads one half from the block scale's position of each {@code Q4_0} block.
     *
     * <p>The stride is the representation's own 18 bytes, so this reads exactly where a Q4_0 kernel
     * reads its scale, at exactly that alignment.
     */
    public static void readBlockScales(KernelContext context, ByteArray blocks, FloatArray out) {
        int i = context.globalIdx;
        out.set(i, blocks.getHalfFloat(i * 18).getFloat32());
    }

    /**
     * The same read at an arbitrary block stride and field offset.
     *
     * <p>Q4_0's scale is at offset 0 of an 18-byte block, Q4_1's scale and minimum at 0 and 2 of a
     * 20-byte block, and Q5_K's {@code d} and {@code dmin} at 0 and 2 of a 176-byte super-block.
     * All three strides are even and both offsets are even, which is what {@code getHalfFloat}
     * requires; this reads each of those layouts where its kernel reads it.
     */
    public static void readBlockField(
            KernelContext context, ByteArray blocks, FloatArray out, int stride, int offset) {
        int i = context.globalIdx;
        out.set(i, blocks.getHalfFloat(i * stride + offset).getFloat32());
    }

    // @formatter:off
    /**
     * Every finite half encoding, read from a {@code Q4_0} block scale, equals the value the host
     * decodes from the same bits.
     *
     * <p>{@code Qwen35MMAKernels.projectionMMAQ4_0} reads its block scale with {@code
     * ByteArray.getHalfFloat} rather than assembling the half from two bytes by hand. The two
     * routes have to agree on **every** encoding, not on the ones a weight file happens to hold:
     * negative scales, both zeros, and the subnormal range are exactly where a hand-written decoder
     * and a hardware conversion part company. Infinities and NaNs are excluded deliberately — a
     * quantized block scale is neither, and the decoders in this repository say so.
     */
    // @formatter:on
    @Test
    public void everyFiniteHalfEncodingReadsBackAsTheHostDecodesIt() throws Exception {
        assumeTrue("environment absent: no accelerator", acceleratorPresent());

        int encodings = 0;
        for (int bits = 0; bits < 65536; bits++) {
            if (((bits >> 10) & 0x1F) != 0x1F) {
                encodings++;
            }
        }
        ByteArray blocks = new ByteArray(encodings * 18);
        short[] pattern = new short[encodings];
        int at = 0;
        for (int bits = 0; bits < 65536; bits++) {
            if (((bits >> 10) & 0x1F) == 0x1F) {
                continue; // infinities and NaNs
            }
            pattern[at] = (short) bits;
            blocks.set(at * 18, (byte) (bits & 0xFF));
            blocks.set(at * 18 + 1, (byte) ((bits >> 8) & 0xFF));
            at++;
        }
        FloatArray out = new FloatArray(encodings);
        out.init(Float.NaN);

        TaskGraph graph =
                new TaskGraph("scales")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, blocks)
                        .task(
                                "read",
                                HalfFloatConversionAccelTest::readBlockScales,
                                new KernelContext(),
                                blocks,
                                out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid worker = new WorkerGrid1D(encodings);
        worker.setLocalWork(32, 1, 1);
        scheduler.addWorkerGrid("scales.read", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int negatives = 0;
        int subnormals = 0;
        int zeros = 0;
        for (int i = 0; i < encodings; i++) {
            short bits = pattern[i];
            float expected = Float.float16ToFloat(bits);
            float actual = out.get(i);
            assertEquals(
                    "encoding 0x" + Integer.toHexString(bits & 0xFFFF),
                    Float.floatToRawIntBits(expected),
                    Float.floatToRawIntBits(actual));
            if ((bits & 0x8000) != 0) {
                negatives++;
            }
            if (((bits >> 10) & 0x1F) == 0 && (bits & 0x3FF) != 0) {
                subnormals++;
            }
            if ((bits & 0x7FFF) == 0) {
                zeros++;
            }
        }
        System.out.printf(
                "[HALF] %d finite encodings agree bit for bit (%d negative, %d subnormal, %d"
                        + " zeros)%n",
                encodings, negatives, subnormals, zeros);
    }

    // @formatter:off
    /**
     * Every finite half encoding again, at the block strides and field offsets the other quantized
     * MMA projections use.
     *
     * <p>{@code projectionMMAQ4_1} reads a scale and a minimum at offsets 0 and 2 of a 20-byte
     * block; {@code projectionMMAQ5_K} reads {@code d} and {@code dmin} at 0 and 2 of a 176-byte
     * super-block. The conversion is the same one either way — what differs is the address, and an
     * odd stride or offset would be rejected outright — so this asserts the value at each of those
     * addresses rather than assuming Q4_0's case covers them.
     */
    // @formatter:on
    @Test
    public void everyFiniteHalfEncodingReadsBackAtEveryBlockLayout() throws Exception {
        assumeTrue("environment absent: no accelerator", acceleratorPresent());

        int[][] layouts = {{18, 0}, {20, 0}, {20, 2}, {176, 0}, {176, 2}};
        for (int[] layout : layouts) {
            int stride = layout[0];
            int offset = layout[1];
            assertEquals("stride must be even", 0, stride % 2);
            assertEquals("offset must be even", 0, offset % 2);

            int encodings = 0;
            for (int bits = 0; bits < 65536; bits++) {
                if (((bits >> 10) & 0x1F) != 0x1F) {
                    encodings++;
                }
            }
            ByteArray blocks = new ByteArray(encodings * stride);
            blocks.init((byte) 0);
            short[] pattern = new short[encodings];
            int at = 0;
            for (int bits = 0; bits < 65536; bits++) {
                if (((bits >> 10) & 0x1F) == 0x1F) {
                    continue;
                }
                pattern[at] = (short) bits;
                blocks.set(at * stride + offset, (byte) (bits & 0xFF));
                blocks.set(at * stride + offset + 1, (byte) ((bits >> 8) & 0xFF));
                at++;
            }
            FloatArray out = new FloatArray(encodings);
            out.init(Float.NaN);

            TaskGraph graph =
                    new TaskGraph("layout")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, blocks)
                            .task(
                                    "read",
                                    HalfFloatConversionAccelTest::readBlockField,
                                    new KernelContext(),
                                    blocks,
                                    out,
                                    stride,
                                    offset)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            GridScheduler scheduler = new GridScheduler();
            WorkerGrid worker = new WorkerGrid1D(encodings);
            worker.setLocalWork(32, 1, 1);
            scheduler.addWorkerGrid("layout.read", worker);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            for (int i = 0; i < encodings; i++) {
                short bits = pattern[i];
                assertEquals(
                        "stride "
                                + stride
                                + " offset "
                                + offset
                                + " encoding 0x"
                                + Integer.toHexString(bits & 0xFFFF),
                        Float.floatToRawIntBits(Float.float16ToFloat(bits)),
                        Float.floatToRawIntBits(out.get(i)));
            }
            System.out.printf(
                    "[HALF] stride %3d offset %d: %d finite encodings agree bit for bit%n",
                    stride, offset, encodings);
        }
    }

    /** Writes each input to FP16 storage; the read-back is what the comparison sees. */
    public static void roundTrip(KernelContext context, FloatArray in, HalfFloatArray out) {
        int i = context.globalIdx;
        out.set(i, new HalfFloat(in.get(i)));
    }

    @Test
    public void fp32ToFp16RoundingMatchesIeeeRoundToNearestEven() throws Exception {
        assumeTrue("environment absent: no accelerator", acceleratorPresent());

        FloatArray in = new FloatArray(N);
        HalfFloatArray out = new HalfFloatArray(N);
        // Values spanning the range the normalized activations actually occupy, deliberately off
        // the representable grid so rounding direction is observable.
        for (int i = 0; i < N; i++) {
            double t = (i - N / 2.0) / (N / 8.0);
            in.set(i, (float) (t * 1.0000305175781));
        }

        TaskGraph graph =
                new TaskGraph("conv")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                        .task(
                                "roundTrip",
                                HalfFloatConversionAccelTest::roundTrip,
                                new KernelContext(),
                                in,
                                out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(N);
        worker.setLocalWork(32, 1, 1);
        GridScheduler scheduler = new GridScheduler("conv.roundTrip", worker);

        ImmutableTaskGraph immutable = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
            plan.withGridScheduler(scheduler).execute();
        }

        int mismatches = 0;
        int firstMismatch = -1;
        double worstUlp = 0;
        for (int i = 0; i < N; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(in.get(i)));
            float actual = out.get(i).getFloat32();
            if (Float.compare(expected, actual) != 0) {
                if (firstMismatch < 0) {
                    firstMismatch = i;
                }
                mismatches++;
                double ulp = Math.abs(expected - actual) / Math.max(Math.ulp(expected), 1e-30);
                worstUlp = Math.max(worstUlp, ulp);
            }
        }

        if (mismatches > 0) {
            System.out.printf(
                    "backend %s: %d/%d conversions differ from round-to-nearest-even,"
                            + " worst %.2f ULP; first at i=%d in=%.9g expected=%.9g actual=%.9g%n",
                    TornadoDevices.current().id().backend(),
                    mismatches,
                    N,
                    worstUlp,
                    firstMismatch,
                    in.get(firstMismatch),
                    Float.float16ToFloat(Float.floatToFloat16(in.get(firstMismatch))),
                    out.get(firstMismatch).getFloat32());
        }
        assertEquals(
                "FP32->FP16 conversion must round to nearest even, as Float.floatToFloat16 does."
                        + " Every Llama-shaped FP16 layer writes its normalized activation through this"
                        + " conversion, so a backend that rounds differently loses precision once per"
                        + " layer per token, in a consistent direction",
                0,
                mismatches);
    }

    private static boolean acceleratorPresent() {
        try {
            return TornadoDevices.current() != null
                    && !"cpu".equals(TornadoDevices.current().id().backend().toString());
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }
}
