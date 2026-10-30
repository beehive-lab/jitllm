package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The grouped int8 tensor-core expert GEMMs over experts in their own representations ({@code
 * Qwen35MoeBatchKernels.groupedGateUp*} and {@code groupedDown*} for Q4_0, Q4_1, Q4_K, Q5_K and
 * Q6_K), against the host tensors for the same bytes and the device's own quantized activations.
 *
 * <p>Three tiles of different experts and heights, the last partial, over gathered tokens: a wrong
 * staging decode, a lost minimum, a sixteen-wide scale applied to the wrong half, or a wrong expert
 * row shows up as a wrong output. Needs int8 tensor cores.
 */
public class Qwen35MoeNativeGroupedGemmAccelTest {

    private static final int DIM = 512;

    private static final int HIDDEN = 256;

    private static final int EXPERTS = 5;

    /** (expert, first sorted row, rows) per tile. */
    private static final int[][] TILES = {{3, 0, 64}, {0, 64, 17}, {4, 81, 40}};

    private static final int ROWS = 121;

    private interface HostTensor {
        FloatTensor of(int size, MemorySegment segment);
    }

    private static HostTensor host(GGMLType type) {
        return switch (type) {
            case Q4_0 -> Q4_0FloatTensor::new;
            case Q4_1 -> Q4_1FloatTensor::new;
            case Q4_K -> Q4_KFloatTensor::new;
            case Q5_K -> Q5_KFloatTensor::new;
            case Q6_K -> Q6_KFloatTensor::new;
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /**
     * Random blocks with finite, modest half-precision scales and minima. A K-quant's are
     * super-block scales, which multiply a six-bit sub-block scale, so they are kept 2^-4 smaller:
     * otherwise the weights reach 30, and a gate that is the small difference of the scale term and
     * the minimum term carries a float rounding the up projection then multiplies by hundreds.
     */
    private static byte[] randomWeights(GGMLType type, int rows, int cols, long seed) {
        int blocks = rows * (cols / type.getBlockSize());
        byte[] raw = new byte[blocks * type.getTypeSize()];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        int[] halves =
                switch (type) {
                    case Q4_0 -> new int[] {0};
                    case Q4_1, Q4_K, Q5_K -> new int[] {0, 2};
                    case Q6_K -> new int[] {208};
                    default -> throw new IllegalArgumentException(type.toString());
                };
        for (int b = 0; b < blocks; b++) {
            for (int offset : halves) {
                int exponent = type == GGMLType.Q4_K || type == GGMLType.Q5_K ? 0x1000 : 0x2000;
                int bits = exponent | rng.nextInt(0x400) | (rng.nextBoolean() ? 0x8000 : 0);
                raw[b * type.getTypeSize() + offset] = (byte) (bits & 0xFF);
                raw[b * type.getTypeSize() + offset + 1] = (byte) (bits >> 8);
            }
        }
        return raw;
    }

    private static ByteArray device(byte[] raw) {
        ByteArray a = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            a.set(i, raw[i]);
        }
        return a;
    }

    private static FloatArray gaussian(int n, long seed) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, (float) rng.nextGaussian());
        }
        return a;
    }

    private static float[] decode(GGMLType type, byte[] raw, int size) {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor tensor = host(type).of(size, segment);
            float[] out = new float[size];
            for (int i = 0; i < size; i++) {
                out[i] = tensor.getFloat(i);
            }
            return out;
        }
    }

    /** {@code w[row] . x[xRow]} over the dequantized weights and the device-quantized x. */
    private static double dot(float[] w, int row, int cols, ByteArray q, FloatArray s, int xRow) {
        double sum = 0;
        for (int c = 0; c < cols; c++) {
            sum +=
                    w[row * cols + c]
                            * (double) q.get(xRow * cols + c)
                            * s.get((xRow * cols + c) / 32);
        }
        return sum;
    }

    private static WorkerGrid lanes(int n, int local) {
        WorkerGrid g = new WorkerGrid1D((n + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static void check(GGMLType type) throws Exception {
        byte[] gateRaw = randomWeights(type, EXPERTS * HIDDEN, DIM, 11L + type.ordinal());
        byte[] upRaw = randomWeights(type, EXPERTS * HIDDEN, DIM, 23L + type.ordinal());
        byte[] downRaw = randomWeights(type, EXPERTS * DIM, HIDDEN, 37L + type.ordinal());
        int tokens = 48;
        FloatArray x = gaussian(tokens * DIM, 1);
        FloatArray h = gaussian(ROWS * HIDDEN, 2);
        IntArray sortedToken = new IntArray(ROWS);
        for (int r = 0; r < ROWS; r++) {
            sortedToken.set(r, (r * 7 + 3) % tokens);
        }
        IntArray tiles = new IntArray(1 + 3 * TILES.length);
        tiles.set(0, TILES.length);
        for (int i = 0; i < TILES.length; i++) {
            for (int f = 0; f < 3; f++) {
                tiles.set(1 + 3 * i + f, TILES[i][f]);
            }
        }
        ByteArray xq = new ByteArray(tokens * DIM);
        FloatArray xs = new FloatArray(tokens * DIM / 32);
        ByteArray hq = new ByteArray(ROWS * HIDDEN);
        FloatArray hs = new FloatArray(ROWS * HIDDEN / 32);
        FloatArray hidden = new FloatArray(ROWS * HIDDEN);
        FloatArray out = new FloatArray(ROWS * DIM);
        ByteArray gate = device(gateRaw), up = device(upRaw), down = device(downRaw);

        TaskGraph g =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                x,
                                h,
                                sortedToken,
                                tiles,
                                xq,
                                xs,
                                hq,
                                hs,
                                hidden,
                                out,
                                gate,
                                up,
                                down)
                        .task(
                                "qx",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                x,
                                xq,
                                xs,
                                DIM)
                        .task(
                                "qh",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                h,
                                hq,
                                hs,
                                HIDDEN);
        switch (type) {
            case Q4_0 -> {
                g.task(
                        "gu",
                        Qwen35MoeBatchKernels::groupedGateUpQ4_0,
                        new KernelContext(),
                        xq,
                        xs,
                        sortedToken,
                        tiles,
                        gate,
                        up,
                        hidden,
                        DIM,
                        HIDDEN);
                g.task(
                        "d",
                        Qwen35MoeBatchKernels::groupedDownQ4_0,
                        new KernelContext(),
                        hq,
                        hs,
                        tiles,
                        down,
                        out,
                        DIM,
                        HIDDEN);
            }
            case Q4_1 -> {
                g.task(
                        "gu",
                        Qwen35MoeBatchKernels::groupedGateUpQ4_1,
                        new KernelContext(),
                        xq,
                        xs,
                        sortedToken,
                        tiles,
                        gate,
                        up,
                        hidden,
                        DIM,
                        HIDDEN);
                g.task(
                        "d",
                        Qwen35MoeBatchKernels::groupedDownQ4_1,
                        new KernelContext(),
                        hq,
                        hs,
                        tiles,
                        down,
                        out,
                        DIM,
                        HIDDEN);
            }
            case Q4_K -> {
                g.task(
                        "gu",
                        Qwen35MoeBatchKernels::groupedGateUpQ4_K,
                        new KernelContext(),
                        xq,
                        xs,
                        sortedToken,
                        tiles,
                        gate,
                        up,
                        hidden,
                        DIM,
                        HIDDEN);
                g.task(
                        "d",
                        Qwen35MoeBatchKernels::groupedDownQ4_K,
                        new KernelContext(),
                        hq,
                        hs,
                        tiles,
                        down,
                        out,
                        DIM,
                        HIDDEN);
            }
            case Q5_K -> {
                g.task(
                        "gu",
                        Qwen35MoeBatchKernels::groupedGateUpQ5_K,
                        new KernelContext(),
                        xq,
                        xs,
                        sortedToken,
                        tiles,
                        gate,
                        up,
                        hidden,
                        DIM,
                        HIDDEN);
                g.task(
                        "d",
                        Qwen35MoeBatchKernels::groupedDownQ5_K,
                        new KernelContext(),
                        hq,
                        hs,
                        tiles,
                        down,
                        out,
                        DIM,
                        HIDDEN);
            }
            default -> {
                g.task(
                        "gu",
                        Qwen35MoeBatchKernels::groupedGateUpQ6_K,
                        new KernelContext(),
                        xq,
                        xs,
                        sortedToken,
                        tiles,
                        gate,
                        up,
                        hidden,
                        DIM,
                        HIDDEN);
                g.task(
                        "d",
                        Qwen35MoeBatchKernels::groupedDownQ6_K,
                        new KernelContext(),
                        hq,
                        hs,
                        tiles,
                        down,
                        out,
                        DIM,
                        HIDDEN);
            }
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, xq, xs, hq, hs, hidden, out);

        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("moe.qx", lanes(tokens * DIM, 128));
        s.addWorkerGrid("moe.qh", lanes(ROWS * HIDDEN, 128));
        WorkerGrid gu = new WorkerGrid2D(HIDDEN / 128 * 256, TILES.length + 2);
        gu.setLocalWork(256, 1, 1);
        s.addWorkerGrid("moe.gu", gu);
        WorkerGrid dg = new WorkerGrid2D(DIM / 128 * 256, TILES.length + 2);
        dg.setLocalWork(256, 1, 1);
        s.addWorkerGrid("moe.d", dg);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }

        float[] gateW = decode(type, gateRaw, EXPERTS * HIDDEN * DIM);
        float[] upW = decode(type, upRaw, EXPERTS * HIDDEN * DIM);
        float[] downW = decode(type, downRaw, EXPERTS * DIM * HIDDEN);
        double worstHidden = 0, worstDown = 0;
        for (int[] tile : TILES) {
            int expert = tile[0];
            for (int r = tile[1]; r < tile[1] + tile[2]; r++) {
                int token = sortedToken.get(r);
                for (int c = 0; c < HIDDEN; c++) {
                    double gg = dot(gateW, expert * HIDDEN + c, DIM, xq, xs, token);
                    double uu = dot(upW, expert * HIDDEN + c, DIM, xq, xs, token);
                    double expect = gg / (1 + Math.exp(-gg)) * uu;
                    worstHidden =
                            Math.max(
                                    worstHidden,
                                    Math.abs(expect - hidden.get(r * HIDDEN + c))
                                            / (1 + Math.abs(expect)));
                }
                for (int c = 0; c < DIM; c++) {
                    double expect = dot(downW, expert * DIM + c, HIDDEN, hq, hs, r);
                    worstDown =
                            Math.max(
                                    worstDown,
                                    Math.abs(expect - out.get(r * DIM + c))
                                            / (1 + Math.abs(expect)));
                }
            }
        }
        assertTrue(type + " gate/up worst " + worstHidden, worstHidden < 1e-3);
        assertTrue(type + " down worst " + worstDown, worstDown < 1e-3);
    }

    @Test
    public void q4_0ExpertsMatchTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        check(GGMLType.Q4_0);
    }

    @Test
    public void q4_1ExpertsMatchTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        check(GGMLType.Q4_1);
    }

    @Test
    public void q4_KExpertsMatchTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        check(GGMLType.Q4_K);
    }

    @Test
    public void q5_KExpertsMatchTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        check(GGMLType.Q5_K);
    }

    @Test
    public void q6_KExpertsMatchTheHost() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        check(GGMLType.Q6_K);
    }
}
