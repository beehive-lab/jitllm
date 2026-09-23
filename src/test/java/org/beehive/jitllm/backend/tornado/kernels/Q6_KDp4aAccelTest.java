package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
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
 * The packed-integer {@code Q6_K} projection, against the host tensor that defines the format.
 *
 * <p>Q6_K is the most intricate layout that reaches the device here: 256 weights per super-block
 * with one fp16 scale, sixteen signed byte scales, and each quantum split between a nibble of
 * {@code ql} and a pair of bits of {@code qh}, with which nibble and which pair depending on where
 * the element sits in the block. Getting that mapping wrong produces weights of plausible
 * magnitude, so the first case checks the <b>unpacking itself</b> rather than a dot product over
 * it.
 *
 * <p>The activation is one-hot, so each output is one decoded weight and can be held against {@link
 * Q6_KFloatTensor}. One row per position covers every group, both nibbles, every {@code qh} shift
 * and every scale index in the block; the weights carry all sixty-four raw values and both signs of
 * byte scale.
 */
// @formatter:on
public class Q6_KDp4aAccelTest {

    private static final int QK_K = 256;
    private static final int BLOCK_BYTES = 210;
    private static final int QH_OFFSET = 128;
    private static final int SCALES_OFFSET = 192;
    private static final int D_OFFSET = 208;
    private static final int QK = 32;
    private static final int LOCAL = 128;

    /** One super-block wide, one row per element in it. */
    private static final int N = QK_K;

    private static final int D = QK_K;

    /**
     * Weights covering all sixty-four raw values and both signs of scale.
     *
     * <p>Row {@code r} shifts the pattern so a given position sees a different raw value in every
     * row, and the byte scales alternate sign across the sixteen of a block.
     */
    private static byte[] weights() {
        byte[] raw = new byte[D * BLOCK_BYTES];
        for (int row = 0; row < D; row++) {
            int base = row * BLOCK_BYTES;
            // d: about 0.0625, positive, so the byte scales carry the sign.
            raw[base + D_OFFSET] = 0x00;
            raw[base + D_OFFSET + 1] = 0x2C;
            for (int i = 0; i < 16; i++) {
                int magnitude = 1 + ((row + i) % 7);
                raw[base + SCALES_OFFSET + i] = (byte) ((i % 2 == 0) ? magnitude : -magnitude);
            }
            // Every element of the block gets a raw value; the walk covers 0..63 as row varies.
            for (int e = 0; e < QK_K; e++) {
                int value = (e + row) % 64;
                writeQuantum(raw, base, e, value);
            }
        }
        return raw;
    }

    /** Places a six-bit value at element {@code e}, in the layout the host tensor reads. */
    private static void writeQuantum(byte[] raw, int base, int e, int value) {
        int half = e / 128;
        int posInHalf = e % 128;
        int groupInHalf = posInHalf / 32;
        int posInGroup = posInHalf % 32;
        int qlAt = base + half * 64 + ((groupInHalf & 1) * 32) + posInGroup;
        int qhAt = base + QH_OFFSET + half * 32 + posInGroup;
        int low = value & 0xF;
        int high = (value >> 4) & 3;
        if (groupInHalf >= 2) {
            raw[qlAt] = (byte) ((raw[qlAt] & 0x0F) | (low << 4));
        } else {
            raw[qlAt] = (byte) ((raw[qlAt] & 0xF0) | low);
        }
        int shift = 2 * groupInHalf;
        raw[qhAt] = (byte) ((raw[qhAt] & ~(3 << shift)) | (high << shift));
    }

    @Test
    public void everyQuantumUnpacksAsTheHostTensorReadsIt() throws Exception {
        byte[] raw = weights();

        // One-hot: row r selects element r, so out[r] is that row's weight r.
        float[] host = new float[N];
        FloatArray x = new FloatArray(N);
        IntArray quants = new IntArray(N / 4);
        FloatArray scales = new FloatArray(N / QK);
        IntArray sums = new IntArray(N / QK);
        FloatArray out = new FloatArray(D);
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }

        float[] decoded = new float[D];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            Q6_KFloatTensor reference = new Q6_KFloatTensor(D * N, segment);
            for (int row = 0; row < D; row++) {
                decoded[row] = reference.getFloat(row * N + row);
            }
        }

        for (int row = 0; row < D; row++) {
            java.util.Arrays.fill(host, 0.0f);
            host[row] = 1.0f;
            for (int i = 0; i < N; i++) {
                x.set(i, host[i]);
            }
            quants.init(0);
            scales.init(0.0f);
            sums.init(0);
            out.init(0.0f);

            TaskGraph graph =
                    new TaskGraph("q6k")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    x,
                                    w,
                                    quants,
                                    scales,
                                    sums,
                                    out)
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
                                    TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A,
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
            WorkerGrid1D blocks = new WorkerGrid1D(N);
            blocks.setLocalWork(QK, 1, 1);
            scheduler.addWorkerGrid("q6k.quantize", blocks);
            WorkerGrid1D rows = new WorkerGrid1D(D * LOCAL);
            rows.setLocalWork(LOCAL, 1, 1);
            scheduler.addWorkerGrid("q6k.matvec", rows);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            float got = out.get(row);
            assertTrue("row " + row + " is " + got, Float.isFinite(got));
            // The one-hot block quantizes to a scale of 1/127 and a quant of 127, so the product
            // is the decoded weight exactly but for that round trip.
            assertEquals(
                    "element " + row + " of its own row, raw value " + ((row + row) % 64),
                    decoded[row],
                    got,
                    Math.max(1e-4f, Math.abs(decoded[row]) * 1e-3f));
        }
    }

    // @formatter:off
    /**
     * A dense projection over several super-blocks, against a reference quantizing identically.
     *
     * <p>The one-hot case proves the unpacking and nothing else: every dot product it forms has a
     * single non-zero term, so an accumulation that dropped or double-counted terms would pass it.
     * This one sums 1024 of them per output with values chosen to cancel — adjacent weights of
     * opposite sign and activations that change sign within a block — so a misplaced term shows up
     * rather than averaging out.
     *
     * <p>Four super-blocks, byte scales of both signs and differing magnitude within each block, a
     * zero activation block whose scale must be zero, and activation magnitudes three orders apart.
     * The reference performs the same block quantization, so what is compared is the kernel's
     * accumulation and scaling, not the representation's cost -- that difference is reported
     * separately, against the floating-point kernel.
     */
    // @formatter:on
    @Test
    public void aDenseProjectionMatchesAReferenceThatQuantizesTheSameWay() throws Exception {
        assertDenseMatchesReference(4 * QK_K, 96, LOCAL);
    }

    // @formatter:off
    /**
     * The same comparison at the width the vocabulary projection actually dispatches.
     *
     * <p>{@code LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS} is 256, which is eight
     * warps, and the reduction folds each warp with shuffles before one barrier and a lane-zero
     * combine. Sixteen super-blocks give 256 scale runs, so <b>every</b> warp carries work, and a
     * partial that never reached the combine would change the row's total. The case above runs 64
     * runs over 128 lanes, where the upper warps are idle and would hide exactly that.
     */
    // @formatter:on
    @Test
    public void aDenseProjectionMatchesTheReferenceAtTheDispatchedWidth() throws Exception {
        assertDenseMatchesReference(16 * QK_K, 24, 256);
    }

    private void assertDenseMatchesReference(int dense, int rows, int local) throws Exception {
        Random random = new Random(20260911L);
        byte[] raw = new byte[rows * (dense / QK_K) * BLOCK_BYTES];
        for (int block = 0; block < rows * (dense / QK_K); block++) {
            int base = block * BLOCK_BYTES;
            raw[base + D_OFFSET] = 0x00;
            raw[base + D_OFFSET + 1] = 0x2C;
            for (int i = 0; i < 16; i++) {
                int magnitude = 1 + ((block + i) % 9);
                raw[base + SCALES_OFFSET + i] = (byte) ((i % 3 == 0) ? -magnitude : magnitude);
            }
            for (int e = 0; e < QK_K; e++) {
                // Neighbours of opposite sign about the midpoint, so terms cancel.
                int value = (e % 2 == 0) ? 32 + random.nextInt(32) : random.nextInt(32);
                writeQuantum(raw, base, e, value);
            }
        }

        float[] host = new float[dense];
        for (int i = 0; i < dense; i++) {
            int block = i / QK;
            if (block == 9) {
                host[i] = 0.0f;
            } else if (block % 3 == 0) {
                host[i] = (float) (Math.sin(0.013 * i) * 1e-3);
            } else if (block % 3 == 1) {
                host[i] = (float) (Math.cos(0.021 * i) * 25.0);
            } else {
                host[i] = (i % 2 == 0 ? 1 : -1) * (0.5f + (i % 11) * 0.4f);
            }
        }

        FloatArray x = new FloatArray(dense);
        for (int i = 0; i < dense; i++) {
            x.set(i, host[i]);
        }
        IntArray quants = new IntArray(dense / 4);
        FloatArray scales = new FloatArray(dense / QK);
        IntArray sums = new IntArray(dense / QK);
        quants.init(0);
        scales.init(0.0f);
        sums.init(0);
        FloatArray out = new FloatArray(rows);
        out.init(0.0f);
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }

        TaskGraph graph =
                new TaskGraph("dense" + local)
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
                                TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                out,
                                w,
                                dense,
                                rows,
                                local)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION, out, quants, scales, sums);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(dense);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("dense" + local + ".quantize", blocks);
        WorkerGrid1D rowGrid = new WorkerGrid1D(rows * local);
        rowGrid.setLocalWork(local, 1, 1);
        scheduler.addWorkerGrid("dense" + local + ".matvec", rowGrid);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        assertEquals("the zero block's scale on the device", 0.0f, scales.get(9), 0.0f);
        assertEquals("the zero block's sum on the device", 0, sums.get(9));

        // The reference: the same quants, the host tensor's own decode, summed in double.
        int[] hostQuants = new int[dense / 4];
        float[] hostScales = new float[dense / QK];
        for (int block = 0; block < dense / QK; block++) {
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
            Q6_KFloatTensor reference = new Q6_KFloatTensor(rows * dense, segment);
            for (int row = 0; row < rows; row++) {
                double sum = 0;
                for (int i = 0; i < dense; i++) {
                    int block = i / QK;
                    int quant = (byte) ((hostQuants[i / 4] >> ((i % 4) * 8)) & 0xFF);
                    sum += reference.getFloat(row * dense + i) * hostScales[block] * quant;
                }
                float got = out.get(row);
                assertTrue("row " + row + " is " + got, Float.isFinite(got));
                largest = Math.max(largest, Math.abs(sum));
                worst = Math.max(worst, Math.abs(sum - got));
            }
        }
        System.out.printf(
                "[Q6K-DENSE] local %d: against the same-quantization reference: worst %.6g,"
                        + " largest |out| %.6g over %d rows of %d%n",
                local, worst, largest, rows, dense);
        assertTrue("worst " + worst + " against largest " + largest, worst <= 1e-4 * largest);
    }
}
