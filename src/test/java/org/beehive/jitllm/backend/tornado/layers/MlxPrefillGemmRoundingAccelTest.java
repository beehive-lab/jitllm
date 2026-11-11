package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.ValueLayout;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxPrefill;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The MLX prefill GEMMs over Q8_0 weights, as {@link MlxPrefillProjections} builds them, against
 * the arithmetic they implement: FP16 activations times the weights rounded to FP16, summed in
 * {@code float}, the sum rounded to FP16.
 *
 * <p>Each output must be that sum within half an FP16 unit in the last place plus the error a
 * {@code float} sum of these terms can have in any order. Example: a sum of 256 terms whose
 * magnitudes add up to 40 may be off by 256 * 2^-24 * 40 = 6.1e-4 before it is rounded. The FP16
 * copies and the affine repack must also give bit-identical outputs where MLX does not split the
 * depth (512 rows here). The weights need more than FP16's 11 significant bits, so sums over the
 * exact weights break the bound, and so do sums over the weights in a wrong k order.
 */
// @formatter:on
public class MlxPrefillGemmRoundingAccelTest {

    private static final int K = 256;
    private static final int N = 1024;

    private static float scale(int row, int block) {
        int m = (row * 37 + block * 101 + 403) % 1023;
        return (1024 + m) / 1024.0f * 0x1p-6f;
    }

    private static int value(int row, int k) {
        return ((row * 13 + k * 7) % 255) - 127;
    }

    /** FP16 values in [-2, 2), from a hash of the token and k. */
    private static float activation(int token, int k) {
        int h = (token * 73 + k * 151 + 17) * 0x9E3779B1;
        return half(((h >>> 8) % 4096 - 2048) / 1024.0f);
    }

    private static float half(float v) {
        return Float.float16ToFloat(Float.floatToFloat16(v));
    }

    /**
     * The FP16 unit in the last place at finite FP16 value {@code v}: 32 at 65504, 2^-24 below
     * 2^-14.
     */
    private static float ulp16(float v) {
        return Math.max(Math.ulp(v) * 0x1p13f, 0x1p-24f);
    }

    /** The Q8_0 weights, row-major: exact, or rounded to FP16. */
    private static float[] weightValues(boolean rounded) {
        float[] w = new float[N * K];
        for (int r = 0; r < N; r++) {
            for (int k = 0; k < K; k++) {
                float exact = value(r, k) * scale(r, k / 32);
                w[r * K + k] = rounded ? half(exact) : exact;
            }
        }
        return w;
    }

    private static Q8_0TornadoTensor weights() {
        ByteArray bytes = new ByteArray(N * K / 32 * 34);
        for (int r = 0; r < N; r++) {
            for (int b = 0; b < K / 32; b++) {
                int base = (r * (K / 32) + b) * 34;
                bytes.getSegment()
                        .set(
                                ValueLayout.JAVA_SHORT_UNALIGNED,
                                base,
                                Float.floatToFloat16(scale(r, b)));
                for (int j = 0; j < 32; j++) {
                    bytes.set(base + 2 + j, (byte) value(r, b * 32 + j));
                }
            }
        }
        return new Q8_0TornadoTensor(bytes, Shape.of(N * K));
    }

    private static HalfFloatArray gemm(MlxPrefill.Projections form, int m) {
        MlxPrefillProjections.Matrix w = MlxPrefillProjections.matrix(form, weights());
        HalfFloatArray x = new HalfFloatArray(m * K);
        for (int t = 0; t < m; t++) {
            for (int k = 0; k < K; k++) {
                x.set(t * K + k, new HalfFloat(activation(t, k)));
            }
        }
        HalfFloatArray y = new HalfFloatArray(m * N);
        TaskGraph g =
                new TaskGraph("gemm" + form + m)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w.arrays());
        MlxPrefillProjections.gemm(g, "gemm", x, w, y, m, K, N);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, y);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return y;
    }

    /**
     * The outputs of {@code y} outside the bound around the sums over {@code w}, read with k
     * shifted by {@code shift} columns. A NaN or infinite output is always outside.
     */
    private static int outside(HalfFloatArray y, int m, float[] w, int shift) {
        int outside = 0;
        for (int t = 0; t < m; t++) {
            for (int n = 0; n < N; n++) {
                float got = y.get(t * N + n).getFloat32();
                if (!Float.isFinite(got)) {
                    outside++;
                    continue;
                }
                double sum = 0;
                double magnitude = 0;
                for (int k = 0; k < K; k++) {
                    double term = (double) activation(t, k) * w[n * K + (k + shift) % K];
                    sum += term;
                    magnitude += Math.abs(term);
                }
                double bound = ulp16(got) / 2 + K * 0x1p-24 * magnitude;
                if (Math.abs(got - sum) > bound) {
                    outside++;
                }
            }
        }
        return outside;
    }

    /** {@code m} rows of the FP16-rounded sums over {@code w}, as an exact GEMM would give them. */
    private static HalfFloatArray roundedSums(int m, float[] w) {
        HalfFloatArray y = new HalfFloatArray(m * N);
        for (int t = 0; t < m; t++) {
            for (int n = 0; n < N; n++) {
                double sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += (double) activation(t, k) * w[n * K + k];
                }
                y.set(t * N + n, new HalfFloat((float) sum));
            }
        }
        return y;
    }

    private static void check(String name, HalfFloatArray y, int m) {
        float[] rounded = weightValues(true);
        assertEquals(name + ": outputs off the FP16-weight sums", 0, outside(y, m, rounded, 0));
        int total = m * N;
        int exact = outside(y, m, weightValues(false), 0);
        int shifted = outside(y, m, rounded, 1);
        System.out.printf(
                "%s: %d outside the exact-weight sums, %d outside the shifted-k sums, of %d%n",
                name, exact, shifted, total);
        assertTrue(
                name + ": the data cannot tell FP16 weights from exact ones", exact * 10 > total);
        assertTrue(name + ": the data cannot tell a wrong k order", shifted * 2 > total);
    }

    /** Host only: the bound accepts the rounded sums and rejects NaN, infinities and 65504. */
    @Test
    public void nonFiniteOutputsAreOutside() {
        float[] rounded = weightValues(true);
        HalfFloatArray y = roundedSums(2, rounded);
        assertEquals("the rounded sums themselves", 0, outside(y, 2, rounded, 0));
        y.set(0, new HalfFloat(Float.NaN));
        y.set(1, new HalfFloat(Float.POSITIVE_INFINITY));
        y.set(N, new HalfFloat(Float.NEGATIVE_INFINITY));
        y.set(N + 1, new HalfFloat(65504.0f));
        assertEquals("NaN, both infinities and 65504", 4, outside(y, 2, rounded, 0));
    }

    @Test
    public void fp16CopiesSumFp16WeightsInFloat() {
        assumeTrue("environment absent: MLX on Metal", MlxPrefill.available());
        check("FP16 copies, 512 rows", gemm(MlxPrefill.Projections.FP16_DECODED, 512), 512);
        check("FP16 copies, 16 rows", gemm(MlxPrefill.Projections.FP16_DECODED, 16), 16);
    }

    /** At 512 rows MLX's quantized GEMM does not split its depth; below, see issue #214. */
    @Test
    public void affineRepackMatchesTheFp16CopiesAt512Rows() {
        assumeTrue("environment absent: MLX on Metal", MlxPrefill.available());
        HalfFloatArray affine = gemm(MlxPrefill.Projections.AFFINE, 512);
        check("affine, 512 rows", affine, 512);
        HalfFloatArray copies = gemm(MlxPrefill.Projections.FP16_DECODED, 512);
        for (int i = 0; i < 512 * N; i++) {
            assertEquals(
                    "output " + i,
                    copies.get(i).getHalfFloatValue(),
                    affine.get(i).getHalfFloatValue());
        }
    }
}
