package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections.LayerWeights;
import org.beehive.jitllm.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.PackedTiles;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The host-side forms the MLX prefill gives Q8_0 and Q4_0 projections, and which model weights its
 * graphs bind. Host memory only: no device, no MLX.
 *
 * <p>The affine repack is claimed to be <b>exact</b>: MLX reconstructs {@code scale * q + bias}
 * from unsigned {@code q}, and Q8_0's {@code s * q} and Q4_0's {@code s * (q - 8)} are that with
 * {@code q + 128} and a bias of {@code -128 * s}, or the nibble and a bias of {@code -8 * s} — a
 * power-of-two multiple of an FP16 scale, so the bias is exact in FP16 too. Each test rebuilds
 * every element from the packed words, the scales and the biases, the way MLX's kernel reads them,
 * and requires it to equal the file's own value bit for bit.
 */
// @formatter:on
public class MlxPrefillProjectionsTest {

    private static final int GROUP = MlxPrefill.GROUP;

    // ── fixtures: GGUF blocks written by hand ────────────────────────────────

    /** Q8_0 blocks: an FP16 scale and 32 signed bytes each. */
    private static Q8_0TornadoTensor q8(float[] scales, byte[][] values) {
        ByteArray bytes = new ByteArray(scales.length * 34);
        for (int b = 0; b < scales.length; b++) {
            bytes.getSegment()
                    .set(
                            ValueLayout.JAVA_SHORT_UNALIGNED,
                            b * 34L,
                            Float.floatToFloat16(scales[b]));
            for (int j = 0; j < GROUP; j++) {
                bytes.set(b * 34 + 2 + j, values[b][j]);
            }
        }
        return new Q8_0TornadoTensor(bytes, Shape.of(scales.length * GROUP));
    }

    /**
     * Q4_0 blocks: an FP16 scale and 16 bytes, byte {@code j} holding element {@code j} in its low
     * nibble and element {@code j + 16} in its high one.
     */
    private static Q4_0TornadoTensor q4(float[] scales, int[][] nibbles) {
        ByteArray bytes = new ByteArray(scales.length * 18);
        for (int b = 0; b < scales.length; b++) {
            bytes.getSegment()
                    .set(
                            ValueLayout.JAVA_SHORT_UNALIGNED,
                            b * 18L,
                            Float.floatToFloat16(scales[b]));
            for (int j = 0; j < GROUP / 2; j++) {
                bytes.set(b * 18 + 2 + j, (byte) (nibbles[b][j] | (nibbles[b][j + 16] << 4)));
            }
        }
        return new Q4_0TornadoTensor(bytes, Shape.of(scales.length * GROUP));
    }

    /** Every signed byte at least once across the blocks, extremes first. */
    private static byte[][] signedRamp(int blocks) {
        byte[][] values = new byte[blocks][GROUP];
        int v = -128;
        for (int b = 0; b < blocks; b++) {
            for (int j = 0; j < GROUP; j++) {
                values[b][j] = (byte) v;
                v = v == 127 ? -128 : v + 1;
            }
        }
        values[0][0] = -128;
        values[0][1] = 127;
        values[0][2] = 0;
        values[0][3] = -1;
        return values;
    }

    /** Element {@code j} of block {@code b}: a different nibble at every position and block. */
    private static int[][] nibbleRamp(int blocks) {
        int[][] nibbles = new int[blocks][GROUP];
        for (int b = 0; b < blocks; b++) {
            for (int j = 0; j < GROUP; j++) {
                nibbles[b][j] = (j * 7 + b * 3) & 0xF;
            }
        }
        return nibbles;
    }

    private static float h(float value) {
        return Float.float16ToFloat(Float.floatToFloat16(value));
    }

    // ── reading the affine form the way MLX does ─────────────────────────────

    /** Element {@code e}: {@code scale[g] * q + bias[g]}, q unpacked little-end first. */
    private static float reconstruct(MlxPrefillProjections.Matrix m, int e) {
        int bits = m.bits();
        int perWord = 32 / bits;
        int word = m.q().get(e / perWord);
        int q = (word >>> ((e % perWord) * bits)) & ((1 << bits) - 1);
        int group = e / GROUP;
        return m.scales().get(group).getFloat32() * q + m.biases().get(group).getFloat32();
    }

    // ── Q8_0 ─────────────────────────────────────────────────────────────────

    @Test
    public void q8AffineReconstructsEverySignedValueExactly() {
        float[] scales = {0.5f, h(-0.0312f), h(1.0e-3f), h(7.25f), 0f, h(-2.5f), h(0.1f), h(3e-5f)};
        byte[][] values = signedRamp(scales.length);
        MlxPrefillProjections.Matrix m = MlxPrefillProjections.affine(q8(scales, values));

        assertEquals(8, m.bits());
        assertNull("the affine form has no FP16 copy", m.fp16());
        assertEquals("four 8-bit values per word", scales.length * GROUP / 4, m.q().getSize());
        for (int b = 0; b < scales.length; b++) {
            float s = h(scales[b]);
            assertEquals("scale of group " + b, s, m.scales().get(b).getFloat32(), 0f);
            assertEquals("bias of group " + b, -128f * s, m.biases().get(b).getFloat32(), 0f);
            for (int j = 0; j < GROUP; j++) {
                float expected = s * values[b][j];
                float got = reconstruct(m, b * GROUP + j);
                assertEquals(
                        "block " + b + " element " + j + " (q=" + values[b][j] + ")",
                        Float.floatToRawIntBits(expected + 0f),
                        Float.floatToRawIntBits(got + 0f));
            }
        }
    }

    @Test
    public void q8ValuesArePackedLittleEndFirst() {
        byte[][] values = new byte[1][GROUP];
        values[0][0] = -128; // q = 0
        values[0][1] = -127; // q = 1
        values[0][2] = 0; // q = 128
        values[0][3] = 127; // q = 255
        MlxPrefillProjections.Matrix m = MlxPrefillProjections.affine(q8(new float[] {1f}, values));
        assertEquals(0x00 | (0x01 << 8) | (0x80 << 16) | (0xFF << 24), m.q().get(0));
    }

    @Test
    public void q8DecodedCopyIsTheFp16OfEveryValue() {
        float[] scales = {0.5f, h(-0.0312f), h(7.25f)};
        byte[][] values = signedRamp(scales.length);
        HalfFloatArray decoded = MlxPrefillProjections.decode(q8(scales, values));
        assertEquals(scales.length * GROUP, decoded.getSize());
        for (int b = 0; b < scales.length; b++) {
            for (int j = 0; j < GROUP; j++) {
                assertEquals(
                        "block " + b + " element " + j,
                        h(h(scales[b]) * values[b][j]),
                        decoded.get(b * GROUP + j).getFloat32(),
                        0f);
            }
        }
    }

    // ── Q4_0 ─────────────────────────────────────────────────────────────────

    @Test
    public void q4AffineKeepsGgufNibbleOrderAndReconstructsExactly() {
        float[] scales = {h(0.25f), h(-1.5f), h(3.0e-3f), 2f};
        int[][] nibbles = nibbleRamp(scales.length);
        MlxPrefillProjections.Matrix m = MlxPrefillProjections.affine(q4(scales, nibbles));

        assertEquals(4, m.bits());
        assertEquals("eight 4-bit values per word", scales.length * GROUP / 8, m.q().getSize());
        for (int b = 0; b < scales.length; b++) {
            float s = h(scales[b]);
            assertEquals("bias of group " + b, -8f * s, m.biases().get(b).getFloat32(), 0f);
            for (int j = 0; j < GROUP; j++) {
                // Element j < 16 is byte j's low nibble, element j >= 16 is byte (j - 16)'s high
                // one: the order llama.cpp's dequantize_row_q4_0 produces.
                float expected = s * (nibbles[b][j] - 8);
                assertEquals(
                        "block " + b + " element " + j,
                        Float.floatToRawIntBits(expected + 0f),
                        Float.floatToRawIntBits(reconstruct(m, b * GROUP + j) + 0f));
            }
        }
    }

    @Test
    public void q4DecodedCopyIsTheFp16OfEveryValue() {
        float[] scales = {h(0.25f), h(-1.5f)};
        int[][] nibbles = nibbleRamp(scales.length);
        HalfFloatArray decoded = MlxPrefillProjections.decode(q4(scales, nibbles));
        for (int b = 0; b < scales.length; b++) {
            for (int j = 0; j < GROUP; j++) {
                assertEquals(
                        "block " + b + " element " + j,
                        h(h(scales[b]) * (nibbles[b][j] - 8)),
                        decoded.get(b * GROUP + j).getFloat32(),
                        0f);
            }
        }
    }

    // ── stacking ─────────────────────────────────────────────────────────────

    @Test
    public void stackedTensorsKeepTheirOrderAndGroups() {
        byte[][] a = signedRamp(2);
        byte[][] b = new byte[1][GROUP];
        Arrays.fill(b[0], (byte) 5);
        MlxPrefillProjections.Matrix m =
                MlxPrefillProjections.matrix(
                        MlxPrefill.Projections.AFFINE,
                        q8(new float[] {0.5f, 0.25f}, a),
                        q8(new float[] {2f}, b));
        assertEquals("three groups, in the order stacked", 3, m.scales().getSize());
        assertEquals(2f, m.scales().get(2).getFloat32(), 0f);
        assertEquals(10f, reconstruct(m, 2 * GROUP + 7), 0f);
        assertEquals(0.5f * a[0][1], reconstruct(m, 1), 0f);
        assertEquals(0.25f * a[1][4], reconstruct(m, GROUP + 4), 0f);
    }

    @Test
    public void anF16MatrixIsTheModelsOwnArrayUnlessStacked() {
        FP16TornadoTensor wo = fp16(64);
        assertSame(
                "a single F16 tensor is multiplied as it is",
                wo.asHalfFloatArray(),
                MlxPrefillProjections.matrix(MlxPrefill.Projections.FP16, wo).fp16());
        HalfFloatArray stacked =
                MlxPrefillProjections.matrix(MlxPrefill.Projections.FP16, fp16(64), fp16(32))
                        .fp16();
        assertEquals(96, stacked.getSize());
    }

    // ── which model weights the prefill graphs bind ──────────────────────────

    @Test
    public void f16SplitProjectionsAreCopiedAndTheRestBound() {
        LayerWeights layer = llamaStyle(DataType.F16);
        Map<Object, Boolean> notBound =
                identitySet(
                        MlxPrefillProjections.notBound(
                                new MlxPrefill(DataType.F16, MlxPrefill.Projections.FP16), layer));
        for (TornadoTensor t : List.of(layer.qkv()[0], layer.qkv()[1], layer.qkv()[2])) {
            assertTrue("stacked QKV is a copy", notBound.containsKey(t));
            assertTrue("as its native array too", notBound.containsKey(t.asHalfFloatArray()));
        }
        for (TornadoTensor t : layer.gateUp()) {
            assertTrue("stacked gate/up is a copy", notBound.containsKey(t.asHalfFloatArray()));
        }
        assertTrue("wo is bound as it is", !notBound.containsKey(layer.wo().asHalfFloatArray()));
        assertTrue("w2 is bound as it is", !notBound.containsKey(layer.w2().asHalfFloatArray()));
        assertEquals(10, notBound.size());
    }

    @Test
    public void f16FusedProjectionsAreAllBound() {
        LayerWeights fused =
                new LayerWeights(
                        new TornadoTensor[] {fp16(96)},
                        fp16(32),
                        new TornadoTensor[] {fp16(128)},
                        fp16(64));
        assertEquals(
                0,
                MlxPrefillProjections.notBound(
                                new MlxPrefill(DataType.F16, MlxPrefill.Projections.FP16), fused)
                        .length);
    }

    @Test
    public void quantizedProjectionsAreNeverBound() {
        for (MlxPrefill.Projections form :
                new MlxPrefill.Projections[] {
                    MlxPrefill.Projections.AFFINE, MlxPrefill.Projections.FP16_DECODED
                }) {
            LayerWeights layer = llamaStyle(DataType.Q8_0);
            Map<Object, Boolean> notBound =
                    identitySet(
                            MlxPrefillProjections.notBound(
                                    new MlxPrefill(DataType.Q8_0, form), layer));
            for (TornadoTensor t : layer.all()) {
                assertTrue(form + ": " + t, notBound.containsKey(t));
                assertTrue(form + ": " + t, notBound.containsKey(t.asByteArray()));
            }
        }
    }

    // ── refusals rather than a fallback ──────────────────────────────────────

    @Test
    public void packedProjectionsAreRefusedByName() {
        Q4_0TornadoTensor packed =
                new Q4_0TornadoTensor(new ByteArray(18 * 4), new PackedTiles(4, 32));
        LayerWeights layer =
                new LayerWeights(
                        new TornadoTensor[] {packed}, packed, new TornadoTensor[] {packed}, packed);
        UnsupportedOperationException e =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                MlxPrefillProjections.of(
                                        new MlxPrefill(
                                                DataType.Q4_0, MlxPrefill.Projections.AFFINE),
                                        null,
                                        null,
                                        null,
                                        16,
                                        1,
                                        32,
                                        32,
                                        32,
                                        32,
                                        1f,
                                        i -> layer));
        assertTrue(e.getMessage(), e.getMessage().contains("MLX batched prefill"));
    }

    @Test
    public void aProjectionOfAnotherTypeIsRefused() {
        LayerWeights mixed =
                new LayerWeights(
                        new TornadoTensor[] {q8(new float[] {1f}, new byte[1][GROUP])},
                        q4(new float[] {1f}, new int[1][GROUP]),
                        new TornadoTensor[] {q8(new float[] {1f}, new byte[1][GROUP])},
                        q8(new float[] {1f}, new byte[1][GROUP]));
        assertThrows(
                IllegalStateException.class,
                () ->
                        MlxPrefillProjections.of(
                                new MlxPrefill(DataType.Q8_0, MlxPrefill.Projections.AFFINE),
                                null,
                                null,
                                null,
                                16,
                                1,
                                32,
                                32,
                                32,
                                32,
                                1f,
                                i -> mixed));
    }

    @Test
    public void aDecisionNamesOnlyFormsOfItsWeights() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new MlxPrefill(DataType.F16, MlxPrefill.Projections.AFFINE));
        assertThrows(
                IllegalArgumentException.class,
                () -> new MlxPrefill(DataType.Q8_0, MlxPrefill.Projections.FP16));
        assertThrows(
                IllegalArgumentException.class,
                () -> new MlxPrefill(DataType.Q4_K, MlxPrefill.Projections.AFFINE));
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static FP16TornadoTensor fp16(int elements) {
        return new FP16TornadoTensor(new HalfFloatArray(elements), Shape.of(elements));
    }

    private static LayerWeights llamaStyle(DataType type) {
        java.util.function.Supplier<TornadoTensor> t =
                () -> type == DataType.F16 ? fp16(GROUP) : q8(new float[] {1f}, new byte[1][GROUP]);
        return new LayerWeights(
                new TornadoTensor[] {t.get(), t.get(), t.get()},
                t.get(),
                new TornadoTensor[] {t.get(), t.get()},
                t.get());
    }

    private static Map<Object, Boolean> identitySet(Object[] objects) {
        Map<Object, Boolean> set = new IdentityHashMap<>();
        for (Object o : objects) {
            set.put(o, Boolean.TRUE);
        }
        return set;
    }
}
