package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.tensor.Q4_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * The Q4_K and Q6_K repacks into MLX's affine format against the host's own decoders ({@link
 * Q4_KFloatTensor}, {@link Q6_KFloatTensor}), over random blocks: Q4_K exact up to the FP16
 * rounding of each group's scale and bias, Q6_K within its 8-bit re-quantization, and both kept in
 * a stack's widest bits.
 */
public class KQuantMlxRepackTest {

    private static final int BLOCKS = 6;

    @Test
    public void q4kRepacksToFourBitsAsTheHostDecodes() {
        byte[] raw = randomQ4K(BLOCKS, new Random(1));
        TornadoTensor t = new Q4_KTornadoTensor(bytes(raw));
        MlxPrefillProjections.Matrix m = MlxPrefillProjections.affine(t);
        assertEquals(4, m.bits());
        Q4_KFloatTensor host = new Q4_KFloatTensor(BLOCKS * 256, nativeCopy(raw));
        float[] affine = dequantize(m);
        for (int i = 0; i < BLOCKS * 256; i++) {
            float expected = host.getFloat(i);
            assertEquals("weight " + i, expected, affine[i], 2e-3f * (1 + Math.abs(expected)));
        }
    }

    @Test
    public void q6kRepacksToEightBitsWithinItsRequantization() {
        byte[] raw = randomQ6K(BLOCKS, new Random(2));
        TornadoTensor t = new Q6_KTornadoTensor(bytes(raw));
        MlxPrefillProjections.Matrix m = MlxPrefillProjections.affine(t);
        assertEquals(8, m.bits());
        Q6_KFloatTensor host = new Q6_KFloatTensor(BLOCKS * 256, nativeCopy(raw));
        float[] affine = dequantize(m);
        for (int g = 0; g < BLOCKS * 8; g++) {
            float lo = Float.MAX_VALUE;
            float hi = -Float.MAX_VALUE;
            for (int l = 0; l < 32; l++) {
                lo = Math.min(lo, host.getFloat(g * 32 + l));
                hi = Math.max(hi, host.getFloat(g * 32 + l));
            }
            // Half a step of the group's 8-bit range, plus the FP16 rounding of scale and bias.
            float tolerance =
                    (hi - lo) / 255f * 0.5f + 2e-3f * Math.max(Math.abs(lo), Math.abs(hi));
            for (int l = 0; l < 32; l++) {
                int i = g * 32 + l;
                assertEquals("weight " + i, host.getFloat(i), affine[i], tolerance);
            }
        }
        float[] decoded = decodeFp16(MlxPrefillProjections.decode(t));
        for (int i = 0; i < BLOCKS * 256; i++) {
            float expected = host.getFloat(i);
            assertEquals("weight " + i, expected, decoded[i], 1e-3f * (1 + Math.abs(expected)));
        }
    }

    @Test
    public void aStackWithQ6KHoldsItsQ4KExactlyAtEightBits() {
        byte[] q4 = randomQ4K(BLOCKS, new Random(3));
        byte[] q6 = randomQ6K(BLOCKS, new Random(4));
        TornadoTensor a = new Q4_KTornadoTensor(bytes(q4));
        TornadoTensor b = new Q6_KTornadoTensor(bytes(q6));
        MlxPrefillProjections.Matrix alone = MlxPrefillProjections.affine(a);
        MlxPrefillProjections.Matrix stack = MlxPrefillProjections.affine(a, b);
        assertEquals(8, stack.bits());
        float[] four = dequantize(alone);
        float[] eight = dequantize(stack);
        for (int i = 0; i < four.length; i++) {
            assertEquals("weight " + i, four[i], eight[i], 0f);
        }
    }

    @Test
    public void theCacheMapsBackWhatItWrote() throws Exception {
        Path dir = Files.createTempDirectory("mlx-cache-test");
        byte[] raw = randomQ4K(BLOCKS, new Random(5));
        TornadoTensor t = new Q4_KTornadoTensor(bytes(raw));
        MlxWeightCache cache = new MlxWeightCache(dir);
        float[] built = dequantize(cache.affine(t));
        float[] mapped = dequantize(cache.affine(t));
        assertTrue(Files.list(dir).anyMatch(p -> p.toString().endsWith(".mlx")));
        float[] direct = dequantize(MlxPrefillProjections.affine(t));
        for (int i = 0; i < direct.length; i++) {
            assertEquals(direct[i], built[i], 0f);
            assertEquals(direct[i], mapped[i], 0f);
        }
    }

    /** {@code scale * q + bias} for every weight of an affine matrix. */
    private static float[] dequantize(MlxPrefillProjections.Matrix m) {
        int groups = m.scales().getSize();
        int perWord = 32 / m.bits();
        int mask = (1 << m.bits()) - 1;
        float[] out = new float[groups * 32];
        for (int g = 0; g < groups; g++) {
            float scale = m.scales().get(g).getFloat32();
            float bias = m.biases().get(g).getFloat32();
            for (int l = 0; l < 32; l++) {
                int word = m.q().get(g * 32 / perWord + l / perWord);
                int value = (word >>> ((l % perWord) * m.bits())) & mask;
                out[g * 32 + l] = scale * value + bias;
            }
        }
        return out;
    }

    private static float[] decodeFp16(uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray a) {
        float[] out = new float[a.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i).getFloat32();
        }
        return out;
    }

    /** The host decoders read through native addresses, not a heap array. */
    private static MemorySegment nativeCopy(byte[] raw) {
        MemorySegment s = java.lang.foreign.Arena.ofAuto().allocate(raw.length);
        MemorySegment.copy(raw, 0, s, ValueLayout.JAVA_BYTE, 0, raw.length);
        return s;
    }

    private static ByteArray bytes(byte[] raw) {
        ByteArray a = new ByteArray(raw.length);
        MemorySegment.copy(raw, 0, a.getSegment(), ValueLayout.JAVA_BYTE, 0, raw.length);
        return a;
    }

    /** Random Q4_K blocks with plausible, small FP16 {@code d} and {@code dmin}. */
    private static byte[] randomQ4K(int blocks, Random r) {
        byte[] raw = new byte[blocks * 144];
        r.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            putHalf(raw, b * 144, 0.001f + r.nextFloat() * 0.01f);
            putHalf(raw, b * 144 + 2, 0.001f + r.nextFloat() * 0.01f);
        }
        return raw;
    }

    /** Random Q6_K blocks with a plausible, small FP16 {@code d}. */
    private static byte[] randomQ6K(int blocks, Random r) {
        byte[] raw = new byte[blocks * 210];
        r.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            putHalf(raw, b * 210 + 208, 0.0005f + r.nextFloat() * 0.002f);
        }
        return raw;
    }

    private static void putHalf(byte[] raw, int offset, float value) {
        short h = Float.floatToFloat16(value);
        raw[offset] = (byte) h;
        raw[offset + 1] = (byte) (h >> 8);
    }
}
