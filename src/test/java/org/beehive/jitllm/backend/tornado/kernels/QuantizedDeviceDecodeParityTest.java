package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.function.BiFunction;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.beehive.jllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jllm.tensor.standard.Q6_KFloatTensor;
import org.beehive.jllm.tensor.standard.Q8_0FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * Every quantized representation the engine retains on a device, decoded on both sides of the same
 * bytes.
 *
 * <p>One test for all six, because the property is the same one and the failure is the same one: a
 * decode that addresses a nibble, a high bit, a scale or a sub-block wrongly produces weights of
 * entirely plausible magnitude, and a model built on it generates fluent, wrong text at normal
 * throughput. Nothing downstream notices. A GPU-versus-GPU comparison cannot see it either, since
 * both sides would be equally wrong.
 *
 * <h2>Adversarial, not merely random</h2>
 *
 * <p>Random bytes exercise the common paths and miss the ones that matter. Here each block's
 * <b>scale fields</b> — at their real offsets, which for Q6_K means 208 and not 0 — cycle through
 * the half-precision values that behave specially: both zeros, both ends of the subnormal range,
 * unity of both signs, and both extremes of the normal range. The payload cycles through all-zero,
 * all-ones and both alternating bit patterns, with every fourth block left random. That drives
 * Q5_K's fifth-bit plane both ways for every position, the {@code subBlock >= 4} branch of the
 * K-quant scale unpacking where a 6-bit value straddles two bytes, Q6_K's signed recentring, and
 * Q4_1's minimum against Q4_0's recentring by eight.
 *
 * <p>The comparison is <b>bit equality</b>. Both sides evaluate the same integer addressing and the
 * same short float expression, so anything else is a real difference rather than rounding.
 */
public class QuantizedDeviceDecodeParityTest {

    /** A decoder over device bytes: (block byte offset, index within block) to value. */
    private interface DeviceDecode {
        float decode(ByteArray w, int blockByteOffset, int withinBlock);
    }

    /**
     * Half-precision bit patterns that behave specially, as raw shorts.
     *
     * <p>Every finite corner: both zeros, both ends of the subnormal range, unity of both signs,
     * and both extremes of the normal range. A scale that is subnormal or negative is where a
     * decoder that reconstructs the exponent by hand rather than reinterpreting bits goes wrong.
     *
     * <p><b>Infinity and NaN are deliberately absent.</b> {@code TransformerComputeKernelsQ6_K}
     * assembles its half from two byte loads and states that it does not handle them, because a
     * quantized block scale is neither — a constraint of well-formed GGUF, not an oversight, and
     * the alternative is two branches in the innermost loop of every Q6_K weight. Injecting them
     * would assert a property the engine does not claim. It is recorded as a known divergence
     * instead: on a corrupt file, that one kernel yields a finite number where the host yields NaN.
     */
    private static final short[] SPECIAL_HALVES = {
        (short) 0x0000, // +0
        (short) 0x8000, // -0
        (short) 0x0001, // smallest subnormal
        (short) 0x03FF, // largest subnormal
        (short) 0x3C00, // 1.0
        (short) 0xBC00, // -1.0
        (short) 0x7BFF, // largest normal
        (short) 0xFBFF, // most negative normal
    };

    /** Payload byte patterns: all clear, all set, and both alternations. */
    private static final byte[] PAYLOAD_PATTERNS = {0x00, (byte) 0xFF, (byte) 0xAA, 0x55};

    // ---- fixture construction -------------------------------------------------

    /**
     * Blocks whose <b>scale fields</b> walk {@link #SPECIAL_HALVES} and whose payloads walk {@link
     * #PAYLOAD_PATTERNS}, with the remainder random so no path is left untried.
     *
     * <p>The scale offsets are passed in rather than assumed to be at the front of the block,
     * because for Q6_K they are not: its 210 bytes are {@code ql}, {@code qh}, sixteen signed
     * sub-block scales and only then {@code d}, at offset 208. Writing "headers" at offset zero
     * would corrupt {@code ql} and leave the scale random — which is a different test, and a worse
     * one.
     *
     * @param scaleOffsets byte offsets of this format's fp16 scale fields within a block
     */
    private static byte[] adversarialBlocks(
            int blocks, int blockBytes, int[] scaleOffsets, long seed) {
        byte[] raw = new byte[blocks * blockBytes];
        Random random = new Random(seed);
        random.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes;
            // Every fourth block keeps its random payload, which is what drives the scale/min
            // packing across its whole range; the rest get a uniform bit pattern.
            if (b % 4 != 3) {
                byte pattern = PAYLOAD_PATTERNS[b % PAYLOAD_PATTERNS.length];
                for (int i = 0; i < blockBytes; i++) {
                    raw[base + i] = pattern;
                }
            }
            // Scales last, so they survive the payload fill.
            for (int h = 0; h < scaleOffsets.length; h++) {
                short half = SPECIAL_HALVES[(b + h) % SPECIAL_HALVES.length];
                raw[base + scaleOffsets[h]] = (byte) (half & 0xFF);
                raw[base + scaleOffsets[h] + 1] = (byte) ((half >> 8) & 0xFF);
            }
        }
        return raw;
    }

    private static ByteArray toDevice(byte[] raw) {
        // Built element-wise rather than wrapped: a TornadoNativeArray segment carries a 16-byte
        // header, so wrapping a bare segment would shift every offset and this would be testing the
        // harness rather than the decode.
        ByteArray array = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            array.set(i, raw[i]);
        }
        return array;
    }

    /**
     * Runs one representation's two decoders over identical adversarial bytes.
     *
     * @param hostTensor builds the CPU tensor from (element count, segment)
     */
    private static void assertDecodersAgree(
            String name,
            GGMLType type,
            int[] scaleOffsets,
            BiFunction<Integer, MemorySegment, FloatTensor> hostTensor,
            DeviceDecode deviceDecode) {
        int blockSize = type.getBlockSize();
        int blockBytes = type.getTypeSize();
        int blocks = 24;
        int elements = blocks * blockSize;
        byte[] raw =
                adversarialBlocks(blocks, blockBytes, scaleOffsets, 20260908L + name.hashCode());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor host = hostTensor.apply(elements, segment);
            ByteArray device = toDevice(raw);

            int compared = 0;
            for (int i = 0; i < elements; i++) {
                int blockIndex = i / blockSize;
                int withinBlock = i - blockIndex * blockSize;
                float expected = host.getFloat(i);
                float actual = deviceDecode.decode(device, blockIndex * blockBytes, withinBlock);
                if (Float.isNaN(expected)) {
                    assertTrue(
                            name + " element " + i + ": host NaN, device " + actual,
                            Float.isNaN(actual));
                } else {
                    assertEquals(
                            name
                                    + " element "
                                    + i
                                    + " (block "
                                    + blockIndex
                                    + ", offset "
                                    + withinBlock
                                    + ") decodes differently on the device than on the host",
                            expected,
                            actual,
                            0f);
                }
                compared++;
            }
            assertTrue(name + " compared nothing", compared == elements);
        }
    }

    // ---- one arm per representation -------------------------------------------

    @Test
    public void q4_0DecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q4_0",
                GGMLType.Q4_0,
                new int[] {0},
                (n, seg) -> new Q4_0FloatTensor(n, seg),
                TransformerComputeKernelsQ4_0::decode);
    }

    @Test
    public void q4_1DecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q4_1",
                GGMLType.Q4_1,
                new int[] {0, 2},
                (n, seg) -> new Q4_1FloatTensor(n, seg),
                TransformerComputeKernelsQ4_1::decode);
    }

    @Test
    public void q4_KDecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q4_K",
                GGMLType.Q4_K,
                new int[] {0, 2},
                (n, seg) -> new Q4_KFloatTensor(n, seg),
                TransformerComputeKernelsQ4_K::decode);
    }

    @Test
    public void q5_KDecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q5_K",
                GGMLType.Q5_K,
                new int[] {0, 2},
                (n, seg) -> new Q5_KFloatTensor(n, seg),
                TransformerComputeKernelsQ5_K::decode);
    }

    @Test
    public void q6_KDecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q6_K",
                // d sits at 208, after ql, qh and the sixteen signed sub-block scales.
                GGMLType.Q6_K,
                new int[] {208},
                (n, seg) -> new Q6_KFloatTensor(n, seg),
                TransformerComputeKernelsQ6_K::decode);
    }

    /**
     * Q8_0 has no shared decode helper — every kernel inlines it — so the expression the kernels
     * use is written out here and held against the host tensor. If a shared helper is ever
     * extracted, this is the check it must keep passing.
     */
    @Test
    public void q8_0DecodesIdenticallyOnBothSides() {
        assertDecodersAgree(
                "Q8_0",
                GGMLType.Q8_0,
                new int[] {0},
                (n, seg) -> new Q8_0FloatTensor(n, seg),
                (w, blockByteOffset, withinBlock) ->
                        w.getHalfFloat(blockByteOffset).getFloat32()
                                * w.get(blockByteOffset + 2 + withinBlock));
    }

    // ---- the test's own sensitivity -------------------------------------------

    /**
     * The comparison above would in fact catch a mis-addressed decode.
     *
     * <p>A parity test is only worth what it detects, and every defect this class exists to find is
     * one that leaves the output well-formed. So each injected fault below is a real mistake
     * somebody could make — the wrong nibble half, Q4_0's recentring applied to Q4_1, Q5_K's fifth
     * bit taken from the wrong bit of the right byte — and each must be seen to disagree with the
     * host. A fault that slipped through would mean the arms above prove less than they appear to.
     */
    @Test
    public void anInjectedAddressingDefectIsDetected() {
        assertDefectDetected(
                "Q4_0 reading the wrong nibble half",
                GGMLType.Q4_0,
                new int[] {0},
                (n, seg) -> new Q4_0FloatTensor(n, seg),
                (w, off, within) -> {
                    int half = 1 - within / 16; // inverted
                    int byteIndex = within - (within / 16) * 16;
                    int packed = w.get(off + 2 + byteIndex) & 0xFF;
                    int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
                    return w.getHalfFloat(off).getFloat32() * (q - 8);
                });

        assertDefectDetected(
                "Q4_1 recentred by eight, as if it were Q4_0",
                GGMLType.Q4_1,
                new int[] {0, 2},
                (n, seg) -> new Q4_1FloatTensor(n, seg),
                (w, off, within) -> {
                    int half = within / 16;
                    int byteIndex = within - half * 16;
                    int packed = w.get(off + 4 + byteIndex) & 0xFF;
                    int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
                    return w.getHalfFloat(off).getFloat32() * (q - 8)
                            + w.getHalfFloat(off + 2).getFloat32();
                });

        assertDefectDetected(
                "Q5_K taking the fifth bit from the wrong bit position",
                GGMLType.Q5_K,
                new int[] {0, 2},
                (n, seg) -> new Q5_KFloatTensor(n, seg),
                (w, off, within) -> {
                    int pairIndex = within / 64;
                    int posInPair = within - pairIndex * 64;
                    int highNibble = posInPair / 32;
                    int subBlock = pairIndex * 2 + highNibble;
                    int posInHalf = posInPair - highNibble * 32;
                    int qsByte = w.get(off + 48 + pairIndex * 32 + posInHalf) & 0xFF;
                    int q = (highNibble == 0) ? (qsByte & 0xF) : ((qsByte >> 4) & 0xF);
                    int qhByte = w.get(off + 16 + posInHalf) & 0xFF;
                    q += ((qhByte >> pairIndex) & 1) * 16; // wrong: ignores the nibble half
                    int scalesBase = off + 4;
                    int sc;
                    int m;
                    if (subBlock < 4) {
                        sc = w.get(scalesBase + subBlock) & 63;
                        m = w.get(scalesBase + subBlock + 4) & 63;
                    } else {
                        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
                        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
                        sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
                        m =
                                ((lowScale >> 4) & 0xF)
                                        | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
                    }
                    return w.getHalfFloat(off).getFloat32() * sc * q
                            - w.getHalfFloat(off + 2).getFloat32() * m;
                });

        assertDefectDetected(
                "Q4_K reading the sub-block scale without the straddled high bits",
                GGMLType.Q4_K,
                new int[] {0, 2},
                (n, seg) -> new Q4_KFloatTensor(n, seg),
                (w, off, within) -> {
                    int pairIndex = within / 64;
                    int posInPair = within - pairIndex * 64;
                    int highNibble = posInPair / 32;
                    int subBlock = pairIndex * 2 + highNibble;
                    int qByte =
                            w.get(off + 16 + pairIndex * 32 + (posInPair - highNibble * 32)) & 0xFF;
                    int q = (highNibble == 0) ? (qByte & 0xF) : ((qByte >> 4) & 0xF);
                    // Wrong: the low six bits for every sub-block, ignoring the >= 4 packing.
                    int sc = w.get(off + 4 + subBlock) & 63;
                    int m = w.get(off + 4 + subBlock + 4) & 63;
                    return w.getHalfFloat(off).getFloat32() * sc * q
                            - w.getHalfFloat(off + 2).getFloat32() * m;
                });
    }

    /** Asserts that a deliberately wrong decoder disagrees with the host somewhere. */
    private static void assertDefectDetected(
            String what,
            GGMLType type,
            int[] scaleOffsets,
            BiFunction<Integer, MemorySegment, FloatTensor> hostTensor,
            DeviceDecode faulty) {
        try {
            assertDecodersAgree(what, type, scaleOffsets, hostTensor, faulty);
        } catch (AssertionError expected) {
            return;
        }
        fail(
                "the parity comparison did not detect "
                        + what
                        + "; the arms of this test therefore prove less than they appear to");
    }
}
