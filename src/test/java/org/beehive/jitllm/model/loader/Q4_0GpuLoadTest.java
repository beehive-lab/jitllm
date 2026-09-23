package org.beehive.jllm.model.loader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import org.beehive.jllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.format.Float16;
import org.beehive.jllm.format.GGMLTensorEntry;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.tensor.standard.Q4_0FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Q4_0 on the GPU path, Class A — no model file and no device.
 *
 * <p>Q4_0 loaded on the CPU and failed on the GPU with {@code "Q4_0 format not supported for
 * TornadoVM yet"}, and a Q4_0 file could not be loaded at all because {@code getModelQuantization}
 * had no case for {@code general.file_type} 2. Both are now handled the way every other
 * GPU-unsupported quantization is: materialized as Q8_0 at load.
 *
 * <p>The tensor is synthesized here rather than read from a fixture — there is no Q4_0 GGUF in the
 * test corpus, and the conversion is exactly the part worth checking numerically.
 */
public class Q4_0GpuLoadTest {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCKS = 3;
    private static final int ELEMENTS = BLOCK_SIZE * BLOCKS;

    /** One Q4_0 block: an fp16 scale, then 16 bytes holding 32 nibbles (low half first). */
    private static MemorySegment synthesizeQ4_0(Arena arena, float[] scalePerBlock) {
        MemorySegment segment = arena.allocate((long) BLOCKS * GGMLType.Q4_0.getTypeSize());
        for (int block = 0; block < BLOCKS; block++) {
            long base = (long) block * GGMLType.Q4_0.getTypeSize();
            segment.set(ValueLayout.JAVA_SHORT, base, floatToFloat16(scalePerBlock[block]));
            for (int i = 0; i < BLOCK_SIZE / 2; i++) {
                int low = (block + i) % 16; // quant for element i
                int high = (block + i + 7) % 16; // quant for element i + 16
                segment.set(
                        ValueLayout.JAVA_BYTE,
                        base + Float16.BYTES + i,
                        (byte) ((high << 4) | low));
            }
        }
        return segment;
    }

    /** What the loader is handed: the tensor data behind TornadoVM's array header. */
    private static GGMLTensorEntry entryWithTornadoHeader(Arena arena, MemorySegment data) {
        long header = TornadoNativeArray.ARRAY_HEADER;
        MemorySegment withHeader = arena.allocate(header + data.byteSize());
        MemorySegment.copy(data, 0, withHeader, header, data.byteSize());
        return new GGMLTensorEntry(
                withHeader, "blk.0.attn_q.weight", GGMLType.Q4_0, new int[] {ELEMENTS}, withHeader);
    }

    @Test
    public void aQ4_0TensorLoadsForTheGpuInsteadOfThrowing() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment data = synthesizeQ4_0(arena, new float[] {0.05f, 0.125f, 0.0075f});
            TornadoTensor loaded =
                    ModelLoader.loadTornadoTensor(entryWithTornadoHeader(arena, data));

            assertTrue(
                    "Q4_0 must materialize as Q8_0 for the device path",
                    loaded instanceof Q8_0TornadoTensor);
            assertEquals(GGMLType.Q8_0, loaded.type());
        }
    }

    /**
     * The materialization must preserve the values, not merely produce a tensor. Q4_0 carries 16
     * levels per block and Q8_0 stores 255, so the re-quantization error is well inside the
     * source's own step size — a mistake in nibble order or scale would blow straight through this
     * bound rather than hiding in it.
     */
    @Test
    public void theMaterializedValuesMatchWhatTheCpuTensorReads() {
        try (Arena arena = Arena.ofConfined()) {
            float[] scales = {0.05f, 0.125f, 0.0075f};
            MemorySegment data = synthesizeQ4_0(arena, scales);

            Q4_0FloatTensor cpu = new Q4_0FloatTensor(ELEMENTS, data);
            Q8_0TornadoTensor gpu =
                    (Q8_0TornadoTensor)
                            ModelLoader.loadTornadoTensor(entryWithTornadoHeader(arena, data));

            for (int i = 0; i < ELEMENTS; i++) {
                float expected = cpu.getFloat(i);
                float actual = readQ8_0(gpu.asByteArray(), i);
                float tolerance = Math.abs(expected) * 0.01f + scales[i / BLOCK_SIZE] * 0.5f;
                assertEquals("element " + i, expected, actual, tolerance);
            }
        }
    }

    /** JDK 21 has no float-to-half intrinsic in the API surface this project targets. */
    private static short floatToFloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;
        if (exponent <= 0) {
            return (short) sign; // underflows to zero; the test uses normals
        }
        return (short) (sign | (exponent << 10) | (mantissa >>> 13));
    }

    /** Reads one value out of a Q8_0 block: [fp16 scale][32 int8 quants]. */
    private static float readQ8_0(ByteArray blocks, int index) {
        int block = index / BLOCK_SIZE;
        int offset = block * 34;
        int scaleBits = (blocks.get(offset) & 0xFF) | ((blocks.get(offset + 1) & 0xFF) << 8);
        float scale = Float.float16ToFloat((short) scaleBits);
        return blocks.get(offset + 2 + (index % BLOCK_SIZE)) * scale;
    }

    /** The device has no 4-bit kernel, so the weight type the plan dispatches on is Q8_0. */
    @Test
    public void theGpuWeightTypeForAQ4_0FileIsQ8_0() {
        assertEquals(GGMLType.Q8_0, AbstractModelLoader.effectiveGpuWeightType(GGMLType.Q4_0));
        assertEquals(GGMLType.Q8_0, AbstractModelLoader.effectiveGpuWeightType(GGMLType.Q4_K));
        assertEquals(
                "a type the device executes is left alone",
                GGMLType.Q8_0,
                AbstractModelLoader.effectiveGpuWeightType(GGMLType.Q8_0));
        assertEquals(GGMLType.F16, AbstractModelLoader.effectiveGpuWeightType(GGMLType.F16));
    }

    /**
     * {@code general.file_type} 2 is Q4_0. Before this change it fell through to {@code
     * UnsupportedOperationException}, so a Q4_0 file could not be loaded on either path — including
     * the CPU one, which has decoded Q4_0 all along.
     */
    @Test
    public void aQ4_0FileTypeIsRecognized() {
        assertEquals(
                "Q8_0", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 2)));
        assertEquals(
                "FP16", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 1)));
        assertEquals(
                "Q8_0", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 7)));
    }

    @Test
    public void anUnknownFileTypeStillFailsLoudly() {
        try {
            AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 99));
            org.junit.Assert.fail("an unknown quantization must not be guessed at");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("99"));
        }
    }
}
