package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#IQ4_XS} format: 256 values to a block in 136
 * bytes, in eight groups of 32.
 *
 * <pre>
 *   offset 0   d         (fp16)   the block's scale
 *   offset 2   scales_h  (16 b)   the high two bits of each group's 6-bit scale, group g at bit 2g
 *   offset 4   scales_l  (4 B)    the low four bits of the group scales, group g in byte g/2
 *   offset 8   qs        (128 B)  16 bytes per group: values 0..15 in the low nibbles, 16..31 high
 * </pre>
 *
 * <p>A value is {@code d * (ls - 32) * KVALUES[q]}, with {@code ls} its group's scale and {@code q}
 * its nibble. {@code KVALUES} is ggml's non-linear 4-bit table ({@code kvalues_iq4nl}).
 *
 * <p>Present because quantizers mix it into K-quant files: {@code Qwen3-0.6B-UD-Q4_K_XL.gguf} holds
 * five layers' q, k, gate and up projections as IQ4_XS.
 */
public final class IQ4_XSFloatTensor extends FloatTensor {

    /** ggml's {@code kvalues_iq4nl}. */
    public static final byte[] KVALUES = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };

    private static final int BLOCK_SIZE = GGMLType.IQ4_XS.getBlockSize();
    private static final int TYPE_SIZE = GGMLType.IQ4_XS.getTypeSize();
    private static final int GROUP = 32;
    private static final int SCALES_H_OFFSET = 2;
    private static final int SCALES_L_OFFSET = 4;
    private static final int QS_OFFSET = 8;

    final int size;
    final MemorySegment memorySegment;

    public IQ4_XSFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.IQ4_XS;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    /** Group {@code g}'s scale, {@code d * (ls - 32)}. */
    private float groupScale(int blockOffset, float d, int g) {
        int scalesH = readShort(memorySegment, blockOffset + SCALES_H_OFFSET) & 0xFFFF;
        int low = (readByte(memorySegment, blockOffset + SCALES_L_OFFSET + g / 2) & 0xFF);
        int ls = ((low >>> (4 * (g % 2))) & 0xF) | (((scalesH >>> (2 * g)) & 3) << 4);
        return d * (ls - 32);
    }

    /** Value {@code within} (0..31) of group {@code g}, with the group's scale {@code dl}. */
    private float decode(int blockOffset, float dl, int g, int within) {
        int half = within / 16;
        int packed =
                readByte(memorySegment, blockOffset + QS_OFFSET + g * 16 + within - half * 16)
                        & 0xFF;
        return dl * KVALUES[(packed >>> (half * 4)) & 0xF];
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockOffset = (index / BLOCK_SIZE) * TYPE_SIZE;
        int withinBlock = index % BLOCK_SIZE;
        int g = withinBlock / GROUP;
        float d = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return decode(blockOffset, groupScale(blockOffset, d, g), g, withinBlock % GROUP);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        int blockOffset = (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        for (; j + BLOCK_SIZE <= size; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = Float.float16ToFloat(readShort(memorySegment, blockOffset));
            for (int g = 0; g < BLOCK_SIZE / GROUP; g++) {
                float dl = groupScale(blockOffset, d, g);
                float group = 0f;
                for (int i = 0; i < GROUP; i++) {
                    group +=
                            decode(blockOffset, dl, g, i)
                                    * that.getFloat(thatOffset + j + g * GROUP + i);
                }
                result += group;
            }
        }
        if (j < size) {
            result += FloatTensor.scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }
}
