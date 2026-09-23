package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_K} format.
 *
 * <p>Q4_K uses super-blocks of 256 elements, each containing:
 *
 * <ul>
 *   <li>2 bytes: d (super-block scale, fp16)
 *   <li>2 bytes: dmin (super-block min, fp16)
 *   <li>12 bytes: scales/mins for 8 sub-blocks (packed 6-bit values)
 *   <li>128 bytes: 4-bit quantized values
 * </ul>
 */
public final class Q4_KFloatTensor extends FloatTensor {

    private static final int QK_K = 256;
    private static final int BLOCK_SIZE = GGMLType.Q4_K.getTypeSize(); // 144

    // Offsets within a block
    private static final int D_OFFSET = 0;
    private static final int DMIN_OFFSET = 2;
    private static final int SCALES_OFFSET = 4;
    private static final int QS_OFFSET = 16; // 4 + 12

    final int size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    /**
     * Unpacks the 6-bit scale value for a given sub-block index. The 12 scale bytes encode 8
     * scale/min pairs in a packed format.
     */
    private static int getScaleK4(int j, MemorySegment ms, long scalesOffset) {
        if (j < 4) {
            return Byte.toUnsignedInt(readByte(ms, scalesOffset + j)) & 63;
        } else {
            return (Byte.toUnsignedInt(readByte(ms, scalesOffset + j + 4)) & 0xF)
                    | ((Byte.toUnsignedInt(readByte(ms, scalesOffset + j - 4)) >> 6) << 4);
        }
    }

    /** Unpacks the 6-bit min value for a given sub-block index. */
    private static int getMinK4(int j, MemorySegment ms, long scalesOffset) {
        if (j < 4) {
            return Byte.toUnsignedInt(readByte(ms, scalesOffset + j + 4)) & 63;
        } else {
            return (Byte.toUnsignedInt(readByte(ms, scalesOffset + j + 4)) >> 4)
                    | ((Byte.toUnsignedInt(readByte(ms, scalesOffset + j)) >> 6) << 4);
        }
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / QK_K;
        int withinBlock = index % QK_K;
        long blockOffset = (long) blockIndex * BLOCK_SIZE;

        float d = Float.float16ToFloat(readShort(memorySegment, blockOffset + D_OFFSET));
        float dmin = Float.float16ToFloat(readShort(memorySegment, blockOffset + DMIN_OFFSET));
        long scalesOff = blockOffset + SCALES_OFFSET;

        // Each group of 64 elements uses 2 sub-blocks (low nibble / high nibble)
        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock % 64; // 0..63

        int subBlock;
        int q;
        if (posInPair < 32) {
            subBlock = pairIndex * 2;
            byte qByte =
                    readByte(memorySegment, blockOffset + QS_OFFSET + pairIndex * 32 + posInPair);
            q = Byte.toUnsignedInt(qByte) & 0xF;
        } else {
            subBlock = pairIndex * 2 + 1;
            byte qByte =
                    readByte(
                            memorySegment,
                            blockOffset + QS_OFFSET + pairIndex * 32 + (posInPair - 32));
            q = (Byte.toUnsignedInt(qByte) >> 4) & 0xF;
        }

        int sc = getScaleK4(subBlock, memorySegment, scalesOff);
        int m = getMinK4(subBlock, memorySegment, scalesOff);

        return d * sc * q - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }
}
