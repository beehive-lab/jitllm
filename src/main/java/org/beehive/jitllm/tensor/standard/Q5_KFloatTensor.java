package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q5_K} format.
 *
 * <p>Q5_K uses super-blocks of 256 elements, each containing:
 *
 * <ul>
 *   <li>2 bytes: d (super-block scale, fp16)
 *   <li>2 bytes: dmin (super-block min, fp16)
 *   <li>12 bytes: scales/mins for 8 sub-blocks (packed 6-bit values, same as Q4_K)
 *   <li>32 bytes: qh (5th bit of each quant)
 *   <li>128 bytes: qs (lower 4 bits of quants)
 * </ul>
 */
public final class Q5_KFloatTensor extends FloatTensor {

    private static final int QK_K = 256;
    private static final int BLOCK_SIZE = GGMLType.Q5_K.getTypeSize(); // 176

    // Offsets within a block
    private static final int D_OFFSET = 0;
    private static final int DMIN_OFFSET = 2;
    private static final int SCALES_OFFSET = 4;
    private static final int QH_OFFSET = 16; // 32 bytes for 5th bit
    private static final int QS_OFFSET = 48; // 128 bytes for lower 4 bits

    final int size;
    final MemorySegment memorySegment;

    public Q5_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q5_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    private static int getScaleK4(int j, MemorySegment ms, long scalesOffset) {
        if (j < 4) {
            return Byte.toUnsignedInt(readByte(ms, scalesOffset + j)) & 63;
        } else {
            return (Byte.toUnsignedInt(readByte(ms, scalesOffset + j + 4)) & 0xF)
                    | ((Byte.toUnsignedInt(readByte(ms, scalesOffset + j - 4)) >> 6) << 4);
        }
    }

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

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock % 64; // 0..63

        int subBlock;
        int q;
        int highBit;
        if (posInPair < 32) {
            subBlock = pairIndex * 2;
            byte qsByte =
                    readByte(memorySegment, blockOffset + QS_OFFSET + pairIndex * 32 + posInPair);
            q = Byte.toUnsignedInt(qsByte) & 0xF;
            // 5th bit from qh: bit position is (pairIndex * 2) for low nibble elements
            byte qhByte = readByte(memorySegment, blockOffset + QH_OFFSET + posInPair);
            highBit = (Byte.toUnsignedInt(qhByte) >> (pairIndex * 2)) & 1;
        } else {
            subBlock = pairIndex * 2 + 1;
            byte qsByte =
                    readByte(
                            memorySegment,
                            blockOffset + QS_OFFSET + pairIndex * 32 + (posInPair - 32));
            q = (Byte.toUnsignedInt(qsByte) >> 4) & 0xF;
            // 5th bit from qh: bit position is (pairIndex * 2 + 1) for high nibble elements
            byte qhByte = readByte(memorySegment, blockOffset + QH_OFFSET + (posInPair - 32));
            highBit = (Byte.toUnsignedInt(qhByte) >> (pairIndex * 2 + 1)) & 1;
        }

        q += highBit * 16; // Add the 5th bit

        int sc = getScaleK4(subBlock, memorySegment, scalesOff);
        int m = getMinK4(subBlock, memorySegment, scalesOff);

        return d * sc * q - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }
}
