package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q6_K} format.
 *
 * <p>Q6_K uses super-blocks of 256 elements, each containing:
 *
 * <ul>
 *   <li>128 bytes: ql (lower 4 bits of 6-bit quants)
 *   <li>64 bytes: qh (upper 2 bits of 6-bit quants)
 *   <li>16 bytes: scales (signed 8-bit per 16-element sub-block)
 *   <li>2 bytes: d (super-block scale, fp16)
 * </ul>
 */
public final class Q6_KFloatTensor extends FloatTensor {

    private static final int QK_K = 256;
    private static final int BLOCK_SIZE = GGMLType.Q6_K.getTypeSize(); // 210

    // Offsets within a block
    private static final int QL_OFFSET = 0; // 128 bytes
    private static final int QH_OFFSET = 128; // 64 bytes
    private static final int SCALES_OFFSET = 192; // 16 bytes
    private static final int D_OFFSET = 208; // 2 bytes

    final int size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q6_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / QK_K;
        int withinBlock = index % QK_K;
        long blockOffset = (long) blockIndex * BLOCK_SIZE;

        float d = Float.float16ToFloat(readShort(memorySegment, blockOffset + D_OFFSET));

        // The block is split into two halves of 128 elements each
        int halfIndex = withinBlock / 128; // 0 or 1
        int posInHalf = withinBlock % 128; // 0..127

        // Within each half, there are 4 groups of 32 elements
        int groupInHalf = posInHalf / 32; // 0..3
        int posInGroup = posInHalf % 32; // 0..31

        // ql/qh pointers advance by 64/32 per half
        long qlBase = blockOffset + QL_OFFSET + halfIndex * 64;
        long qhBase = blockOffset + QH_OFFSET + halfIndex * 32;
        long scBase = blockOffset + SCALES_OFFSET + halfIndex * 8;

        // Scale index: is = posInGroup / 16 (0 or 1), then offset by group
        int is = posInGroup / 16;

        int qValue;
        switch (groupInHalf) {
            case 0 -> {
                int ql = Byte.toUnsignedInt(readByte(memorySegment, qlBase + posInGroup));
                int qh = Byte.toUnsignedInt(readByte(memorySegment, qhBase + posInGroup));
                qValue = ((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
                return d * (byte) readByte(memorySegment, scBase + is) * qValue;
            }
            case 1 -> {
                int ql = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + posInGroup));
                int qh = Byte.toUnsignedInt(readByte(memorySegment, qhBase + posInGroup));
                qValue = ((ql & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
                return d * (byte) readByte(memorySegment, scBase + is + 2) * qValue;
            }
            case 2 -> {
                int ql = Byte.toUnsignedInt(readByte(memorySegment, qlBase + posInGroup));
                int qh = Byte.toUnsignedInt(readByte(memorySegment, qhBase + posInGroup));
                qValue = ((ql >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                return d * (byte) readByte(memorySegment, scBase + is + 4) * qValue;
            }
            case 3 -> {
                int ql = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + posInGroup));
                int qh = Byte.toUnsignedInt(readByte(memorySegment, qhBase + posInGroup));
                qValue = ((ql >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                return d * (byte) readByte(memorySegment, scBase + is + 6) * qValue;
            }
            default -> throw new AssertionError();
        }
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }
}
