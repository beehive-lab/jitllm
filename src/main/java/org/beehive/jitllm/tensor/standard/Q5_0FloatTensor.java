package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.Float16;
import org.beehive.jitllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q5_0} format: 32 values to a block, an fp16
 * scale {@code d}, the fifth bit of every value in a 32-bit word {@code qh}, and the low four bits
 * as nibbles (values 0..15 low, 16..31 high). A value is {@code d * (q - 16)}.
 *
 * <p>Present because quantizers mix it into K-quant files: {@code
 * Qwen2.5-0.5B-Instruct-Q4_K_M.gguf} holds its q, k, output, gate and up projections as Q5_0.
 *
 * <p>The dot product decodes one block at a time into a small array and multiplies it, rather than
 * decoding each value through {@link #getFloat}, which would read the block's scale and fifth bits
 * 32 times.
 */
public final class Q5_0FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = GGMLType.Q5_0.getBlockSize();
    private static final int TYPE_SIZE = GGMLType.Q5_0.getTypeSize();

    /** The fifth bits follow the scale. */
    private static final int QH_OFFSET = Float16.BYTES;

    /** The packed nibbles follow the fifth bits. */
    private static final int NIBBLES_OFFSET = QH_OFFSET + Integer.BYTES;

    final int size;
    final MemorySegment memorySegment;

    public Q5_0FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q5_0;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    private int highBits(int blockOffset) {
        return (readByte(memorySegment, blockOffset + QH_OFFSET) & 0xFF)
                | ((readByte(memorySegment, blockOffset + QH_OFFSET + 1) & 0xFF) << 8)
                | ((readByte(memorySegment, blockOffset + QH_OFFSET + 2) & 0xFF) << 16)
                | ((readByte(memorySegment, blockOffset + QH_OFFSET + 3) & 0xFF) << 24);
    }

    /** Value {@code withinBlock} of the block at {@code blockOffset}. */
    private float decode(int blockOffset, float scale, int qh, int withinBlock) {
        int half = withinBlock / (BLOCK_SIZE / 2);
        int packed =
                readByte(memorySegment, blockOffset + NIBBLES_OFFSET + withinBlock - half * 16)
                        & 0xFF;
        int q = ((packed >>> (half * 4)) & 0x0F) | (((qh >>> withinBlock) & 1) << 4);
        return scale * (q - 16);
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockOffset = (index / BLOCK_SIZE) * TYPE_SIZE;
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return decode(blockOffset, scale, highBits(blockOffset), index % BLOCK_SIZE);
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
            float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
            int qh = highBits(blockOffset);
            float block = 0f;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                block += decode(blockOffset, scale, qh, i) * that.getFloat(thatOffset + j + i);
            }
            result += block;
        }
        if (j < size) {
            result += FloatTensor.scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }
}
