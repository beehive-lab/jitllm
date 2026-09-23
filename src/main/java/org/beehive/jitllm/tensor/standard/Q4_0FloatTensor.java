package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.JllmApp;
import org.beehive.jllm.format.Float16;
import org.beehive.jllm.format.GGMLType;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_0} format.
 *
 * <p>This tensor implementation is not compatible with {@link FloatTensor}, but {@link #dot(int,
 * FloatTensor, int, int)} has a vectorized implementation that is used when the second argument
 * implements {@link FloatTensor}.
 */
public final class Q4_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_0;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant;
        int modIndex = index % GGMLType.Q4_0.getBlockSize();
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant =
                    (byte)
                            ((readByte(
                                                    memorySegment,
                                                    blockOffset
                                                            + Float16.BYTES
                                                            + modIndex
                                                            - GGMLType.Q4_0.getBlockSize() / 2)
                                            >>> 4)
                                    & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (JllmApp.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(
            Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset =
                (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = size / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (;
                j < upperBound;
                j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var B_SPECIES = ByteVector.SPECIES_128;
            var wBytes =
                    ByteVector.fromMemorySegment(
                            B_SPECIES,
                            thiz.memorySegment,
                            blockOffset + Float16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            if (F_SPECIES.vectorBitSize() == 256) {
                var sum0 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                                .mul(loBytes.castShape(F_SPECIES, 0));
                var sum1 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                                .mul(loBytes.castShape(F_SPECIES, 1));
                var sum2 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                                .mul(hiBytes.castShape(F_SPECIES, 0));
                var sum3 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                                .mul(hiBytes.castShape(F_SPECIES, 1));
                val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
            } else if (F_SPECIES.vectorBitSize() == 128) {
                // This loop cannot be unrolled, why?
                for (int i = 0; i < 2; ++i) {
                    var tmp = i == 0 ? loBytes : hiBytes;
                    var sum0 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + (i * 4 + 0) * F_SPECIES.length())
                                    .mul(tmp.castShape(F_SPECIES, 0));
                    var sum1 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                                    .mul(tmp.castShape(F_SPECIES, 1));
                    var sum2 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                                    .mul(tmp.castShape(F_SPECIES, 2));
                    var sum3 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                                    .mul(tmp.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
            } else {
                throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
