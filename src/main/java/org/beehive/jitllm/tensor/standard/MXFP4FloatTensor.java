package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

/**
 * {@link FloatTensor} in the {@link GGMLType#MXFP4} format: blocks of 32 values, each block one
 * E8M0 exponent byte followed by 16 bytes of E2M1 nibbles (element {@code j} in the low nibble of
 * byte {@code j}, element {@code j + 16} in its high nibble).
 *
 * <p>Decoded as llama.cpp decodes it: the nibble indexes {@link #KVALUES}, which holds twice the
 * E2M1 value so it fits a byte, and the exponent is applied halved, {@code 2^(e - 128)}.
 */
public final class MXFP4FloatTensor extends FloatTensor {

    /** Twice the E2M1 value of each nibble. */
    public static final byte[] KVALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    private static final int BLOCK = 32;
    private static final int BLOCK_BYTES = 17;

    private static final float[] KVALUES_FLOAT = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };

    final int size;
    final MemorySegment memorySegment;

    public MXFP4FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    /** {@code 2^(e - 128)}: the E8M0 scale, halved to match {@link #KVALUES}. */
    public static float scale(int e) {
        int bits = e < 2 ? 0x00200000 << e : (e - 1) << 23;
        return Float.intBitsToFloat(bits);
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
        return GGMLType.MXFP4;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        long blockOffset = (long) (index / BLOCK) * BLOCK_BYTES;
        float scale = scale(Byte.toUnsignedInt(readByte(memorySegment, blockOffset)));
        int j = index % BLOCK;
        int packed = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 1 + (j & 15)));
        int nibble = j < 16 ? packed & 0x0F : packed >>> 4;
        return KVALUES[nibble] * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API
                && supportsVectorizedDot(F_SPECIES)
                && that instanceof ArrayFloatTensor array
                && thisOffset % BLOCK == 0) {
            return vectorDot(this, thisOffset, array, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /**
     * Decodes each block into a 32-float scratch with the table, then multiplies it with the vector
     * API. The decode is a scalar table lookup rather than a {@code VectorShuffle}: shuffles fail
     * to build in a native image with shared-arena support.
     */
    private static float vectorDot(
            MXFP4FloatTensor thiz,
            int thisOffset,
            ArrayFloatTensor that,
            int thatOffset,
            int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = size / BLOCK * BLOCK;
        long blockOffset = (long) (thisOffset / BLOCK) * BLOCK_BYTES;
        int lanes = F_SPECIES.length();
        float[] w = new float[BLOCK];
        int j = 0;
        for (; j < upperBound; j += BLOCK, blockOffset += BLOCK_BYTES) {
            float scale = scale(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));
            for (int k = 0; k < BLOCK / 2; k++) {
                int q = Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset + 1 + k));
                w[k] = KVALUES_FLOAT[q & 0x0F];
                w[k + BLOCK / 2] = KVALUES_FLOAT[q >>> 4];
            }
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int part = 0; part < BLOCK; part += lanes) {
                acc =
                        that.getFloatVector(F_SPECIES, thatOffset + j + part)
                                .fma(FloatVector.fromArray(F_SPECIES, w, part), acc);
            }
            val = acc.fma(FloatVector.broadcast(F_SPECIES, scale), val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }
}
