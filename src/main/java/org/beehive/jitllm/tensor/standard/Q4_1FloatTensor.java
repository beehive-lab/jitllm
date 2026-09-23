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
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_1} format.
 *
 * <p>The affine sibling of {@link Q4_0FloatTensor}: 32 values to a block, four-bit unsigned
 * quantities, and two half-precision block parameters rather than one. Where Q4_0 recenters the
 * nibble by subtracting eight and multiplies by a scale, Q4_1 stores a scale <i>and</i> a minimum
 * and reconstructs {@code d * nibble + m}, so the nibble is never biased and the block need not be
 * symmetric about zero.
 *
 * <p>Present because the `qwen35` architecture mixes it in: `Qwen3.8-27B-Q4_0.gguf` is Q4_0
 * throughout except for the first eight layers' {@code ffn_down}, which the quantizer emitted as
 * Q4_1.
 *
 * <p>The dot product distributes over the affine form — {@code Σ xᵢ(d·qᵢ + m) = d·Σ xᵢqᵢ + m·Σ xᵢ}
 * — which is why the vectorized loop carries the running products and the running activations and
 * scales each by its own block parameter, rather than reconstructing weights.
 */
public final class Q4_1FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = GGMLType.Q4_1.getBlockSize();
    private static final int TYPE_SIZE = GGMLType.Q4_1.getTypeSize();

    /** Both block parameters precede the packed nibbles: {@code d} at 0, {@code m} at 2. */
    private static final int NIBBLES_OFFSET = 2 * Float16.BYTES;

    final int size;
    final MemorySegment memorySegment;

    public Q4_1FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_1;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockOffset = (index / BLOCK_SIZE) * TYPE_SIZE;
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        float min = Float.float16ToFloat(readShort(memorySegment, blockOffset + Float16.BYTES));

        int modIndex = index % BLOCK_SIZE;
        int nibble;
        if (modIndex < BLOCK_SIZE / 2) {
            nibble = readByte(memorySegment, blockOffset + NIBBLES_OFFSET + modIndex) & 0x0F;
        } else {
            nibble =
                    (readByte(
                                            memorySegment,
                                            blockOffset
                                                    + NIBBLES_OFFSET
                                                    + modIndex
                                                    - BLOCK_SIZE / 2)
                                    >>> 4)
                            & 0x0F;
        }
        return nibble * scale + min;
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
            Q4_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to a block boundary; the loop below reads whole blocks.
        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % BLOCK_SIZE == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = size / BLOCK_SIZE * BLOCK_SIZE;
        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            var wScale =
                    FloatVector.broadcast(
                            F_SPECIES,
                            Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset)));
            var wMin =
                    FloatVector.broadcast(
                            F_SPECIES,
                            Float.float16ToFloat(
                                    readShort(thiz.memorySegment, blockOffset + Float16.BYTES)));

            var B_SPECIES = ByteVector.SPECIES_128;
            var wBytes =
                    ByteVector.fromMemorySegment(
                            B_SPECIES,
                            thiz.memorySegment,
                            blockOffset + NIBBLES_OFFSET,
                            ByteOrder.LITTLE_ENDIAN);
            // Unsigned nibbles: no recentering, which is the whole difference from Q4_0.
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF);

            if (F_SPECIES.vectorBitSize() == 256) {
                var x0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length());
                var x1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length());
                var x2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length());
                var x3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length());
                var products =
                        x0.mul(loBytes.castShape(F_SPECIES, 0))
                                .add(x1.mul(loBytes.castShape(F_SPECIES, 1)))
                                .add(x2.mul(hiBytes.castShape(F_SPECIES, 0)))
                                .add(x3.mul(hiBytes.castShape(F_SPECIES, 1)));
                var activations = x0.add(x1).add(x2).add(x3);
                val = products.fma(wScale, val);
                val = activations.fma(wMin, val);
            } else if (F_SPECIES.vectorBitSize() == 128) {
                for (int i = 0; i < 2; ++i) {
                    var tmp = i == 0 ? loBytes : hiBytes;
                    var x0 =
                            that.getFloatVector(
                                    F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length());
                    var x1 =
                            that.getFloatVector(
                                    F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length());
                    var x2 =
                            that.getFloatVector(
                                    F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length());
                    var x3 =
                            that.getFloatVector(
                                    F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length());
                    var products =
                            x0.mul(tmp.castShape(F_SPECIES, 0))
                                    .add(x1.mul(tmp.castShape(F_SPECIES, 1)))
                                    .add(x2.mul(tmp.castShape(F_SPECIES, 2)))
                                    .add(x3.mul(tmp.castShape(F_SPECIES, 3)));
                    var activations = x0.add(x1).add(x2).add(x3);
                    val = products.fma(wScale, val);
                    val = activations.fma(wMin, val);
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
