package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.Float16;
import org.beehive.jitllm.format.GGMLType;

public final class Q8_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    public MemorySegment getMemorySegment() {
        return memorySegment;
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
        return GGMLType.Q8_0;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE =
            ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    /**
     * When enabled, quantize the activation to Q8_0 and do an int8·int8 dot, matching llama.cpp's
     * Q8_0 matmul quantization scheme.
     *
     * <p><b>Opt-in, because it is slower and less accurate than the default path.</b> It is kept
     * for comparing this implementation's numerics against llama.cpp's, which is what it is good
     * for.
     *
     * <p>Measured on Llama-3.2-1B-Instruct-Q8_0, CPU, 128 tokens, 5 interleaved runs each: <b>32.86
     * tok/s off, 7.54 tok/s on</b> — 4.4x slower. {@link #dotQ8Activation} is a scalar loop and
     * returns before the {@code USE_VECTOR_API} branch below, so enabling it trades the SIMD float
     * dot for a scalar int8 one; quantizing to int8 does cut arithmetic, but not by enough to pay
     * for losing vectorization. A {@code ByteVector} implementation could plausibly win — that is
     * why llama.cpp does this — but this is not one.
     *
     * <p>It also breaks CPU/GPU parity, since the GPU path does not quantize activations: {@code
     * CpuGpuParity} reports 64.08% of elements outside budget with it on and 0% with it off. The
     * output stays semantically intact (0/64 argmax disagreements), so this is a numerics
     * divergence, not a defect — but it is one the default should not carry.
     */
    static final boolean QUANTIZE_ACTIVATION =
            Boolean.parseBoolean(System.getProperty("jitllm.quantizeActivation", "false"));

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (QUANTIZE_ACTIVATION) {
            return dotQ8Activation(thisOffset, that, thatOffset, size);
        }
        if (USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    /**
     * Q8_0 weight · activation, where the activation is first quantized to Q8_0 (per 32-element
     * block) and the dot is accumulated as int8·int8 -> int32, then scaled. This mirrors
     * llama.cpp's ggml Q8_0 matmul path. Assumes thisOffset and size are multiples of the Q8_0
     * block size (32), which holds for all matmul callers.
     */
    private float dotQ8Activation(int thisOffset, FloatTensor that, int thatOffset, int size) {
        final int BS = GGMLType.Q8_0.getBlockSize(); // 32
        final int TS = GGMLType.Q8_0.getTypeSize(); // 34
        float result = 0f;
        int nBlocks = size / BS;
        for (int b = 0; b < nBlocks; b++) {
            int elemBase = b * BS;
            int wBlockOffset = (thisOffset + elemBase) / BS * TS;
            float wScale = Float.float16ToFloat(readShort(memorySegment, wBlockOffset));

            // find the max abs of this activation block -> activation scale
            float amax = 0f;
            for (int i = 0; i < BS; i++) {
                float av = Math.abs(that.getFloat(thatOffset + elemBase + i));
                if (av > amax) amax = av;
            }
            // Match ggml's Q8_0 quantization order: derive the int8 values using the
            // full-precision scale, but store/use the scale itself as f16.
            float quantizationScale = amax / 127f;
            float aScale = Float.float16ToFloat(Float.floatToFloat16(quantizationScale));
            float aInv = quantizationScale != 0f ? 1f / quantizationScale : 0f;

            // int8 · int8 accumulation (round-half-away-from-zero, matching ggml roundf)
            int isum = 0;
            for (int i = 0; i < BS; i++) {
                float s = that.getFloat(thatOffset + elemBase + i) * aInv;
                int aq = (int) (s + Math.copySign(0.5f, s));
                byte wq = readByte(memorySegment, wBlockOffset + Float16.BYTES + i);
                isum += aq * wq;
            }
            result += isum * (wScale * aScale);
        }
        return result;
    }

    private static float vectorDot(
            Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset =
                (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (;
                j < upperBound;
                j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            if (F_SPECIES.vectorBitSize() == 256) {
                var wBytes =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_256,
                                thiz.memorySegment,
                                blockOffset + Float16.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var sum0 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 0));
                var sum1 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 1));
                var sum2 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 2));
                var sum3 =
                        that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 3));
                val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
            } else if (F_SPECIES.vectorBitSize() == 128) {
                VectorSpecies<Byte> B_128 = ByteVector.SPECIES_128;
                // This loop cannot be unrolled, why?
                for (int i = 0; i < 2; ++i) {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    B_128,
                                    thiz.memorySegment,
                                    blockOffset + Float16.BYTES + i * B_128.vectorByteSize(),
                                    ByteOrder.LITTLE_ENDIAN);
                    var sum0 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + i * 16 + 0 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + i * 16 + 1 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + i * 16 + 2 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 =
                            that.getFloatVector(
                                            F_SPECIES,
                                            thatOffset + j + i * 16 + 3 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 3));
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
