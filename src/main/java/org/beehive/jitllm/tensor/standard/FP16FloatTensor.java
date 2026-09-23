package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

public final class FP16FloatTensor extends FloatTensor {

    /** An FP16 denormal is {@code mantissa * 2^-24}. */
    private static final float DENORMAL_SCALE = 0x1p-24f;

    final int size;
    final MemorySegment memorySegment;

    public FP16FloatTensor(int size, MemorySegment memorySegment) {
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
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return null;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return Float.float16ToFloat(readShort(memorySegment, index * GGMLType.FLOAT16_BYTES));
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(
            FP16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 =
                    ShortVector.fromMemorySegment(
                            S_SPECIES_HALF,
                            thiz.memorySegment,
                            (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES,
                            ByteOrder.LITTLE_ENDIAN);

            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts(); // (int) bits16
            // Does not support infinities nor NaNs, preserves sign. Denormals ARE converted: they
            // are 0.2%-0.7% of the weights in the FP16 checkpoints measured (Granite's w1 is the
            // worst at 0.655%), and flushing them to zero here made the CPU — the parity gate's
            // reference — the inaccurate side. Restoring them takes the CPU↔GPU relative L2 from
            // 4.5e-3 to 6.3e-4 on Llama-3.2-1B and from 1.4e-2 to 5.2e-4 on granite-4.0-1b, i.e.
            // down to the level explained by the GPU's FP16 activation storage alone.
            // Expects well-formed float16 values only (e.g. model weights).
            // Fast Float16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───···───[10]─┬─[9]────····────[0]─┐
            // │ Sign │ Exponent (5 bits) │ Mantissa (10 bits) │ Float16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            //
            // Shifts and adjustments:
            // - Sign:       float16[15] -> float32[31] (shift 16 bits up)
            // - Exponent:   float16[10-14] -> float32[23-30] (+ bias adjustment)
            // - Mantissa:   float16[0-9] -> float32[13-22] (shift 13 bits up)
            //
            // exp = bits32 & 0x7C00
            // zeroExponentMask = exp == 0 ? 0: ~0
            var zeroExponentMask =
                    bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31); // = (-exp) >> 31
            var sign = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16);
            var normal =
                    sign.or(
                            // exponent and mantissa combined
                            bits32.and(0x7FFF)
                                    .add(0x1C000)
                                    .lanewise(VectorOperators.LSHL, 13)
                                    .and(zeroExponentMask) // -0 and +0 fall out here as zero
                            );

            FloatVector thizVector = normal.reinterpretAsFloats(); // Float.intBitsToFloat(vi)

            // Denormals (exponent 0, mantissa != 0) are not representable by the shift above, which
            // is what produced the zero. Their value is mantissa * 2^-24, so convert the mantissa
            // as an integer and scale. Blended in rather than branched on: the mask is false for
            // almost every lane, and a branch would cost more than the two extra ops.
            var mantissa = bits32.and(0x03FF);
            var isDenormal = bits32.and(0x7C00).eq(0).and(mantissa.eq(0).not());
            if (isDenormal.anyTrue()) {
                FloatVector magnitude = (FloatVector) mantissa.convert(VectorOperators.I2F, 0);
                var signed = magnitude.mul(DENORMAL_SCALE).reinterpretAsInts().or(sign);
                thizVector =
                        thizVector.blend(signed.reinterpretAsFloats(), isDenormal.cast(F_SPECIES));
            }
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result +=
                    scalarDot(
                            thiz,
                            thisOffset + upperBound,
                            that,
                            thatOffset + upperBound,
                            size - upperBound);
        }

        return result;
    }
}
