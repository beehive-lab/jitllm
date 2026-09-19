package org.beehive.jllm.runtime.tensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A read-only tensor whose element count exceeds what an {@code int} can index.
 *
 * <p>The engine's {@code FloatTensor} and the device array types are int-indexed, which caps them
 * at 2^31 elements. Gemma-4's per-layer token embedding table is about 2.35 billion elements, so it
 * cannot be wrapped in either. It also never needs to be: only one embedding row is read per token.
 *
 * <p>This type is that case and only that case — long-indexed, element-at-a-time, no bulk
 * materialization. It exists so weight sets can hold such a tensor without holding a {@code
 * GGMLTensorEntry}, which would put GGUF's vocabulary in the runtime (Rule 4). The dequantization
 * lives here, keyed on {@link DataType}, rather than on the file's type.
 *
 * <p>Not a general tensor abstraction, and deliberately not on the {@code TensorDescriptor} path:
 * it describes storage the runtime reads, not a tensor a backend allocates.
 */
public final class LongIndexedTensor {

    private final MemorySegment data;
    private final DataType dataType;

    public LongIndexedTensor(MemorySegment data, DataType dataType) {
        this.data = data;
        this.dataType = dataType;
    }

    /** How the values are represented. */
    public DataType dataType() {
        return dataType;
    }

    /**
     * The value at an absolute element index, converted to float.
     *
     * <p>Element-at-a-time on purpose: callers read a single row out of a tensor far too large to
     * copy, so there is nothing to amortize a bulk path over.
     *
     * @param elementIndex index into the flattened, row-major tensor
     */
    public float valueAt(long elementIndex) {
        return switch (dataType) {
            case F32 -> data.get(ValueLayout.JAVA_FLOAT_UNALIGNED, elementIndex * Float.BYTES);
            case F16 ->
                    Float.float16ToFloat(
                            data.get(ValueLayout.JAVA_SHORT_UNALIGNED, elementIndex * Short.BYTES));
                // BF16 is the top 16 bits of the F32 bit pattern.
            case BF16 ->
                    Float.intBitsToFloat(
                            ((int)
                                            data.get(
                                                    ValueLayout.JAVA_SHORT_UNALIGNED,
                                                    elementIndex * Short.BYTES))
                                    << 16);
            case Q8_0 -> q8_0ValueAt(elementIndex);
            case Q5_K -> q5_KValueAt(elementIndex);
            default ->
                    throw new UnsupportedOperationException(
                            "LongIndexedTensor does not read "
                                    + dataType
                                    + "; it is only used for embedding"
                                    + " tables, which are never stored in the format-decoded types");
        };
    }

    /**
     * Q5_K: super-blocks of 256 values -- an FP16 scale and minimum, eight 6-bit sub-block
     * scale/minimum pairs packed into 12 bytes, one high bit per value, and a low nibble per value.
     *
     * <p>The arithmetic is {@code Q5_KFloatTensor}'s, which is the accepted host decoder for this
     * format; only the indexing widens, because this table has more elements than an {@code int}
     * can address. A Gemma 4 Q4_0 file stores {@code per_layer_token_embd} this way, so without it
     * the file cannot be read at all -- on either path, since both gather its rows from here.
     */
    private float q5_KValueAt(long elementIndex) {
        final int superBlock = 256;
        final int blockBytes = 176; // 2 d + 2 dmin + 12 scales + 32 qh + 128 qs
        final int dminOffset = 2;
        final int scalesOffset = 4;
        final int qhOffset = 16;
        final int qsOffset = 48;

        long blockOffset = (elementIndex / superBlock) * blockBytes;
        int withinBlock = (int) (elementIndex % superBlock);

        float d = Float.float16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, blockOffset));
        float dmin =
                Float.float16ToFloat(
                        data.get(ValueLayout.JAVA_SHORT_UNALIGNED, blockOffset + dminOffset));
        long scalesOff = blockOffset + scalesOffset;

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock % 64; // 0..63

        int subBlock;
        int q;
        int highBit;
        if (posInPair < 32) {
            subBlock = pairIndex * 2;
            int qsByte =
                    Byte.toUnsignedInt(
                            data.get(
                                    ValueLayout.JAVA_BYTE,
                                    blockOffset + qsOffset + (long) pairIndex * 32 + posInPair));
            q = qsByte & 0xF;
            int qhByte =
                    Byte.toUnsignedInt(
                            data.get(ValueLayout.JAVA_BYTE, blockOffset + qhOffset + posInPair));
            highBit = (qhByte >> (pairIndex * 2)) & 1;
        } else {
            subBlock = pairIndex * 2 + 1;
            int qsByte =
                    Byte.toUnsignedInt(
                            data.get(
                                    ValueLayout.JAVA_BYTE,
                                    blockOffset
                                            + qsOffset
                                            + (long) pairIndex * 32
                                            + (posInPair - 32)));
            q = (qsByte >> 4) & 0xF;
            int qhByte =
                    Byte.toUnsignedInt(
                            data.get(
                                    ValueLayout.JAVA_BYTE,
                                    blockOffset + qhOffset + (posInPair - 32)));
            highBit = (qhByte >> (pairIndex * 2 + 1)) & 1;
        }
        q += highBit * 16;

        return d * scaleK4(subBlock, scalesOff) * q - dmin * minK4(subBlock, scalesOff);
    }

    private int scaleK4(int j, long scalesOffset) {
        if (j < 4) {
            return Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j)) & 63;
        }
        return (Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j + 4)) & 0xF)
                | ((Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j - 4)) >> 6)
                        << 4);
    }

    private int minK4(int j, long scalesOffset) {
        if (j < 4) {
            return Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j + 4)) & 63;
        }
        return (Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j + 4)) >> 4)
                | ((Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, scalesOffset + j)) >> 6)
                        << 4);
    }

    /** Q8_0: 32 signed 8-bit values behind one FP16 scale, blocks tiling the row-major data. */
    private float q8_0ValueAt(long elementIndex) {
        final int blockSize = 32;
        final int blockBytes = Short.BYTES + blockSize; // FP16 scale + 32 quants
        long blockOffset = (elementIndex / blockSize) * blockBytes;
        int withinBlock = (int) (elementIndex % blockSize);
        float scale = Float.float16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, blockOffset));
        byte quant = data.get(ValueLayout.JAVA_BYTE, blockOffset + Short.BYTES + withinBlock);
        return scale * quant;
    }
}
