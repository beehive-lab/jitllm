package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.beehive.jitllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Materializes GGUF tensors as this backend's device tensors.
 *
 * <p>Nothing here is a new abstraction. The methods are the ones the loaders already called, with
 * their bodies unchanged; what moved is which package holds them.
 */
public final class TornadoTensorLoader {

    private TornadoTensorLoader() {}

    /**
     * A device tensor over a host float array — rope tables, and anything else a loader computes
     * rather than reads.
     */
    public static FP32TornadoTensor fromFloats(float[] values) {
        return new FP32TornadoTensor(FloatArray.fromArray(values), Shape.of(values.length));
    }

    /**
     * A segment with TornadoVM's array header sliced off.
     *
     * <p>A tensor loaded for the device carries a 16-byte {@link TornadoNativeArray} header prefix,
     * so the data can be wrapped as a native array without copying. A reader that wants the raw
     * bytes has to skip it, and <b>how wide that header is</b> is backend knowledge — the loaders
     * used to hold it.
     *
     * <p>Takes and returns a segment rather than a tensor entry: a GGUF type in this package would
     * put the file format inside the backend, which is Rule 4's subject and not something this move
     * is entitled to change.
     */
    public static MemorySegment withoutArrayHeader(MemorySegment segment) {
        return segment.asSlice(TornadoNativeArray.ARRAY_HEADER);
    }

    /**
     * Converts a BF16 tensor to an FP16 {@link FP16TornadoTensor} for TornadoVM/GPU execution.
     * TornadoVM has no native BF16 kernel support, so weights are widened to FP32 (a lossless,
     * simple bit-shift for BF16) and narrowed to IEEE FP16 at load time -- the same representation
     * the existing FP16 GPU kernels already expect.
     */
    public static FP16TornadoTensor convertBF16ToFP16(FloatTensor source, Shape shape) {
        long headerBytes = TornadoNativeArray.ARRAY_HEADER;
        int numElements = source.size();

        MemorySegment nativeSegment =
                Arena.ofAuto().allocate(headerBytes + (long) numElements * Short.BYTES, 4);
        for (long i = 0; i < headerBytes; i++) {
            nativeSegment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        for (int i = 0; i < numElements; i++) {
            short f16Bits = Float.floatToFloat16(source.getFloat(i));
            nativeSegment.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    headerBytes + (long) i * Short.BYTES,
                    f16Bits);
        }
        return FP16TornadoTensor.fromTornadoMemorySegment(nativeSegment, shape);
    }

    /**
     * Like {@code CpuOperations.embeddingLookupLongIndexed}, but writes into a TornadoVM {@link
     * FloatArray} (optionally scaling each element) -- used by the GPU path to gather a per-token
     * embedding row directly into a buffer ready for transfer to the device.
     */
    public static void copyEmbeddingRowToFloatArray(
            LongIndexedTensor table, long rowIndex, int rowSize, FloatArray dest, float scale) {
        copyEmbeddingRowToFloatArray(table, rowIndex, rowSize, dest, 0, scale);
    }

    /**
     * The same gather into a row of a chunk-wide destination: {@code destOffset} is where this
     * token's row starts in a buffer that holds one row per prompt token.
     */
    public static void copyEmbeddingRowToFloatArray(
            LongIndexedTensor table,
            long rowIndex,
            int rowSize,
            FloatArray dest,
            int destOffset,
            float scale) {
        long rowStart = rowIndex * rowSize;
        for (int i = 0; i < rowSize; i++) {
            dest.set(destOffset + i, table.valueAt(rowStart + i) * scale);
        }
    }
}
