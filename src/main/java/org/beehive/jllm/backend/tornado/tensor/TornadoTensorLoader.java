package org.beehive.jllm.backend.tornado.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.beehive.jllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
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
        return new FP32TornadoTensor(FloatArray.fromArray(values));
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
     * Materializes a tensor the GPU has no kernel for (Q4_0, Q4_K, Q5_K, Q6_K) as Q8_0.
     *
     * <p>A load-time conversion, done through the CPU tensor for that format — which already knows
     * how to decode it — so the device path needs one set of quantized kernels rather than one per
     * file format. The cost is device memory: an 18-byte Q4_0 block becomes a 34-byte Q8_0 block,
     * so a 4-bit file occupies roughly twice as much on the device as it does on disk. The accuracy
     * cost is negligible in the other direction: Q4_0 carries 16 levels per block and Q8_0 stores
     * 255, so the re-quantization error is far below the source's own step size.
     */
    public static Q8_0TornadoTensor dequantizeToQ8_0(FloatTensor sourceTensor) {
        long headerBytes = TornadoNativeArray.ARRAY_HEADER;
        int numElements = sourceTensor.size();
        int blockSize = 32;
        int blocksNeeded = (numElements + blockSize - 1) / blockSize;
        int q8BlockBytes = 34; // 2 bytes scale + 32 bytes quants
        int q8BytesNeeded = blocksNeeded * q8BlockBytes;

        byte[] q8Data = new byte[q8BytesNeeded];

        for (int b = 0; b < blocksNeeded; b++) {
            int start = b * blockSize;
            int end = Math.min(start + blockSize, numElements);

            // Find max absolute value for scale
            float maxAbs = 0;
            for (int i = start; i < end; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(sourceTensor.getFloat(i)));
            }
            float scale = maxAbs / 127.0f;

            // Write scale as fp16 (little-endian)
            short scaleF16 = Float.floatToFloat16(scale);
            int blockOff = b * q8BlockBytes;
            q8Data[blockOff] = (byte) (scaleF16 & 0xFF);
            q8Data[blockOff + 1] = (byte) ((scaleF16 >> 8) & 0xFF);

            // Quantize values
            float invScale = scale != 0 ? 1.0f / scale : 0;
            for (int i = start; i < end; i++) {
                int qi = Math.round(sourceTensor.getFloat(i) * invScale);
                qi = Math.max(-128, Math.min(127, qi));
                q8Data[blockOff + 2 + (i - start)] = (byte) qi;
            }
        }

        // Allocate native memory with TornadoNativeArray header, matching GGUF.loadTensorsTornado
        // layout
        MemorySegment nativeSegment = Arena.ofAuto().allocate(headerBytes + q8BytesNeeded, 4);
        // Zero out the header
        for (int i = 0; i < headerBytes; i++) {
            nativeSegment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        // Copy Q8_0 data after header
        MemorySegment.copy(
                MemorySegment.ofArray(q8Data), 0, nativeSegment, headerBytes, q8BytesNeeded);
        return Q8_0TornadoTensor.fromTornadoMemorySegment(nativeSegment);
    }

    /**
     * Converts a BF16 tensor to an FP16 {@link FP16TornadoTensor} for TornadoVM/GPU execution.
     * TornadoVM has no native BF16 kernel support, so weights are widened to FP32 (a lossless,
     * simple bit-shift for BF16) and narrowed to IEEE FP16 at load time -- the same representation
     * the existing FP16 GPU kernels already expect (see {@link #loadTornadoTensor}).
     */
    public static FP16TornadoTensor convertBF16ToFP16(FloatTensor source) {
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
        return FP16TornadoTensor.fromTornadoMemorySegment(nativeSegment);
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
