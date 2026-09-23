package org.beehive.jllm.backend.tornado.kernels;

import static org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered.matrixVectorRowMajorOptimized;
import static org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered.matrixVectorRowMajorOptimizedQ8_0Byte;
import static org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered.matrixVectorRowMajorOptimizedSingle;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class GraniteKernels {

    public static void convertFP16toFP32withGraniteScale(
            KernelContext context, HalfFloatArray x, FloatArray wrapX, float embeddingScale) {
        int i = context.globalIdx;
        wrapX.set(i, embeddingScale * x.get(i).getFloat32());
    }

    public static void convertQ8_0toFP32withGraniteScale(
            KernelContext context, ByteArray x, FloatArray wrapX, float embeddingScale) {
        int globalId = context.globalIdx;
        int totalElements = wrapX.getSize();

        if (globalId >= totalElements) {
            return;
        }

        // Q8_0 block structure constants
        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Calculate which block and position within block
        int blockIdx = globalId / blockSize;
        int withinBlockIdx = globalId % blockSize;

        // Calculate byte offset for this Q8_0 block
        int blockByteOffset = blockIdx * Q8_0_BLOCK_BYTES;

        // Load scale (first 2 bytes of block as HalfFloat)
        HalfFloat scale = x.getHalfFloat(blockByteOffset);
        float scaleFloat = scale.getFloat32();

        // Load quantized value (skip 2-byte scale, then index within block)
        byte quantValue = x.get(blockByteOffset + 2 + withinBlockIdx);

        // Dequantize: float_value = quantized_value * scale
        float dequantizedValue = ((float) quantValue) * scaleFloat;

        // Store result in output FloatArray
        wrapX.set(globalId, embeddingScale * dequantizedValue);
    }

    // @formatter:off
    public static void matrixVectorGenericWithGraniteScale(
            KernelContext context,
            HalfFloatArray x,
            FloatArray hb, // output
            HalfFloatArray w,
            int dim1, // inner loop
            int dim0, // outer loop
            int localWorkGroupSize,
            float logitsScale) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= dim0) {
            return;
        }
        float sum = matrixVectorRowMajorOptimizedSingle(context, localSize, x, w, dim1);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum * logitsScale);
        }
    }

    // @formatter:on

    public static void matrixVectorGenericWithResidualGranite(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            HalfFloatArray w,
            int n,
            int d,
            int localWorkGroupSize,
            float residualScale) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float residual = residualScale * sum;
            float result = hb.get(rowId) + residual;
            hb.set(rowId, result);
        }
    }

    public static void matrixVectorGenericWithResidualQ8_0ByteWithGraniteScale(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize,
            float residualScale) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimizedQ8_0Byte(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float residualScaledSum = residualScale * sum;
            float result = hb.get(rowId) + residualScaledSum;
            hb.set(rowId, result);
        }
    }

    public static void matrixVectorGenericQ8ByteWithGraniteScale(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray q,
            int dim1,
            int dim0,
            int localWorkGroupSize,
            float logitsScale) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= dim0) {
            return;
        }

        float sum = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, q, dim1);

        // Thread 0 writes the result
        if (localId == 0) {
            output.set(rowId, sum * logitsScale);
        }
    }
}
