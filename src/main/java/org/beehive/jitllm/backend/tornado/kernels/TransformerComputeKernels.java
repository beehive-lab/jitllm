package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class TransformerComputeKernels {

    /**
     * On-device greedy sampling: single-workgroup argmax over the {@code vocab} logits. Each thread
     * scans a strided slice tracking its local (max value, index); a local-memory tree reduction
     * picks the global argmax and lane 0 writes the token id to {@code out[0]}. Lets the decode
     * step transfer only that one int to the host instead of the whole vocab-sized logits row (D2H
     * copy + host scan removed). Launch with one workgroup: {@code global == local ==
     * localMemSize}.
     */
    // @formatter:off
    /**
     * {@code x[i] *= scale}, in place, one lane per element.
     *
     * <p>Format-neutral and family-neutral: Gemma scales an embedding by {@code sqrt(dim)}, and a
     * delta-net layer scales its queries by {@code 1/sqrt(headKeyDim)} before the recurrence. Same
     * arithmetic, so one kernel — a copy named after either family would be the second one.
     *
     * <p>Each lane reads and writes only its own element, so the in-place write destroys nothing
     * another lane still needs.
     */
    // @formatter:on
    static void scaleInPlaceLane(FloatArray x, float scale, int lane) {
        x.set(lane, x.get(lane) * scale);
    }

    /** One lane per element. */
    public static void scaleInPlace(KernelContext context, FloatArray x, float scale, int size) {
        int lane = context.globalIdx;
        if (lane >= size) {
            return;
        }
        scaleInPlaceLane(x, scale, lane);
    }

    // @formatter:off
    /**
     * One element of a fused three-way projection, copied into the slice it belongs to.
     *
     * <p>The widths are stated separately rather than as {@code (q, kv, kv)}. {@code splitQKV} in
     * {@code TransformerComputeKernelsLayered} assumes a key and a value of equal width, which is
     * true of attention and false of a delta-net mixer: its fused projection is {@code 2048 | 2048
     * | 6144}, and splitting it on equal halves would take the value slice from inside the keys.
     *
     * <p>Source and destinations are distinct buffers, so no lane overwrites an element another
     * lane has yet to read.
     *
     * @param lane an element of the fused buffer, {@code 0 .. dimA + dimB + dimC - 1}
     */
    // @formatter:on
    static void splitThreeWayLane(
            FloatArray fused,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int lane) {
        if (lane < dimA) {
            a.set(lane, fused.get(lane));
        } else if (lane < dimA + dimB) {
            b.set(lane - dimA, fused.get(lane));
        } else {
            c.set(lane - dimA - dimB, fused.get(lane));
        }
    }

    /** One lane per element of the fused buffer. */
    public static void splitThreeWay(
            KernelContext context,
            FloatArray fused,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC) {
        int lane = context.globalIdx;
        if (lane >= dimA + dimB + dimC) {
            return;
        }
        splitThreeWayLane(fused, a, b, c, dimA, dimB, lane);
    }

    public static void argmaxLogits(
            KernelContext context, FloatArray logits, IntArray out, int vocab, int localMemSize) {
        int tid = context.localIdx;
        int localSz = context.localGroupSizeX;
        float[] vals = context.allocateFloatLocalArray(localMemSize);
        int[] idxs = context.allocateIntLocalArray(localMemSize);

        float best = Float.NEGATIVE_INFINITY;
        int bestIdx = 0;
        for (int i = tid; i < vocab; i += localSz) {
            float v = logits.get(i);
            if (v > best) {
                best = v;
                bestIdx = i;
            }
        }
        vals[tid] = best;
        idxs[tid] = bestIdx;
        context.localBarrier();

        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid + s] > vals[tid]) {
                    vals[tid] = vals[tid + s];
                    idxs[tid] = idxs[tid + s];
                }
            }
            context.localBarrier();
        }
        if (tid == 0) {
            out.set(0, idxs[0]);
        }
    }

    /** Default constructor for the TransformerComputeKernels class. */
    public TransformerComputeKernels() {}

    public static void emptyTaskToForceCopyIn(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    public static void convertFP32toFP16v2(
            KernelContext context, FloatArray input, HalfFloatArray output) {
        int i = context.globalIdx;
        HalfFloat val = new HalfFloat(input.get(i));
        output.set(i, val);
    }

    public static void mapContextWithQuantize(
            KernelContext context,
            HalfFloatArray outputFP16, // Direct FP16 output
            FloatArray x,
            FloatArray weights,
            FloatArray temp) {

        int gid = context.globalIdx;
        float ss = temp.get(0);
        float result = weights.get(gid) * (ss * x.get(gid));
        outputFP16.set(gid, new HalfFloat(result));
    }

    public static void convertFP16toFP32(
            KernelContext context, HalfFloatArray x, FloatArray wrapX) {
        int i = context.globalIdx;
        // Guard as convertQ8_0toFP32 does: the grid is rounded up to the local size, so a model
        // whose dim is not a multiple of it would otherwise write past the end of wrapX.
        if (i >= wrapX.getSize()) {
            return;
        }
        wrapX.set(i, x.get(i).getFloat32());
    }

    /**
     * The embedding row, decoded from Q4_0 blocks into FP32.
     *
     * <p>The Q4_0 twin of {@link #convertQ8_0toFP32}. A model whose token embeddings are retained
     * as Q4_0 stages 18-byte blocks rather than 34-byte ones, and this is what turns them into the
     * activation the first layer reads. Without it a Q4_0 embedding would have to be materialized
     * as Q8_0 purely to be looked up, which is the conversion the rest of this backend no longer
     * does.
     */
    public static void convertQ4_0toFP32(KernelContext context, ByteArray x, FloatArray wrapX) {
        int globalId = context.globalIdx;
        if (globalId >= wrapX.getSize()) {
            return;
        }

        int blockSize = 32;
        int Q4_0_BLOCK_BYTES = 18; // 2 bytes scale + 16 bytes of packed nibbles

        int blockIdx = globalId / blockSize;
        int withinBlockIdx = globalId - blockIdx * blockSize;
        int blockByteOffset = blockIdx * Q4_0_BLOCK_BYTES;

        // Assembled from two byte loads rather than through getHalfFloat: that call inlined into a
        // kernel is one TornadoVM's sketcher rejects, as TransformerComputeKernelsQ6_K records.
        int lo = x.get(blockByteOffset) & 0xFF;
        int hi = x.get(blockByteOffset + 1) & 0xFF;
        int h = (hi << 8) | lo;
        int mantissa = h & 0x3FF;
        int exponent = (h >>> 10) & 0x1F;
        float magnitude;
        if (exponent == 0) {
            magnitude = mantissa * 5.9604645E-8f;
        } else {
            float scaled = 1.0f + mantissa * 9.765625E-4f;
            int shift = exponent - 15;
            float power = 1.0f;
            if (shift > 0) {
                power = (float) (1 << shift);
            } else if (shift < 0) {
                power = 1.0f / (float) (1 << (-shift));
            }
            magnitude = scaled * power;
        }
        float scale = magnitude;
        if ((h & 0x8000) != 0) {
            scale = -magnitude;
        }

        int half = withinBlockIdx / 16;
        int byteIndex = withinBlockIdx - half * 16;
        int packed = x.get(blockByteOffset + 2 + byteIndex) & 0xFF;
        int q = packed & 0xF;
        if (half == 1) {
            q = (packed >> 4) & 0xF;
        }
        wrapX.set(globalId, scale * (q - 8));
    }

    public static void convertQ8_0toFP32(KernelContext context, ByteArray x, FloatArray wrapX) {
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
        wrapX.set(globalId, dequantizedValue);
    }

    public static void convertFP32toFP16(
            KernelContext context, FloatArray wrapX, HalfFloatArray x) {
        int i = context.globalIdx;
        float valInput = wrapX.get(i);
        HalfFloat val = new HalfFloat(valInput);
        x.set(i, val);
    }

    /**
     * Performs RMS (Root Mean Square) normalization using parallel reduction. This is a two-phase
     * reduction: first within work groups, then across work groups.
     *
     * <p>Phase 1: Each work group computes a partial sum of squares Phase 2: First thread combines
     * all partial sums and computes normalization factor
     *
     * @param context Kernel execution context
     * @param output Array to store partial sums and final normalization factor
     * @param x Input array to normalize
     * @param size Number of elements to process
     * @param ermsNorm Epsilon value for numerical stability (epsilon * epsilon)
     * @param localMemSize Size of local memory allocation (work group size)
     */
    public static void reductionOneBlockWithLayer(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int size,
            float ermsNorm,
            int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= (size / localMemSize); i++) { // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss); // Store the final scale factor
        }
    }

    /**
     * Applies the computed normalization factor to scale weights. This is the second phase of RMS
     * normalization.
     *
     * @param context Kernel execution context
     * @param output Array for normalized output
     * @param weights Weight values to normalize
     * @param temp Temporary array containing a normalization factor at index 0
     */
    public static void reductionOneBlock2WithLogits(
            KernelContext context, FloatArray output, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * output.get(gid)));
    }

    public static void mapContextWithQuantizeLogits(
            KernelContext context,
            HalfFloatArray output,
            FloatArray input,
            FloatArray weights,
            FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        float in = ss * input.get(gid);
        float interim = weights.get(gid) * in;
        output.set(gid, new HalfFloat(interim));
    }
}
