package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The element-wise glue between the MLX GEMMs of a batched prefill and the JIT kernels around them:
 * MLX multiplies FP16 into FP16, and the rest of the layer keeps its activations in FP32.
 *
 * <p>Every kernel takes one thread per element over {@code total} elements; the grid may be rounded
 * up past it.
 */
public final class MlxPrefillKernels {

    private MlxPrefillKernels() {}

    /** {@code out[i] = FP16(in[i])}. */
    public static void toHalf(KernelContext context, FloatArray in, HalfFloatArray out, int total) {
        int id = context.globalIdx;
        if (id < total) {
            out.set(id, new HalfFloat(in.get(id)));
        }
    }

    /** Splits the stacked {@code [B][qDim | kvDim | kvDim]} QKV product into FP32 Q, K and V. */
    public static void splitQkv(
            KernelContext context,
            HalfFloatArray qkv,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            int qDim,
            int kvDim,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int width = qDim + 2 * kvDim;
            int b = id / width;
            int c = id % width;
            float value = qkv.get(id).getFloat32();
            if (c < qDim) {
                q.set(b * qDim + c, value);
            } else if (c < qDim + kvDim) {
                k.set(b * kvDim + c - qDim, value);
            } else {
                v.set(b * kvDim + c - qDim - kvDim, value);
            }
        }
    }

    /** {@code x[i] += scale * y[i]}: the residual connection after a projection. */
    public static void addResidual(
            KernelContext context, FloatArray x, HalfFloatArray y, float scale, int total) {
        int id = context.globalIdx;
        if (id < total) {
            x.set(id, x.get(id) + scale * y.get(id).getFloat32());
        }
    }

    /**
     * {@code hb[b][i] = SiLU(gateUp[b][i]) * gateUp[b][hidden + i]} over the stacked {@code
     * [B][gate | up]} product, in FP16 for the down projection.
     */
    public static void swiglu(
            KernelContext context,
            HalfFloatArray gateUp,
            HalfFloatArray hb,
            int hidden,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int b = id / hidden;
            int i = id % hidden;
            float gate = gateUp.get(b * 2 * hidden + i).getFloat32();
            float up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
            float silu = gate / (1.0f + TornadoMath.exp(-gate));
            hb.set(id, new HalfFloat(silu * up));
        }
    }

    /**
     * The largest magnitude {@link #toHalfScaled} and {@link #swigluScaled} let a row of a
     * projection's input reach: low enough that the GEMM's FP16 products stay finite, high enough
     * that ordinary rows are not scaled at all.
     */
    public static final float PROJECTION_INPUT_LIMIT = 256.0f;

    /**
     * {@code rowScale[b] = min(1, PROJECTION_INPUT_LIMIT / max_i |in[b][i]|)}: the factor that
     * keeps row {@code b} of a projection, input and product, inside FP16. One workgroup of {@code
     * localSize} threads per row of {@code width}, over {@code rows} rows.
     */
    public static void rowScale(
            KernelContext context,
            FloatArray in,
            FloatArray rowScale,
            int width,
            int rows,
            int localSize) {
        int tid = context.localIdx;
        int b = context.groupIdx;
        float[] partial = context.allocateFloatLocalArray(localSize);
        float max = 0.0f;
        if (b < rows) {
            for (int i = tid; i < width; i += localSize) {
                max = TornadoMath.max(max, TornadoMath.abs(in.get(b * width + i)));
            }
        }
        partial[tid] = max;
        context.localBarrier();
        for (int s = localSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] = TornadoMath.max(partial[tid], partial[tid + s]);
            }
            context.localBarrier();
        }
        if (tid == 0 && b < rows) {
            float peak = partial[0];
            rowScale.set(b, peak > PROJECTION_INPUT_LIMIT ? PROJECTION_INPUT_LIMIT / peak : 1.0f);
        }
    }

    /** {@code out[b][i] = FP16(in[b][i] * rowScale[b])}: {@link #toHalf} with {@link #rowScale}. */
    public static void toHalfScaled(
            KernelContext context,
            FloatArray in,
            HalfFloatArray out,
            FloatArray rowScale,
            int width,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            out.set(id, new HalfFloat(in.get(id) * rowScale.get(id / width)));
        }
    }

    /**
     * {@code rowScale[b] = min(1, PROJECTION_INPUT_LIMIT / max_i |SiLU(gate[b][i]) * up[b][i]|)}:
     * the factor that keeps row {@code b} of the down projection, input and product, inside FP16.
     * Some tokens (the first, in Granite 4.0) carry activations past FP16's 65504 there. One
     * workgroup of {@code localSize} threads per row, over {@code rows} rows.
     */
    public static void swigluRowScale(
            KernelContext context,
            HalfFloatArray gateUp,
            FloatArray rowScale,
            int hidden,
            int rows,
            int localSize) {
        int tid = context.localIdx;
        int b = context.groupIdx;
        float[] partial = context.allocateFloatLocalArray(localSize);
        float max = 0.0f;
        if (b < rows) {
            for (int i = tid; i < hidden; i += localSize) {
                float gate = gateUp.get(b * 2 * hidden + i).getFloat32();
                float up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
                float silu = gate / (1.0f + TornadoMath.exp(-gate));
                max = TornadoMath.max(max, TornadoMath.abs(silu * up));
            }
        }
        partial[tid] = max;
        context.localBarrier();
        for (int s = localSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] = TornadoMath.max(partial[tid], partial[tid + s]);
            }
            context.localBarrier();
        }
        if (tid == 0 && b < rows) {
            float peak = partial[0];
            rowScale.set(b, peak > PROJECTION_INPUT_LIMIT ? PROJECTION_INPUT_LIMIT / peak : 1.0f);
        }
    }

    /** {@link #swiglu} with row {@code b} multiplied by {@code rowScale[b]} before FP16. */
    public static void swigluScaled(
            KernelContext context,
            HalfFloatArray gateUp,
            HalfFloatArray hb,
            FloatArray rowScale,
            int hidden,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int b = id / hidden;
            int i = id % hidden;
            float gate = gateUp.get(b * 2 * hidden + i).getFloat32();
            float up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
            float silu = gate / (1.0f + TornadoMath.exp(-gate));
            hb.set(id, new HalfFloat(silu * up * rowScale.get(b)));
        }
    }

    /**
     * {@code x[b][i] += scale * y[b][i] / rowScale[b]}: {@link #addResidual} undoing the row factor
     * of {@link #toHalfScaled} or {@link #swigluScaled} in FP32.
     */
    public static void addRowScaledResidual(
            KernelContext context,
            FloatArray x,
            HalfFloatArray y,
            FloatArray rowScale,
            float scale,
            int width,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            x.set(id, x.get(id) + scale * y.get(id).getFloat32() / rowScale.get(id / width));
        }
    }

    /** {@code out[i] = in[i]}: an MLX product back into FP32. */
    public static void toFloat(
            KernelContext context, HalfFloatArray in, FloatArray out, int total) {
        int id = context.globalIdx;
        if (id < total) {
            out.set(id, in.get(id).getFloat32());
        }
    }

    /**
     * {@code out[b][i] = in[b][i] / rowScale[b]}: {@link #toFloat} undoing the row factor of {@link
     * #gegluScaled} in FP32.
     */
    public static void toFloatRowScaled(
            KernelContext context,
            HalfFloatArray in,
            FloatArray out,
            FloatArray rowScale,
            int width,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            out.set(id, in.get(id).getFloat32() / rowScale.get(id / width));
        }
    }

    /**
     * {@code out[b][i] = src[b][offset + i]} for {@code i < width}: one projection's columns of a
     * stacked product of {@code srcWidth} per row, back into FP32. Over {@code total = rows *
     * width}.
     */
    public static void splitRows(
            KernelContext context,
            HalfFloatArray src,
            FloatArray out,
            int srcWidth,
            int offset,
            int width,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int b = id / width;
            int i = id % width;
            out.set(id, src.get(b * srcWidth + offset + i).getFloat32());
        }
    }

    /**
     * {@code rowScale[b] = min(1, PROJECTION_INPUT_LIMIT / max_i |GELU(gate[b][i]) * up[b][i]|)}:
     * {@link #swigluRowScale} for a GeGLU feed-forward (Gemma), over the stacked {@code [B][gate |
     * up]} product. One workgroup of {@code localSize} threads per row, over {@code rows} rows.
     */
    public static void gegluRowScale(
            KernelContext context,
            HalfFloatArray gateUp,
            FloatArray rowScale,
            int hidden,
            int rows,
            int localSize) {
        int tid = context.localIdx;
        int b = context.groupIdx;
        float[] partial = context.allocateFloatLocalArray(localSize);
        float max = 0.0f;
        if (b < rows) {
            for (int i = tid; i < hidden; i += localSize) {
                float gate = gateUp.get(b * 2 * hidden + i).getFloat32();
                float up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
                float gelu = TransformerComputeKernelsLayered.geluActivation(gate);
                max = TornadoMath.max(max, TornadoMath.abs(gelu * up));
            }
        }
        partial[tid] = max;
        context.localBarrier();
        for (int s = localSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] = TornadoMath.max(partial[tid], partial[tid + s]);
            }
            context.localBarrier();
        }
        if (tid == 0 && b < rows) {
            float peak = partial[0];
            rowScale.set(b, peak > PROJECTION_INPUT_LIMIT ? PROJECTION_INPUT_LIMIT / peak : 1.0f);
        }
    }

    /**
     * {@code hb[b][i] = FP16(GELU(gate[b][i]) * up[b][i] * rowScale[b])} over the stacked {@code
     * [B][gate | up]} product, for the down projection.
     */
    public static void gegluScaled(
            KernelContext context,
            HalfFloatArray gateUp,
            HalfFloatArray hb,
            FloatArray rowScale,
            int hidden,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int b = id / hidden;
            int i = id % hidden;
            float gate = gateUp.get(b * 2 * hidden + i).getFloat32();
            float up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
            float gelu = TransformerComputeKernelsLayered.geluActivation(gate);
            hb.set(id, new HalfFloat(gelu * up * rowScale.get(b)));
        }
    }

    /**
     * {@code dst[h][t][d] = FP16(src[t][h * headDim + d])}: token-major activations to the
     * head-major layout of MLX's attention, over {@code total = tokens * heads * headDim}.
     */
    public static void packHeads(
            KernelContext context,
            FloatArray src,
            HalfFloatArray dst,
            int heads,
            int tokens,
            int headDim,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int h = id / (tokens * headDim);
            int r = id % (tokens * headDim);
            int t = r / headDim;
            int d = r % headDim;
            dst.set(id, new HalfFloat(src.get(t * heads * headDim + h * headDim + d)));
        }
    }

    /** {@code dst[t][h * headDim + d] = src[h][t][d]}: the inverse of {@link #packHeads}. */
    public static void unpackHeads(
            KernelContext context,
            HalfFloatArray src,
            HalfFloatArray dst,
            int heads,
            int tokens,
            int headDim,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int t = id / (heads * headDim);
            int c = id % (heads * headDim);
            int h = c / headDim;
            int d = c % headDim;
            dst.set(id, src.get(h * tokens * headDim + t * headDim + d));
        }
    }

    /** {@link #unpackHeads} into FP32, for the JIT output projection. */
    public static void unpackHeadsToFloat(
            KernelContext context,
            HalfFloatArray src,
            FloatArray dst,
            int heads,
            int tokens,
            int headDim,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int t = id / (heads * headDim);
            int c = id % (heads * headDim);
            int h = c / headDim;
            int d = c % headDim;
            dst.set(id, src.get(h * tokens * headDim + t * headDim + d).getFloat32());
        }
    }

    /**
     * Gathers layer {@code layer}'s keys and values of positions {@code 0 .. start + rows - 1} from
     * the paged FP16 cache into {@code [head][position][headDim]} buffers of {@code maxPositions}
     * rows, for MLX's attention over the whole context. Positions of padding rows, which the cache
     * does not hold, are written as zeros: only padding queries reach them. One thread per element
     * of {@code total = maxPositions * kvDim}.
     */
    public static void gatherKvFP16(
            KernelContext context,
            IntArray batchStartPosHolder,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray k,
            HalfFloatArray v,
            IntArray blockTable,
            int layer,
            int kvDim,
            int headDim,
            int maxPositions,
            int rows,
            int blockCfg,
            int blockStride,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int pos = id / kvDim;
            int c = id % kvDim;
            int start = batchStartPosHolder.get(0);
            if (pos < start + rows) {
                int dst = (c / headDim) * maxPositions * headDim + pos * headDim + c % headDim;
                if (pos < start + batchStartPosHolder.get(1)) {
                    int src =
                            KvBlockAddress.offset(
                                            blockTable,
                                            batchStartPosHolder.get(2),
                                            pos,
                                            KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + c;
                    k.set(dst, keyCache.get(src));
                    v.set(dst, valueCache.get(src));
                } else {
                    k.set(dst, new HalfFloat(0.0f));
                    v.set(dst, new HalfFloat(0.0f));
                }
            }
        }
    }

    /** {@link #gatherKvFP16} from the FP32 cache. */
    public static void gatherKvFP32(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray k,
            HalfFloatArray v,
            IntArray blockTable,
            int layer,
            int kvDim,
            int headDim,
            int maxPositions,
            int rows,
            int blockCfg,
            int blockStride,
            int total) {
        int id = context.globalIdx;
        if (id < total) {
            int pos = id / kvDim;
            int c = id % kvDim;
            int start = batchStartPosHolder.get(0);
            if (pos < start + rows) {
                int dst = (c / headDim) * maxPositions * headDim + pos * headDim + c % headDim;
                if (pos < start + batchStartPosHolder.get(1)) {
                    int src =
                            KvBlockAddress.offset(
                                            blockTable,
                                            batchStartPosHolder.get(2),
                                            pos,
                                            KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + c;
                    k.set(dst, new HalfFloat(keyCache.get(src)));
                    v.set(dst, new HalfFloat(valueCache.get(src)));
                } else {
                    k.set(dst, new HalfFloat(0.0f));
                    v.set(dst, new HalfFloat(0.0f));
                }
            }
        }
    }
}
