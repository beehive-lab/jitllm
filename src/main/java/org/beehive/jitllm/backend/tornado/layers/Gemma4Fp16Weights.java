package org.beehive.jitllm.backend.tornado.layers;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * A projection's weights decoded once, on the host, into FP16 row-major matrices for the native
 * (cuBLAS) batched prefill: {@code [n][k]}, several tensors of one layer stacked along {@code n} so
 * one GEMM computes them side by side.
 *
 * <p><b>Values.</b> Each weight is {@code fp16(scale * q)} with the file's FP16 block scale widened
 * to FP32 — Q8_0 {@code q} its signed byte, Q4_0 its nibble minus 8 — and Q4_1 {@code fp16(scale *
 * q + min)}: the same rounding the JIT path's device dequantizers apply before their FP16 GEMMs.
 */
// @formatter:on
final class Gemma4Fp16Weights {

    private static final int BLOCK = 32;

    private Gemma4Fp16Weights() {}

    static boolean decodable(TornadoTensor t) {
        return t.dataType() == DataType.Q8_0
                || t.dataType() == DataType.Q4_0
                || t.dataType() == DataType.Q4_1;
    }

    /** The tensors, each {@code rows[i]} by {@code k}, stacked into one {@code [n][k]} matrix. */
    static HalfFloatArray stack(int k, TornadoTensor[] parts, int[] rows) {
        int n = 0;
        for (int r : rows) {
            n += r;
        }
        HalfFloatArray out = new HalfFloatArray(n * k);
        int kBlocks = k / BLOCK;
        int base = 0;
        for (int p = 0; p < parts.length; p++) {
            TornadoTensor t = parts[p];
            DataType type = t.dataType();
            if (!decodable(t)) {
                throw new IllegalArgumentException("no FP16 decode for " + type);
            }
            final int rowBase = base;
            final int blockBytes = type == DataType.Q8_0 ? 34 : type == DataType.Q4_0 ? 18 : 20;
            final ByteArray src = t.asByteArray();
            final MemorySegment dstSeg = out.getSegment();
            org.beehive.jitllm.auxiliary.Parallel.parallelFor(
                    0,
                    rows[p],
                    r -> {
                        long dst = (long) (rowBase + r) * k;
                        for (int kb = 0; kb < kBlocks; kb++) {
                            int off = (r * kBlocks + kb) * blockBytes;
                            float scale = src.getHalfFloat(off).getFloat32();
                            long at = dst + kb * BLOCK;
                            switch (type) {
                                case Q8_0 -> {
                                    for (int i = 0; i < BLOCK; i++) {
                                        put(dstSeg, at + i, scale * src.get(off + 2 + i));
                                    }
                                }
                                case Q4_0 -> {
                                    for (int i = 0; i < 16; i++) {
                                        int packed = src.get(off + 2 + i) & 0xFF;
                                        put(dstSeg, at + i, scale * ((packed & 0x0F) - 8));
                                        put(dstSeg, at + i + 16, scale * ((packed >>> 4) - 8));
                                    }
                                }
                                default -> {
                                    float min = src.getHalfFloat(off + 2).getFloat32();
                                    for (int i = 0; i < 16; i++) {
                                        int packed = src.get(off + 4 + i) & 0xFF;
                                        put(dstSeg, at + i, scale * (packed & 0x0F) + min);
                                        put(dstSeg, at + i + 16, scale * (packed >>> 4) + min);
                                    }
                                }
                            }
                        }
                    });
            base += rows[p];
        }
        return out;
    }

    /** Element {@code i} of an FP16 segment, rounded as {@code new HalfFloat(v)} rounds. */
    private static void put(MemorySegment seg, long i, float v) {
        seg.setAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i, Float.floatToFloat16(v));
    }
}
