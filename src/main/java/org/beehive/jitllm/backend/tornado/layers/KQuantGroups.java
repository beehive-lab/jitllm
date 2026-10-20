package org.beehive.jitllm.backend.tornado.layers;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

// @formatter:off
/**
 * Q4_K and Q6_K weights read as groups of 32, the group MLX's affine format quantizes over.
 *
 * <p><b>Q4_K</b> (144 bytes per 256 weights): an FP16 {@code d} and {@code dmin}, twelve bytes of
 * packed 6-bit scale/min pairs, one pair per sub-block of 32, and 128 bytes of nibbles. A weight is
 * {@code d * sc * q - dmin * m}, so each sub-block is exactly one affine group: {@code q} in 0..15,
 * scale {@code d * sc}, bias {@code -dmin * m}. The two products are rounded to FP16, which MLX
 * stores them in; nothing else changes.
 *
 * <p><b>Q6_K</b> (210 bytes per 256 weights): 6-bit values {@code q - 32} in sub-blocks of 16, each
 * with its own signed 8-bit scale, under one FP16 {@code d}. Two sub-blocks with different scales
 * make one group of 32, which no single affine scale represents exactly, so the decoded group is
 * re-quantized to 8 bits over its own range: an error of at most half of {@code (max - min) / 255}
 * per weight, about an eighth of the 6-bit step.
 */
// @formatter:on
final class KQuantGroups {

    static final int GROUP = 32;

    static final int Q4_K_BLOCK_BYTES = 144;
    static final int Q6_K_BLOCK_BYTES = 210;
    private static final int QK_K = 256;

    private KQuantGroups() {}

    /** Groups of 32 in {@code bytes} of Q4_K. */
    static long q4kGroups(long bytes) {
        return bytes / Q4_K_BLOCK_BYTES * (QK_K / GROUP);
    }

    /** Groups of 32 in {@code bytes} of Q6_K. */
    static long q6kGroups(long bytes) {
        return bytes / Q6_K_BLOCK_BYTES * (QK_K / GROUP);
    }

    /**
     * Q4_K group {@code g}: its 32 nibbles into {@code values}, and {@code {scale, bias}} into
     * {@code affine}.
     */
    static void q4kAffine(MemorySegment src, long g, int[] values, float[] affine) {
        long block = g / 8;
        int j = (int) (g % 8);
        long base = block * Q4_K_BLOCK_BYTES;
        float d = Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, base));
        float dmin = Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, base + 2));
        int sc;
        int m;
        if (j < 4) {
            sc = scaleByte(src, base, j) & 63;
            m = scaleByte(src, base, j + 4) & 63;
        } else {
            sc = (scaleByte(src, base, j + 4) & 0xF) | ((scaleByte(src, base, j - 4) >> 6) << 4);
            m = (scaleByte(src, base, j + 4) >> 4) | ((scaleByte(src, base, j) >> 6) << 4);
        }
        // Sub-blocks 2k and 2k + 1 share the 32 bytes at 32k: low nibbles, then high.
        long qs = base + 16 + 32L * (j / 2);
        boolean high = (j & 1) == 1;
        for (int l = 0; l < GROUP; l++) {
            int b = src.get(ValueLayout.JAVA_BYTE, qs + l) & 0xFF;
            values[l] = high ? b >> 4 : b & 0xF;
        }
        affine[0] = d * sc;
        affine[1] = -dmin * m;
    }

    private static int scaleByte(MemorySegment src, long base, int i) {
        return src.get(ValueLayout.JAVA_BYTE, base + 4 + i) & 0xFF;
    }

    /** Q6_K group {@code g} decoded to its 32 weights. */
    static void q6kDecode(MemorySegment src, long g, float[] out) {
        long block = g / 8;
        int group = (int) (g % 8);
        long base = block * Q6_K_BLOCK_BYTES;
        float d = Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, base + 208));
        // A block is two halves of 128 weights; each half is four runs of 32 (the q1..q4 of the
        // reference decoder), and a group is one such run.
        int half = group / 4;
        int run = group % 4;
        long ql = base + 64L * half;
        long qh = base + 128 + 32L * half;
        long sc = base + 192 + 8L * half;
        for (int l = 0; l < GROUP; l++) {
            int low =
                    switch (run) {
                        case 0 -> src.get(ValueLayout.JAVA_BYTE, ql + l) & 0xF;
                        case 1 -> src.get(ValueLayout.JAVA_BYTE, ql + 32 + l) & 0xF;
                        case 2 -> (src.get(ValueLayout.JAVA_BYTE, ql + l) & 0xFF) >> 4;
                        default -> (src.get(ValueLayout.JAVA_BYTE, ql + 32 + l) & 0xFF) >> 4;
                    };
            int highBits = ((src.get(ValueLayout.JAVA_BYTE, qh + l) & 0xFF) >> (2 * run)) & 3;
            int q = (low | (highBits << 4)) - 32;
            int scale = src.get(ValueLayout.JAVA_BYTE, sc + l / 16 + 2 * run);
            out[l] = d * scale * q;
        }
    }

    /**
     * Q6_K group {@code g} re-quantized to 8 bits over its own range: {@code values} in 0..255,
     * {@code affine = {scale, bias}} with {@code w ≈ scale * value + bias}, both as FP16 holds
     * them.
     */
    static void q6kAffine(MemorySegment src, long g, int[] values, float[] affine) {
        float[] w = new float[GROUP];
        q6kDecode(src, g, w);
        float lo = w[0];
        float hi = w[0];
        for (float v : w) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        float bias = Float.float16ToFloat(Float.floatToFloat16(lo));
        float scale = Float.float16ToFloat(Float.floatToFloat16((hi - lo) / 255f));
        for (int l = 0; l < GROUP; l++) {
            values[l] = scale == 0f ? 0 : Math.clamp(Math.round((w[l] - bias) / scale), 0, 255);
        }
        affine[0] = scale;
        affine[1] = bias;
    }
}
