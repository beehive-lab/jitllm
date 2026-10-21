package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Device-side repacking of Q8_0 and Q4_0 weights into the tiles of {@link
 * Int8GemmKernels#packQ8_0Tiles} and {@link Int8GemmKernels#packQ4_0Tiles}: the same bytes as
 * {@link PackedTilePacker}, written by the GPU from the GGUF layout already on the device.
 *
 * <p>{@link #copy} first moves the uploaded GGUF bytes into a scratch buffer, then a repack kernel
 * writes the packed tiles back over the weight's own buffer, so the packed form takes no device
 * memory beyond one scratch buffer the size of the largest weight.
 *
 * <p>Workers, local size {@link #LOCAL}: {@link #copy} {@code bytes / 8} lanes (rounded up); {@link
 * #repackQ8_0} {@code tiles * 2304} lanes; {@link #repackQ4_0} {@code tiles * 768} lanes, {@code
 * tiles = (n / 128) * (k / 64)}.
 */
// @formatter:on
public final class PackedRepackKernels {

    /** Threads of a workgroup. */
    public static final int LOCAL = 256;

    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int Q4_0_BLOCK_BYTES = 18;

    private PackedRepackKernels() {}

    /** Lanes of {@link #repackQ8_0} per tile: 2048 quant words and 256 scales. */
    public static final int Q8_0_LANES_PER_TILE = 2304;

    /** Lanes of {@link #repackQ4_0} per tile: 512 nibble words and 256 scales. */
    public static final int Q4_0_LANES_PER_TILE = 768;

    /** {@code dst[0, bytes) = src[0, bytes)}, eight bytes a lane and the tail a byte at a time. */
    public static void copy(KernelContext context, ByteArray src, ByteArray dst, int bytes) {
        int id = context.globalIdx;
        int offset = id << 3;
        if (offset + 8 <= bytes) {
            dst.setLong(offset, src.getLong(offset));
        } else if (offset < bytes) {
            for (int i = offset; i < bytes; i++) {
                dst.set(i, src.get(i));
            }
        }
    }

    /** Byte {@code i} of {@code a} as an unsigned value. */
    private static int unsigned(ByteArray a, int i) {
        return a.get(i) & 0xFF;
    }

    /** The two bytes at {@code i} as an unsigned little-endian 16-bit value. */
    private static int unsigned16(ByteArray a, int i) {
        return unsigned(a, i) | (unsigned(a, i + 1) << 8);
    }

    /** Repacks the {@code n x k} Q8_0 weight in {@code raw} into {@code packed}. */
    public static void repackQ8_0(
            KernelContext context, ByteArray raw, ByteArray packed, int n, int k) {
        int id = context.globalIdx;
        int rounds = k >> 6;
        int tiles = (n >> 7) * rounds;
        if (id < tiles * Q8_0_LANES_PER_TILE) {
            int tile = id / Q8_0_LANES_PER_TILE;
            int lane = id - tile * Q8_0_LANES_PER_TILE;
            int colTile = tile / rounds;
            int round = tile - colTile * rounds;
            int rowBytes = (k >> 5) * Q8_0_BLOCK_BYTES;
            int base = tile * Int8GemmKernels.PACKED_TILE_BYTES;
            if (lane < 2048) {
                int block = (round << 1) + (lane >> 10);
                int half = (lane >> 9) & 1;
                int slot = lane & 511;
                int colPair = ((slot >> 6) << 2) | (slot & 3);
                int kRow = (slot >> 2) & 15;
                int col = (colTile << 7) + (half << 6) + (colPair << 1);
                int src = col * rowBytes + block * Q8_0_BLOCK_BYTES + 2 + (kRow << 1);
                packed.setInt(
                        base + (lane << 2),
                        unsigned16(raw, src) | (unsigned16(raw, src + rowBytes) << 16));
            } else {
                int j = lane - 2048;
                int col = (colTile << 7) + (j & 127);
                int src = col * rowBytes + ((round << 1) + (j >> 7)) * Q8_0_BLOCK_BYTES;
                int dst = base + Int8GemmKernels.PACKED_QUANT_BYTES + (j << 1);
                packed.set(dst, raw.get(src));
                packed.set(dst + 1, raw.get(src + 1));
            }
        }
    }

    /**
     * The two nibbles of quants {@code q, q + 1} of a Q4_0 block's column, {@code q} even, from its
     * bytes at {@code src} (low nibbles for {@code q < 16}, else high), as one byte.
     */
    private static int nibblePair(ByteArray raw, int src, int high) {
        int b0 = unsigned(raw, src);
        int b1 = unsigned(raw, src + 1);
        int n0 = high != 0 ? b0 >> 4 : b0 & 0xF;
        int n1 = high != 0 ? b1 >> 4 : b1 & 0xF;
        return n0 | (n1 << 4);
    }

    /** Repacks the {@code n x k} Q4_0 weight in {@code raw} into {@code packed}. */
    public static void repackQ4_0(
            KernelContext context, ByteArray raw, ByteArray packed, int n, int k) {
        int id = context.globalIdx;
        int rounds = k >> 6;
        int tiles = (n >> 7) * rounds;
        if (id < tiles * Q4_0_LANES_PER_TILE) {
            int tile = id / Q4_0_LANES_PER_TILE;
            int lane = id - tile * Q4_0_LANES_PER_TILE;
            int colTile = tile / rounds;
            int round = tile - colTile * rounds;
            int rowBytes = (k >> 5) * Q4_0_BLOCK_BYTES;
            int base = tile * Int8GemmKernels.PACKED_Q4_TILE_BYTES;
            if (lane < 512) {
                int kRow = lane & 15;
                int colPair = (lane >> 4) & 31;
                int high = kRow >> 3;
                int byteInBlock = 2 + ((kRow << 1) & 15);
                int[] groups = new int[4];
                for (int s = 0; s < 4; s++) {
                    int col = (colTile << 7) + ((s & 1) << 6) + (colPair << 1);
                    int src =
                            col * rowBytes
                                    + ((round << 1) + (s >> 1)) * Q4_0_BLOCK_BYTES
                                    + byteInBlock;
                    groups[s] =
                            nibblePair(raw, src, high)
                                    | (nibblePair(raw, src + rowBytes, high) << 8);
                }
                int dst = base + (lane << 3);
                packed.setInt(dst, groups[0] | (groups[1] << 16));
                packed.setInt(dst + 4, groups[2] | (groups[3] << 16));
            } else {
                int j = lane - 512;
                int col = (colTile << 7) + (j & 127);
                int src = col * rowBytes + ((round << 1) + (j >> 7)) * Q4_0_BLOCK_BYTES;
                int dst = base + Int8GemmKernels.PACKED_Q4_QUANT_BYTES + (j << 1);
                packed.set(dst, raw.get(src));
                packed.set(dst + 1, raw.get(src + 1));
            }
        }
    }

    /** {@link #copy}, unless {@code done[0]} says the weights are already packed. */
    public static void copyOnce(
            KernelContext context, ByteArray src, ByteArray dst, int bytes, IntArray done) {
        if (done.get(0) == 0) {
            copy(context, src, dst, bytes);
        }
    }

    /** {@link #repackQ8_0}, unless {@code done[0]} says the weights are already packed. */
    public static void repackQ8_0Once(
            KernelContext context, ByteArray raw, ByteArray packed, int n, int k, IntArray done) {
        if (done.get(0) == 0) {
            repackQ8_0(context, raw, packed, n, k);
        }
    }

    /** {@link #repackQ4_0}, unless {@code done[0]} says the weights are already packed. */
    public static void repackQ4_0Once(
            KernelContext context, ByteArray raw, ByteArray packed, int n, int k, IntArray done) {
        if (done.get(0) == 0) {
            repackQ4_0(context, raw, packed, n, k);
        }
    }

    /** Marks the weights packed: the last task of the repack graph. */
    public static void markDone(KernelContext context, IntArray done) {
        if (context.globalIdx == 0) {
            done.set(0, 1);
        }
    }
}
