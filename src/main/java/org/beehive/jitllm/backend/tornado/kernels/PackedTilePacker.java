package org.beehive.jitllm.backend.tornado.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.stream.IntStream;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * Host-side repacking of Q8_0 and Q4_0 weights into the tiles of {@link
 * Int8GemmKernels#packQ8_0Tiles} and {@link Int8GemmKernels#packQ4_0Tiles}.
 *
 * <p>A column tile (128 rows of the weight) only reads its own rows, and its packed tiles are one
 * contiguous run of the output, so the column tiles are packed in parallel. Each worker copies its
 * 128 rows off the weight's segment, which pages a memory-mapped model in on many threads instead
 * of one, and moves the quants as 16- and 32-bit words rather than byte by byte.
 */
public final class PackedTilePacker {

    private static final int BN = Int8GemmKernels.I8_BN;
    private static final int BK = Int8GemmKernels.I8_BK;
    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int Q4_0_BLOCK_BYTES = 18;
    private static final int TILE_WORDS = 1024;

    private static final VarHandle SHORT =
            MethodHandles.byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle INT =
            MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle LONG =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    private PackedTilePacker() {}

    public static byte[] packQ8_0(ByteArray w, int n, int k) {
        byte[] out = new byte[(int) packedBytes(false, n, k)];
        pack(false, w, n, k, MemorySegment.ofArray(out));
        return out;
    }

    public static byte[] packQ4_0(ByteArray w, int n, int k) {
        byte[] out = new byte[(int) packedBytes(true, n, k)];
        pack(true, w, n, k, MemorySegment.ofArray(out));
        return out;
    }

    /** Bytes of the packed form of an {@code n x k} Q4_0 ({@code q4}) or Q8_0 weight. */
    public static long packedBytes(boolean q4, int n, int k) {
        return (long) (n / BN)
                * (k / BK)
                * (q4 ? Int8GemmKernels.PACKED_Q4_TILE_BYTES : Int8GemmKernels.PACKED_TILE_BYTES);
    }

    /**
     * Packs {@code w} into {@code out}, {@link #packedBytes} long. Each column tile is packed on
     * the heap and copied out whole.
     */
    public static void pack(boolean q4, ByteArray w, int n, int k, MemorySegment out) {
        int rowBytes = (k / 32) * (q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
        int rounds = k / BK;
        int tileBytes =
                q4 ? Int8GemmKernels.PACKED_Q4_TILE_BYTES : Int8GemmKernels.PACKED_TILE_BYTES;
        MemorySegment weights = w.getSegment();
        IntStream.range(0, n / BN)
                .parallel()
                .forEach(
                        colTile -> {
                            byte[] rows = slice(weights, colTile, rowBytes);
                            byte[] tiles = new byte[rounds * tileBytes];
                            for (int round = 0; round < rounds; round++) {
                                if (q4) {
                                    packQ4_0Tile(rows, rowBytes, round, tiles, round * tileBytes);
                                } else {
                                    packQ8_0Tile(rows, rowBytes, round, tiles, round * tileBytes);
                                }
                            }
                            MemorySegment.copy(
                                    tiles,
                                    0,
                                    out,
                                    ValueLayout.JAVA_BYTE,
                                    (long) colTile * tiles.length,
                                    tiles.length);
                        });
    }

    /** The 128 rows of column tile {@code colTile}, copied to the heap. */
    private static byte[] slice(MemorySegment weights, int colTile, int rowBytes) {
        byte[] rows = new byte[BN * rowBytes];
        MemorySegment.copy(
                weights,
                ValueLayout.JAVA_BYTE,
                (long) colTile * BN * rowBytes,
                rows,
                0,
                rows.length);
        return rows;
    }

    // @formatter:off
    /**
     * One Q8_0 tile. Word {@code ((colPair >> 2) << 6) + (kRow << 2) + (colPair & 3)} of a 512-word
     * half holds quants {@code 2 kRow, 2 kRow + 1} of columns {@code 2 colPair} (low half) and
     * {@code 2 colPair + 1} (high half): written here a column pair at a time, the pair's 16 words
     * a stride of four apart.
     */
    // @formatter:on
    private static void packQ8_0Tile(byte[] rows, int rowBytes, int round, byte[] out, int base) {
        for (int block = 0; block < 2; block++) {
            int blockOffset = (round * 2 + block) * Q8_0_BLOCK_BYTES;
            for (int half = 0; half < 2; half++) {
                int words = base + ((block << 10) + (half << 9)) * 4;
                for (int colPair = 0; colPair < 32; colPair++) {
                    int src = (half * 64 + 2 * colPair) * rowBytes + blockOffset + 2;
                    int dst = words + (((colPair >> 2) << 6) + (colPair & 3)) * 4;
                    for (int kRow = 0; kRow < 16; kRow++) {
                        int lo = (short) SHORT.get(rows, src + 2 * kRow) & 0xFFFF;
                        int hi = (short) SHORT.get(rows, src + rowBytes + 2 * kRow);
                        INT.set(out, dst + kRow * 16, lo | (hi << 16));
                    }
                }
            }
        }
        scales(
                rows,
                rowBytes,
                round,
                Q8_0_BLOCK_BYTES,
                out,
                base + Int8GemmKernels.PACKED_QUANT_BYTES);
    }

    // @formatter:off
    /**
     * One Q4_0 tile: GEMM thread {@code t} ({@code kRow = t & 15}, {@code colPair = (t >> 4) & 31})
     * finds at byte {@code 8t} four 16-bit nibble groups, group {@code s} holding quants {@code 2
     * kRow, 2 kRow + 1} of columns {@code c, c + 1}, {@code c = (s & 1) * 64 + 2 colPair}, of the
     * round's block {@code s >> 1}. Quants {@code q, q + 1} are the low nibbles of bytes {@code q,
     * q + 1} when {@code q < 16}, else the high nibbles of bytes {@code q - 16, q - 15}: one 16-bit
     * load per column, its two nibbles pulled together.
     */
    // @formatter:on
    private static void packQ4_0Tile(byte[] rows, int rowBytes, int round, byte[] out, int base) {
        for (int t = 0; t < 512; t++) {
            int kRow = t & 15;
            int colPair = (t >> 4) & 31;
            int shift = kRow < 8 ? 0 : 4;
            int byteInBlock = 2 + ((kRow << 1) & 15);
            long nibbles = 0L;
            for (int s = 0; s < 4; s++) {
                int src =
                        ((s & 1) * 64 + (colPair << 1)) * rowBytes
                                + (round * 2 + (s >> 1)) * Q4_0_BLOCK_BYTES
                                + byteInBlock;
                int group = pair(rows, src, shift) | (pair(rows, src + rowBytes, shift) << 8);
                nibbles |= (long) group << (s << 4);
            }
            LONG.set(out, base + (t << 3), nibbles);
        }
        scales(
                rows,
                rowBytes,
                round,
                Q4_0_BLOCK_BYTES,
                out,
                base + Int8GemmKernels.PACKED_Q4_QUANT_BYTES);
    }

    /**
     * The two nibbles at {@code shift} of the bytes at {@code src}, {@code src + 1}, as one byte.
     */
    private static int pair(byte[] rows, int src, int shift) {
        int v = (((short) SHORT.get(rows, src) & 0xFFFF) >>> shift) & 0x0F0F;
        return (v | (v >>> 4)) & 0xFF;
    }

    /**
     * The round's 256 block scales, block-major: scale {@code j} is column {@code j & 127} of block
     * {@code j >> 7}.
     */
    private static void scales(
            byte[] rows, int rowBytes, int round, int blockBytes, byte[] out, int dst) {
        for (int j = 0; j < 2 * BN; j++) {
            int src = (j & 127) * rowBytes + (round * 2 + (j >> 7)) * blockBytes;
            SHORT.set(out, dst + 2 * j, (short) SHORT.get(rows, src));
        }
    }
}
