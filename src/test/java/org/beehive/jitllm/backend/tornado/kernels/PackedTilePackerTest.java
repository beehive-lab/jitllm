package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertArrayEquals;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/** The parallel packer against the byte-at-a-time packing it replaced. */
public class PackedTilePackerTest {

    private static final int[][] SHAPES = {{128, 64}, {256, 192}, {384, 1024}, {1024, 5120}};

    @Test
    public void q8_0MatchesTheReference() {
        for (int[] shape : SHAPES) {
            ByteArray w = random(shape[0] * (shape[1] / 32) * 34, shape[0] + shape[1]);
            assertArrayEquals(
                    shape[0] + "x" + shape[1],
                    referenceQ8_0(w, shape[0], shape[1]),
                    Int8GemmKernels.packQ8_0TileBytes(w, shape[0], shape[1]));
        }
    }

    @Test
    public void q4_0MatchesTheReference() {
        for (int[] shape : SHAPES) {
            ByteArray w = random(shape[0] * (shape[1] / 32) * 18, shape[0] * 3 + shape[1]);
            assertArrayEquals(
                    shape[0] + "x" + shape[1],
                    referenceQ4_0(w, shape[0], shape[1]),
                    Int8GemmKernels.packQ4_0TileBytes(w, shape[0], shape[1]));
        }
    }

    /** Packing time of a Gemma 4 31B FFN weight, old against new; printed, not asserted. */
    @Test
    public void timing() {
        int n = 21504;
        int k = 5376;
        ByteArray q8 = random(n * (k / 32) * 34, 1);
        ByteArray q4 = random(n * (k / 32) * 18, 2);
        for (int rep = 0; rep < 3; rep++) {
            long t0 = System.nanoTime();
            referenceQ8_0(q8, n, k);
            long t1 = System.nanoTime();
            Int8GemmKernels.packQ8_0TileBytes(q8, n, k);
            long t2 = System.nanoTime();
            referenceQ4_0(q4, n, k);
            long t3 = System.nanoTime();
            Int8GemmKernels.packQ4_0TileBytes(q4, n, k);
            long t4 = System.nanoTime();
            System.out.printf(
                    "PACKTIME %dx%d q8 ref %.0f ms new %.0f ms | q4 ref %.0f ms new %.0f ms%n",
                    n, k, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t4 - t3) / 1e6);
        }
    }

    private static ByteArray random(int bytes, long seed) {
        byte[] data = new byte[bytes];
        new Random(seed).nextBytes(data);
        return ByteArray.fromArray(data);
    }

    private static byte[] referenceQ8_0(ByteArray w, int n, int k) {
        int kBlocks = k / 32;
        int rowBytes = kBlocks * 34;
        int rounds = k / 64;
        int tiles = (n / 128) * rounds;
        byte[] raw = w.toHeapArray();
        byte[] out = new byte[tiles * Int8GemmKernels.PACKED_TILE_BYTES];
        for (int colTile = 0; colTile < n / 128; colTile++) {
            for (int round = 0; round < rounds; round++) {
                int base = (colTile * rounds + round) * Int8GemmKernels.PACKED_TILE_BYTES;
                for (int word = 0; word < 2 * 1024; word++) {
                    int block = round * 2 + (word >> 10);
                    int half = (word >> 9) & 1;
                    int slotIndex = word & 511;
                    int colPair = ((slotIndex >> 6) << 2) | (slotIndex & 3);
                    int kRow = (slotIndex >> 2) & 15;
                    int col = colTile * 128 + half * 64 + 2 * colPair;
                    int src = col * rowBytes + block * 34 + 2 + 2 * kRow;
                    out[base + word * 4] = raw[src];
                    out[base + word * 4 + 1] = raw[src + 1];
                    out[base + word * 4 + 2] = raw[src + rowBytes];
                    out[base + word * 4 + 3] = raw[src + rowBytes + 1];
                }
                for (int j = 0; j < 2 * 128; j++) {
                    int col = colTile * 128 + (j & 127);
                    int block = round * 2 + (j >> 7);
                    int src = col * rowBytes + block * 34;
                    int dst = base + Int8GemmKernels.PACKED_QUANT_BYTES + j * 2;
                    out[dst] = raw[src];
                    out[dst + 1] = raw[src + 1];
                }
            }
        }
        return out;
    }

    private static byte[] referenceQ4_0(ByteArray w, int n, int k) {
        int kBlocks = k / 32;
        int rowBytes = kBlocks * 18;
        int rounds = k / 64;
        byte[] raw = w.toHeapArray();
        byte[] out = new byte[(n / 128) * rounds * Int8GemmKernels.PACKED_Q4_TILE_BYTES];
        for (int colTile = 0; colTile < n / 128; colTile++) {
            for (int round = 0; round < rounds; round++) {
                int base = (colTile * rounds + round) * Int8GemmKernels.PACKED_Q4_TILE_BYTES;
                for (int t = 0; t < 512; t++) {
                    int kRow = t & 15;
                    int colPair = (t >> 4) & 31;
                    long nibbles = 0L;
                    for (int s = 0; s < 4; s++) {
                        int block = round * 2 + (s >> 1);
                        int col = colTile * 128 + (s & 1) * 64 + (colPair << 1);
                        int q = kRow << 1; // quants q and q + 1 of the block
                        long group = 0L;
                        for (int e = 0; e < 4; e++) {
                            int c = col + (e >> 1);
                            int qi = q + (e & 1);
                            int b = raw[c * rowBytes + block * 18 + 2 + (qi & 15)] & 0xFF;
                            int nibble = qi < 16 ? b & 0xF : b >>> 4;
                            group |= (long) nibble << (e << 2);
                        }
                        nibbles |= group << (s << 4);
                    }
                    for (int i = 0; i < 8; i++) {
                        out[base + (t << 3) + i] = (byte) (nibbles >>> (i << 3));
                    }
                }
                for (int j = 0; j < 2 * 128; j++) {
                    int src = (colTile * 128 + (j & 127)) * rowBytes + (round * 2 + (j >> 7)) * 18;
                    out[base + Int8GemmKernels.PACKED_Q4_QUANT_BYTES + 2 * j] = raw[src];
                    out[base + Int8GemmKernels.PACKED_Q4_QUANT_BYTES + 2 * j + 1] = raw[src + 1];
                }
            }
        }
        return out;
    }
}
