package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The host decode of {@link Gemma4Fp16Weights}: for Q8_0, Q4_0 and Q4_1 blocks, every element is
 * the FP16 rounding of the block's value computed from the file's layout independently here — scale
 * times the signed byte, the nibble minus 8, or scale times the nibble plus the minimum — and a
 * stack places the second tensor's rows after the first's.
 */
// @formatter:on
public class Gemma4Fp16WeightsTest {

    @Test
    public void stackedBlocksDecodeToTheirFp16Values() {
        int k = 96;
        int n0 = 5;
        int n1 = 3;
        int n2 = 4;
        Random rng = new Random(11);
        ByteArray q8 = randomBlocks(n0, k, 34, rng);
        ByteArray q4 = randomBlocks(n1, k, 18, rng);
        ByteArray q41 = randomBlocks(n2, k, 20, rng);
        HalfFloatArray out =
                Gemma4Fp16Weights.stack(
                        k,
                        new TornadoTensor[] {
                            tensor(q8, DataType.Q8_0),
                            tensor(q4, DataType.Q4_0),
                            tensor(q41, DataType.Q4_1)
                        },
                        new int[] {n0, n1, n2});
        assertEquals((n0 + n1 + n2) * k, out.getSize());
        for (int r = 0; r < n0 + n1 + n2; r++) {
            for (int c = 0; c < k; c++) {
                float expected;
                int blk = c / 32;
                int i = c % 32;
                if (r < n0) {
                    int off = (r * (k / 32) + blk) * 34;
                    expected = q8.getHalfFloat(off).getFloat32() * q8.get(off + 2 + i);
                } else if (r < n0 + n1) {
                    int off = ((r - n0) * (k / 32) + blk) * 18;
                    int b = q4.get(off + 2 + (i % 16)) & 0xFF;
                    int q = i < 16 ? b & 0x0F : b >>> 4;
                    expected = q4.getHalfFloat(off).getFloat32() * (q - 8);
                } else {
                    int off = ((r - n0 - n1) * (k / 32) + blk) * 20;
                    int b = q41.get(off + 4 + (i % 16)) & 0xFF;
                    int q = i < 16 ? b & 0x0F : b >>> 4;
                    expected =
                            q41.getHalfFloat(off).getFloat32() * q
                                    + q41.getHalfFloat(off + 2).getFloat32();
                }
                assertEquals(
                        "row " + r + " col " + c,
                        new HalfFloat(expected).getHalfFloatValue(),
                        out.get(r * k + c).getHalfFloatValue());
            }
        }
    }

    private static ByteArray randomBlocks(int rows, int k, int blockBytes, Random rng) {
        int blocks = rows * (k / 32);
        ByteArray out = new ByteArray(blocks * blockBytes);
        for (int blk = 0; blk < blocks; blk++) {
            int off = blk * blockBytes;
            out.setHalfFloat(off, new HalfFloat(0.01f + rng.nextFloat() * 0.05f));
            int from = 2;
            if (blockBytes == 20) {
                out.setHalfFloat(off + 2, new HalfFloat(-0.3f + rng.nextFloat() * 0.1f));
                from = 4;
            }
            for (int i = from; i < blockBytes; i++) {
                out.set(off + i, (byte) rng.nextInt(256));
            }
        }
        return out;
    }

    private static TornadoTensor tensor(ByteArray bytes, DataType type) {
        return new TornadoTensor() {
            @Override
            public DataType dataType() {
                return type;
            }

            @Override
            public ByteArray asByteArray() {
                return bytes;
            }

            @Override
            public GGMLType type() {
                return switch (type) {
                    case Q8_0 -> GGMLType.Q8_0;
                    case Q4_0 -> GGMLType.Q4_0;
                    default -> GGMLType.Q4_1;
                };
            }
        };
    }
}
