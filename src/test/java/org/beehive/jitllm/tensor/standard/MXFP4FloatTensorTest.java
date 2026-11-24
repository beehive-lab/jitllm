package org.beehive.jitllm.tensor.standard;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/** MXFP4 decoded as llama.cpp's {@code dequantize_row_mxfp4}, element by element and in a dot. */
public class MXFP4FloatTensorTest {

    /**
     * E2M1 values, the reference for the table: twice these are {@link MXFP4FloatTensor#KVALUES}.
     */
    private static final float[] E2M1 = {
        0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f
    };

    private static MemorySegment blocks(Random random, int blocks, byte[] exps, int[][] nibbles) {
        MemorySegment segment = Arena.ofAuto().allocate(blocks * 17L);
        for (int b = 0; b < blocks; b++) {
            exps[b] = (byte) (120 + random.nextInt(10));
            segment.set(ValueLayout.JAVA_BYTE, b * 17L, exps[b]);
            for (int j = 0; j < 16; j++) {
                int lo = random.nextInt(16);
                int hi = random.nextInt(16);
                nibbles[b][j] = lo;
                nibbles[b][j + 16] = hi;
                segment.set(ValueLayout.JAVA_BYTE, b * 17L + 1 + j, (byte) (lo | (hi << 4)));
            }
        }
        return segment;
    }

    @Test
    public void decodesEachElementAsE2M1TimesTheSharedPowerOfTwo() {
        Random random = new Random(7);
        int blocks = 3;
        byte[] exps = new byte[blocks];
        int[][] nibbles = new int[blocks][32];
        MXFP4FloatTensor t =
                new MXFP4FloatTensor(blocks * 32, blocks(random, blocks, exps, nibbles));
        assertEquals(DataType.MXFP4, t.dataType());
        for (int b = 0; b < blocks; b++) {
            float scale = (float) Math.pow(2, Byte.toUnsignedInt(exps[b]) - 127);
            for (int j = 0; j < 32; j++) {
                assertEquals(E2M1[nibbles[b][j]] * scale, t.getFloat(b * 32 + j), 0f);
            }
        }
    }

    @Test
    public void vectorDotMatchesTheScalarDot() {
        Random random = new Random(11);
        int blocks = 90;
        MXFP4FloatTensor t =
                new MXFP4FloatTensor(
                        blocks * 32, blocks(random, blocks, new byte[blocks], new int[blocks][32]));
        FloatTensor x = ArrayFloatTensor.allocate(blocks * 32);
        for (int i = 0; i < x.size(); i++) {
            x.setFloat(i, random.nextFloat() - 0.5f);
        }
        float scalar = FloatTensor.scalarDot(t, 32, x, 0, 64 * 32);
        assertEquals(scalar, t.dot(32, x, 0, 64 * 32), Math.abs(scalar) * 1e-5f + 1e-6f);
    }
}
