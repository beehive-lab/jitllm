package org.beehive.jitllm.backend.tornado.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import org.junit.Test;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

public class Q8_0TornadoTensorTest {

    @Test
    public void repacksScalesAndQuantsIntoAlignedGroups() {
        ByteArray interleaved = new ByteArray(68);
        interleaved.setHalfFloat(0, new HalfFloat(0.5f));
        interleaved.setHalfFloat(34, new HalfFloat(0.25f));
        for (int i = 0; i < 32; i++) {
            interleaved.set(2 + i, (byte) (i - 16));
            interleaved.set(36 + i, (byte) (31 - i));
        }

        Q8_0TornadoTensor tensor = new Q8_0TornadoTensor(interleaved);
        ByteArray repacked = tensor.asRepackedByteArray();

        assertEquals(544, repacked.getSize());
        assertEquals(0.5f, repacked.getHalfFloat(0).getFloat32(), 0.0f);
        assertEquals(0.25f, repacked.getHalfFloat(2).getFloat32(), 0.0f);
        for (int i = 0; i < 32; i++) {
            assertEquals((byte) (i - 16), repacked.get(32 + i));
            assertEquals((byte) (31 - i), repacked.get(64 + i));
        }
        assertSame(repacked, tensor.asRepackedByteArray());
    }
}
