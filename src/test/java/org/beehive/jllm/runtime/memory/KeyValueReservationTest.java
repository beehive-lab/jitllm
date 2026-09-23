package org.beehive.jllm.runtime.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

public class KeyValueReservationTest {

    /** Qwen3-0.6B: 28 layers, kvDim 1024. The pool that exhausted the device at context 8192. */
    @Test
    public void aPoolReservesEverySessionPlusItsScratchBlock() {
        var pool = new KeyValueReservation(8, true, 16);
        assertEquals((8L * 512 + 1) * 16 * 28 * 1024, pool.elementsPerArray(8192, 28, 1024));
        var single = new KeyValueReservation(1, true, 16);
        assertEquals((512L + 1) * 16 * 28 * 1024, single.elementsPerArray(8192, 28, 1024));
    }

    @Test
    public void privateStorageIsOneBlockRoundedSessionRegardlessOfTheCount() {
        long oneSession = 32L * 16 * 28 * 1024; // 500 tokens occupy 32 whole blocks
        assertEquals(
                oneSession, KeyValueReservation.singlePrivate().elementsPerArray(500, 28, 1024));
        assertEquals(
                oneSession, new KeyValueReservation(4, false, 16).elementsPerArray(500, 28, 1024));
    }

    @Test
    public void rejectsAnEmptyReservation() {
        assertThrows(IllegalArgumentException.class, () -> new KeyValueReservation(0, true, 16));
        assertThrows(IllegalArgumentException.class, () -> new KeyValueReservation(1, false, 0));
    }
}
