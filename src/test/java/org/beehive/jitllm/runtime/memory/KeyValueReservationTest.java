/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.runtime.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

public class KeyValueReservationTest {

    /** Qwen3-0.6B: 28 layers, kvDim 1024. The pool that exhausted the device at context 8192. */
    @Test
    public void aPoolReservesEverySessionPlusItsScratchBlock() {
        var pool = new KeyValueReservation(8, true, 16, false);
        assertEquals((8L * 512 + 1) * 16 * 28 * 1024, pool.elementsPerArray(8192, 28, 1024));
        var single = new KeyValueReservation(1, true, 16, false);
        assertEquals((512L + 1) * 16 * 28 * 1024, single.elementsPerArray(8192, 28, 1024));
    }

    @Test
    public void privateStorageIsOneBlockRoundedSessionRegardlessOfTheCount() {
        long oneSession = 32L * 16 * 28 * 1024; // 500 tokens occupy 32 whole blocks
        assertEquals(
                oneSession,
                KeyValueReservation.singlePrivate(false).elementsPerArray(500, 28, 1024));
        assertEquals(
                oneSession,
                new KeyValueReservation(4, false, 16, false).elementsPerArray(500, 28, 1024));
    }

    @Test
    public void rejectsAnEmptyReservation() {
        assertThrows(
                IllegalArgumentException.class, () -> new KeyValueReservation(0, true, 16, false));
        assertThrows(
                IllegalArgumentException.class, () -> new KeyValueReservation(1, false, 0, false));
    }
}
