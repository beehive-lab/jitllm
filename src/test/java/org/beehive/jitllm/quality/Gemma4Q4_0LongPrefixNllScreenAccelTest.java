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
package org.beehive.jitllm.quality;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * {@link Gemma4BatchedPrefillNllScreenAccelTest} on the Q4_0 file with a prefix past the sliding
 * window, at the width the benchmark runs.
 *
 * <p>The short screen's 320-token prefix sits inside the 512-position window, so a sliding layer
 * never drops a key there and a mistake in where a chunk's window starts cannot show. This one
 * ingests 1100 positions in chunks of 512 — two whole chunks and a partial one — so every chunk
 * after the first attends a window that begins inside an earlier chunk, and then scores decode
 * positions whose window has moved past everything the first chunk wrote.
 *
 * <p>On the default FP16 key/value cache, which the host reference then holds in FP16 too; the
 * short screens keep the FP32 one. Same limit as the short screen. Its own class, and so its own
 * JVM: device memory a closed session frees returns to TornadoVM's buffer provider rather than to
 * the driver.
 */
// @formatter:on
public class Gemma4Q4_0LongPrefixNllScreenAccelTest {

    static final Gemma4BatchedPrefillNllScreenAccelTest.Schedule LONG =
            new Gemma4BatchedPrefillNllScreenAccelTest.Schedule(512, 1100, 48, 9000);

    @Test
    public void aPrefixPastTheWindowDoesNotMakeHeldOutTextLessLikely() throws Exception {
        Gemma4BatchedPrefillNllScreenAccelTest.screen(Fixture.GEMMA_4_E2B_Q4_0, LONG);
    }
}
