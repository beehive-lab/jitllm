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
 * {@link Gemma4BatchedPrefillNllScreenAccelTest} on the Q4_0 file — the one that exercises the
 * 18-byte and 20-byte block decoders, and whose per-layer embedding table is Q5_K.
 *
 * <p>Its own class because surefire forks per class here, and two Gemma 4 fixtures in one JVM
 * exhaust a 24 GB device: the second plan asks for 67,108,896 bytes and is refused while the first
 * model is still resident. The same reason the Q4_0 parity gate is a separate class.
 */
// @formatter:on
public class Gemma4Q4_0BatchedPrefillNllScreenAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void theBatchedPrefillDoesNotMakeHeldOutTextLessLikely() throws Exception {
        Gemma4BatchedPrefillNllScreenAccelTest.screen(
                Fixture.GEMMA_4_E2B_Q4_0, Gemma4BatchedPrefillNllScreenAccelTest.SHORT);
    }
}
