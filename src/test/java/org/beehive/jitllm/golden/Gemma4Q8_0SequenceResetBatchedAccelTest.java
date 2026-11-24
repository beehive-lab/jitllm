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
package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * The reset property for Gemma 4's Q8_0 file on the batched-prefill path, which is what the
 * benchmark runs: a fresh sequence, a longer different one, a reset, and the first sequence again,
 * compared bit for bit.
 *
 * <p>A width of seven, so the prompt is ingested as whole chunks and a partial final one, and the
 * polluting sequence leaves key/value entries beyond every position the replay reads.
 *
 * <p>Its own class, and therefore its own JVM: device memory a closed session frees returns to
 * TornadoVM's buffer provider rather than to the driver.
 */
public class Gemma4Q8_0SequenceResetBatchedAccelTest extends Qwen35SequenceReset {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Override
    Fixture fixture() {
        return Fixture.GEMMA_4_E2B_Q8_0;
    }

    @Test
    public void gemma4E2bQ80ResetRestoresTheSequenceBatched() throws Exception {
        assertResetRestoresTheSequence(7);
    }
}
