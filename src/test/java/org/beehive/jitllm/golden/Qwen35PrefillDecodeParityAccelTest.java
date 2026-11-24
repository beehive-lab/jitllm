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
 * Qwen3.8-27B with the prompt ingested as its own sequential phase, against the CPU reference.
 *
 * <p>Its own class, and therefore its own JVM: this fixture holds 15.5 GiB on the device, and
 * device memory a closed session frees returns to TornadoVM's buffer provider rather than to the
 * driver — two of these in one process exhausts the card partway through the second, which is what
 * {@link CpuGpuParity} says and what a shared class demonstrated.
 *
 * <p>Same bounds as the single-token path, because it is the same computation. A mode that needed
 * looser bounds would be computing something else, and that would be the finding rather than the
 * configuration.
 */
public class Qwen35PrefillDecodeParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen3_8_27b_q4_0_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.QWEN3_8_27B_Q4_0, Q8_0_FULLY_PACKED);
    }
}
