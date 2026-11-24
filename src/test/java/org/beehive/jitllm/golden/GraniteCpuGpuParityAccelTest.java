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

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Granite's logits against the CPU reference. See {@link CpuGpuParity}. */
public class GraniteCpuGpuParityAccelTest extends CpuGpuParity {

    /**
     * The packed-integer path off, as in {@link Qwen3CpuGpuParityAccelTest}, which says what it
     * computes instead and why these bounds do not describe it.
     */
    static {
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void granite_3_2_2b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.GRANITE_3_2_2B_F16, FP16);
    }

    @Test
    public void granite_3_2_2b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.GRANITE_3_2_2B_Q8_0, Q8_0);
    }

    /** A Q4_K_M file kept as it is: Q4_K and Q6_K read per tensor. */
    @Test
    public void granite_3_2_2b_q4_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.GRANITE_3_2_2B_Q4_K_M, Q8_0);
    }
}
