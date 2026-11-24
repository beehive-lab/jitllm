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
 * Qwen3.8-27B ingested in chunks of 64, against the CPU reference.
 *
 * <p>Wider than the prompt: a single chunk, most of it inactive. The active count comes from the
 * chunk rather than from the launch width, and this is where that distinction is load-bearing.
 *
 * <p>One width per class, and therefore per JVM. This fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process exhausts the card. The widths are separated rather than looped.
 */
public class Qwen35BatchedPrefillWidth64ParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static {
        // The scalar batched path, which the tensor-core default would otherwise replace: this
        // class is the scalar kernels' coverage; the Qwen35Mma* classes cover the tensor cores.
        System.setProperty("jitllm.qwen35.tensorCores", "false");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt64() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 64);
    }
}
