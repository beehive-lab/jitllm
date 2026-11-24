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

/**
 * Gemma 4's Q4_K_M file, kept as it is, against the CPU reference with the prompt ingested through
 * the batched prefill graphs, then decoded.
 *
 * <p>The prompt rows go through FP16-operand GEMMs over the weights decoded per type (tensor cores
 * on CUDA, the {@code batchedGemmHalf…Strided} SIMD-group kernels on Metal), which is the
 * arithmetic {@link CpuGpuParity#GEMMA4_Q8_0_BATCHED} was measured for; the width of seven is
 * {@link Gemma4Q8_0BatchedPrefillParityAccelTest}'s. The packed-integer projections are off, as in
 * {@link Gemma4KQuantParityAccelTest}, so the decoded rows compute in floating point.
 *
 * <p>Its own class, and therefore its own JVM: device memory a closed session frees returns to
 * TornadoVM's buffer provider rather than to the driver.
 */
public class Gemma4KQuantBatchedPrefillParityAccelTest extends CpuGpuParity {

    static {
        System.setProperty("jitllm.gemma4.packedIntegerDot", "false");
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ4_K_MBatchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.GEMMA_4_E2B_Q4_K_M, GEMMA4_Q8_0_BATCHED, 7);
    }
}
