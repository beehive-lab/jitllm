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

import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;

/**
 * Qwen3.5-0.8B Q4_K_M and Q4_0 on Metal against the CPU reference, with the prompt ingested in
 * chunks of 8.
 *
 * <p>Each projection runs by its own type. In the batched prefill the K-quant and Q4_1 weights take
 * the SIMD-group GEMMs the other families use, which stage the weights as FP16, so the rows are
 * held to {@link #FP16}. The Q4_K_M file mixes Q4_K, Q5_K, Q6_K and Q8_0 projections; both files
 * keep their embeddings Q6_K.
 *
 * <p>Uses the files under the fixture directory, by name; skips when one is absent or the backend
 * is not Metal.
 */
public class Qwen35MetalBatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen35_0_8b_q4_k_m_batchedPrefillParityOnMetal() throws Exception {
        assertParityBatched(metalFile("Qwen3.5-0.8B-Q4_K_M.gguf"), "Q4_K_M", FP16, 8);
    }

    @Test
    public void qwen35_0_8b_q4_0_batchedPrefillParityOnMetal() throws Exception {
        assertParityBatched(metalFile("Qwen3.5-0.8B-Q4_0.gguf"), "Q4_0", FP16, 8);
    }

    private static Path metalFile(String model) {
        Path file = GoldenFixture.modelsRoot().resolve(model);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        BackendId backend = TornadoDevices.current().backend();
        assumeTrue(
                "this runs on the " + backend + " backend; the kernels are Metal's",
                BackendId.METAL.equals(backend));
        return file;
    }
}
