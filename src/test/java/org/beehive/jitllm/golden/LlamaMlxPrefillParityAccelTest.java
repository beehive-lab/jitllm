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
 * Llama-3.2-1B against the CPU reference with its batched prefill on MLX; see {@link
 * MlxPrefillParity}.
 */
public class LlamaMlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void f16() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_F16, FP16, Weights.AS_LOADED);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_Q8_0, FP16, Weights.AFFINE);
    }

    @Test
    public void q8_0Fp16Copies() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_Q8_0, FP16, Weights.FP16_COPIES);
    }
}
