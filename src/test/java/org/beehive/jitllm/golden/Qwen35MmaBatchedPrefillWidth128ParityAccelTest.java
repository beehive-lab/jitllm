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

import org.beehive.jitllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * The batched prefill parity at width 128 with the tensor-core path selected — the width at which
 * the Q4_0 projections wide enough for it run as a dequantization into FP16 scratch followed by the
 * tiled FP16 GEMM. The 63-token fixture is one partial chunk of the 128-row launch, so the pair's
 * inactive rows are computed and never read. The dispatch is asserted off this run's own plan: the
 * attention-output projection's dequantization task and GEMM grid in every attention layer.
 */
// @formatter:on
public class Qwen35MmaBatchedPrefillWidth128ParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt128OnTensorCores() throws Exception {
        GoldenCapture.Result gpu =
                assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 128);
        PlanDispatchEvidence.assertQwen35AttentionOutputOnGemm(
                gpu.gridScheduler, 128, gpu.dim, 6144);
        PlanDispatchEvidence.assertQwen35SsmOutOnDequantGemm(gpu.gridScheduler, 128, gpu.dim, 6144);
    }
}
