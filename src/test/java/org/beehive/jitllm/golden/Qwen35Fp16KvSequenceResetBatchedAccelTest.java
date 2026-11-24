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
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.junit.Test;

/**
 * {@link Qwen35SequenceResetBatchedAccelTest} with a half-precision key/value cache.
 *
 * <p>The batched prefill/decode plan reaches the same decode attention, so this is the reset case
 * for split-KV under the mode the benchmarks run.
 */
public class Qwen35Fp16KvSequenceResetBatchedAccelTest extends Qwen35SequenceReset {

    static {
        System.setProperty("jitllm.kvcache.fp16", "true");
    }

    @Override
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {
        PlanDispatchEvidence.assertQwen35DecodeAttentionForBackend(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
    }

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceBatchedOnFp16Kv() throws Exception {
        assertResetRestoresTheSequence(32);
    }
}
