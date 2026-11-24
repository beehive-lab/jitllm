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
 * {@link Qwen35SequenceResetStandardAccelTest} with a half-precision key/value cache, which is the
 * path that decodes with split-KV attention.
 *
 * <p>The reset case is the one that matters for a split plan: the per-head partials live in scratch
 * that is reused across tokens and across sequences, so a reset that left a longer run's partials
 * behind would show here as the second sequence disagreeing with the first.
 */
public class Qwen35Fp16KvSequenceResetStandardAccelTest extends Qwen35SequenceReset {

    static {
        System.setProperty("jitllm.kvcache.fp16", "true");
    }

    @Override
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {
        PlanDispatchEvidence.assertQwen35DecodeAttentionForBackend(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
    }

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceStandardOnFp16Kv() throws Exception {
        assertResetRestoresTheSequence(1);
    }
}
