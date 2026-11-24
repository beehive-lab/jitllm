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
package org.beehive.jitllm.api;

import static org.junit.Assert.assertNotNull;

import org.beehive.jitllm.backend.tornado.PlanDispatchEvidence;

/**
 * {@link Qwen35BatchedFp16KvLifecycleAccelTest} with the tensor-core path selected.
 *
 * <p>The FP16 key/value lifecycle and its session reset over the MMA batched prefill. The property
 * selects the path; {@link #verifyDispatch} checks it against the scheduler of the plan this
 * session generated with, and fails if that evidence is missing.
 */
public class Qwen35MmaBatchedFp16KvLifecycleAccelTest
        extends Qwen35BatchedFp16KvLifecycleAccelTest {

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
    }

    @Override
    protected void verifyDispatch(DelegatingSession session, int batch, int dim) {
        var plan = session.planIfBuilt();
        assertNotNull("the session generated without building a plan", plan);
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan), batch, dim);
    }
}
