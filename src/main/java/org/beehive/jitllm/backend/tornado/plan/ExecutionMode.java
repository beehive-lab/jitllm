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
package org.beehive.jitllm.backend.tornado.plan;

import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * Selects the GPU execution topology for a TornadoVM inference plan.
 *
 * <ul>
 *   <li>{@link #STANDARD} — single forward pass (activation + N layers + logits).
 *   <li>{@link #PREFILL_DECODE} — shared N+2 graph plan; prefill skips logits, decode runs all.
 *   <li>{@link #BATCH_PREFILL_DECODE} — 2N+3 graph plan; N batch-prefill graphs + N decode graphs +
 *       logits.
 * </ul>
 */
public enum ExecutionMode {
    STANDARD,
    PREFILL_DECODE,
    BATCH_PREFILL_DECODE;

    /** The mode a session running {@code policy} executes in. */
    public static ExecutionMode of(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return STANDARD;
        }
        return policy.prefillBatchSize() > 1 ? BATCH_PREFILL_DECODE : PREFILL_DECODE;
    }
}
// @formatter:on
