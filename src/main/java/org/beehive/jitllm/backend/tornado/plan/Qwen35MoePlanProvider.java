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

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.Qwen35MoePlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code qwen35moe} plan components: the {@code qwen35} layer graphs, whose feed-forward is the
 * routed and shared experts. Q8_0 only — the expert kernels read Q8_0 blocks.
 */
public final class Qwen35MoePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen35moe");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q8_0);
    }

    /** Its trunk projections are Qwen3.5's, read prepacked the same way. */
    @Override
    public Set<DataType> prepackableTypes(ExecutionPolicy policy) {
        return Set.of(DataType.Q8_0, DataType.Q4_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen35MoeState typed = PlanStates.expect(Qwen35MoeState.class, state, ID);
        return new Qwen35MoePlanComponents(typed, model);
    }

    /** The layers index the caches and recurrent state by absolute layer. */
    @Override
    public boolean stageCacheHoldsOnlyItsLayers() {
        return false;
    }

    /** Every stage's layers are built from a range, single-token and batched alike. */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.empty();
    }

    /** The qwen35 layers' batched prefill, with the experts in place of the dense feed-forward. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return Qwen35PlanProvider.batchPrefillOn(c);
    }

    /** The qwen35 layers' FP16 key/value cache. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }
}
