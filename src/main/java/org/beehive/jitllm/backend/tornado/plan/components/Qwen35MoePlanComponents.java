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
package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.Qwen35BatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeBatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeFeedForward;
import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.weights.tornado.Qwen35MoeTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen35moe.Qwen35Moe;

/**
 * The {@code qwen35moe} plan components: the {@code qwen35} layer graphs, each layer's feed-forward
 * the routed and shared experts.
 */
public class Qwen35MoePlanComponents extends Qwen35PlanComponents {

    private final Qwen35MoeState state;
    private final Qwen35MoeTornadoWeights weights;
    private final Qwen35Moe model;

    public Qwen35MoePlanComponents(Qwen35MoeState state, Model model) {
        super(state, model);
        this.state = state;
        this.weights = (Qwen35MoeTornadoWeights) model.weights();
        this.model = (Qwen35Moe) model;
    }

    @Override
    protected Qwen35FeedForward feedForward() {
        return new Qwen35MoeFeedForward(
                state.workspace, model.experts(), weights.experts(), model.configuration().dim());
    }

    @Override
    protected Qwen35BatchFeedForward batchFeedForward(int batchSize) {
        return new Qwen35MoeBatchFeedForward(
                state.workspace,
                model.experts(),
                weights.experts(),
                model.configuration().dim(),
                batchSize);
    }
}
