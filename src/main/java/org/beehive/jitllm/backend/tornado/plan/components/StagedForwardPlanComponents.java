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

import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;

// @formatter:off
/**
 * Plan components that can build one stage of a model split by layers across several devices: the
 * layers {@code [firstLayer, endLayer)} on that stage's state.
 *
 * <p>The pipeline plan builds every stage from these, and from the family's own activation and
 * logits components for the first and last stage; it names no family. A family whose components do
 * not implement this is not split, and says so.
 *
 * <p>The first layer of a range takes the hidden state from the graph before it in the stage's
 * plan, which on a later stage is the hand-off that received it. Whether a stage's key/value cache
 * holds its own layers only or the whole model's is the family's decision, made where its stage
 * state is built ({@code TornadoPlanProvider.stageState}) and honoured by its layers.
 */
// @formatter:on
public interface StagedForwardPlanComponents extends SingleTokenForwardPlanComponents {

    /** The single-token layers {@code [firstLayer, endLayer)}. */
    TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer);

    /**
     * The batched prefill layers {@code [firstLayer, endLayer)}, for a family that is also {@link
     * BatchPrefillDecodeForwardPlanComponents}.
     */
    default BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        throw new UnsupportedOperationException(
                "this family's batched prefill is not split across devices; drop"
                        + " --batch-prefill-size to split its single-token plan");
    }

    /** The decode layers {@code [firstLayer, endLayer)} that follow a staged batched prefill. */
    default TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        throw new UnsupportedOperationException(
                "this family's batched prefill is not split across devices; drop"
                        + " --batch-prefill-size to split its single-token plan");
    }
}
