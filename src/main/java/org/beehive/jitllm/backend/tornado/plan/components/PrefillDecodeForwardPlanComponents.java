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

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanPrefillDecode;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.PrefillDecodeForwardPlan;
import org.beehive.jitllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout;

// @formatter:off
/**
 * The necessary components that any model+quantization combination should implement to support
 * *prefill-decode inference*.
 *
 * <p>Prefill-decode inference with TornadoVM is implemented by {@link
 * TornadoVMMasterPlanPrefillDecode}. It employs a {@link PrefillDecodeForwardPlan} instance to
 * represent the complete prefill-decode forward operation as a chain of distinct TornadoVM
 * TaskGraphs. The components of this chain are represented by the following components:
 *
 * <ul>
 *   <li>{@link #prefillDecodeActivation()} — KV-cache-aware activation (the first inference graph)
 *   <li>{@link #prefillDecodeTransformerLayers()} — N KV-cache-aware transformer layer TaskGraphs
 *   <li>{@link #decodeLogits(String)} — final RMSNorm + vocabulary projection (the last graph)
 * </ul>
 *
 * <p>The prepacking graphs, when the plan uses weight prepacking, come before them; {@link
 * PrefillDecodeForwardTaskGraphLayout} gives every position.
 */
// @formatter:on
public interface PrefillDecodeForwardPlanComponents extends SingleTokenForwardPlanComponents {

    ActivationTaskGraph prefillDecodeActivation();

    TransformerLayerTaskGraphs prefillDecodeTransformerLayers();

    AbstractLogitsTaskGraph decodeLogits(String previousGraphId);
}
