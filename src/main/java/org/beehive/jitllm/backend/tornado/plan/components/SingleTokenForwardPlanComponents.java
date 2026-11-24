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

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanSingleToken;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.SingleTokenForwardPlan;
import org.beehive.jitllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;

// @formatter:off
/**
 * The necessary components that any model+quantization combination should implement to support
 * *single-token inference*.
 *
 * <p>Single-token inference with TornadoVM is implemented by {@link
 * TornadoVMMasterPlanSingleToken}. It employees a {@link SingleTokenForwardPlan} instance to
 * represent the complete single-token forward operation as a chain of distinct TornadoVM
 * TaskGraphs. The components of this chain are represented by the following components:
 *
 * <ul>
 *   <li>{@link #singleTokenActivation()} — embedding lookup → FP32 activation (the first inference
 *       graph)
 *   <li>{@link #singleTokenTransformerLayers()} — N transformer layer TaskGraphs
 *   <li>{@link #singleTokenLogits(String)} — final RMSNorm + vocabulary projection (the last graph)
 * </ul>
 *
 * <p>The prepacking graphs, when the plan uses weight prepacking, come before them; {@link
 * SingleTokenForwardTaskGraphLayout} gives every position.
 */
// @formatter:on
public interface SingleTokenForwardPlanComponents {

    ActivationTaskGraph singleTokenActivation();

    TransformerLayerTaskGraphs singleTokenTransformerLayers();

    AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId);
}
