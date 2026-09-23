package org.beehive.jllm.backend.tornado.plan.components;

import org.beehive.jllm.backend.tornado.TornadoVMMasterPlanSingleToken;
import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.plan.SingleTokenForwardPlan;

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
 *   <li>{@link #singleTokenActivation()} — embedding lookup → FP32 activation (graph 0)
 *   <li>{@link #singleTokenTransformerLayers()} — N transformer layer TaskGraphs (graphs 1.N)
 *   <li>{@link #singleTokenLogits(String)} — final RMSNorm + vocabulary projection (graph N+1)
 * </ul>
 *
 * Note: Consult also the {@link
 * org.beehive.jllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout}
 */
// @formatter:on
public interface SingleTokenForwardPlanComponents {

    ActivationTaskGraph singleTokenActivation();

    TransformerLayerTaskGraphs singleTokenTransformerLayers();

    AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId);
}
