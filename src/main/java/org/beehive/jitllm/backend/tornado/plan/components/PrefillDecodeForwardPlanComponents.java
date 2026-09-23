package org.beehive.jllm.backend.tornado.plan.components;

import org.beehive.jllm.backend.tornado.TornadoVMMasterPlanPrefillDecode;
import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.plan.PrefillDecodeForwardPlan;

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
 *   <li>{@link #prefillDecodeActivation()} — KV-cache-aware activation (graph 0)
 *   <li>{@link #prefillDecodeTransformerLayers()} — N KV-cache-aware transformer layer TaskGraphs
 *       (graphs 1.N)
 *   <li>{@link #decodeLogits(String)} — final RMSNorm + vocabulary projection (graph N+1)
 * </ul>
 *
 * Note: Consult also the {@link
 * org.beehive.jllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout}
 */
// @formatter:on
public interface PrefillDecodeForwardPlanComponents extends SingleTokenForwardPlanComponents {

    ActivationTaskGraph prefillDecodeActivation();

    TransformerLayerTaskGraphs prefillDecodeTransformerLayers();

    AbstractLogitsTaskGraph decodeLogits(String previousGraphId);
}
