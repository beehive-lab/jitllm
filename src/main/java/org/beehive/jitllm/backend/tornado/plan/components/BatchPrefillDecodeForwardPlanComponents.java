package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.BatchPrefillDecodeForwardPlan;
import org.beehive.jitllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;

// @formatter:off
/**
 * The necessary components that any model+quantization combination should implement to support
 * *batch-prefill/decode inference*.
 *
 * <p>Batch-prefill/decode inference with TornadoVM is implemented by {@link
 * TornadoVMMasterPlanBatchPrefillDecode}. It employs a {@link BatchPrefillDecodeForwardPlan}
 * instance to represent the complete batch-prefill/decode forward operation as a chain of distinct
 * TornadoVM TaskGraphs. The components of this chain are represented by the following components:
 *
 * <ul>
 *   <li>{@link #batchPrefillActivation(int)} — B×dim embedding → FP32 batch activation (the first
 *       inference graph)
 *   <li>{@link #batchPrefillTransformerLayers(int)} — N batch transformer layer TaskGraphs
 *   <li>{@link #batchDecodeActivation(String)} — single-token decode activation
 *   <li>{@link #batchDecodeTransformerLayers()} — N decode transformer layer TaskGraphs
 *   <li>{@link #decodeLogits(String)} (inherited) — final RMSNorm + vocabulary projection (the last
 *       graph)
 * </ul>
 *
 * <p>The prepacking graphs, when the plan uses weight prepacking, come before them; {@link
 * BatchPrefillDecodeForwardTaskGraphLayout} gives every position, including a fallback family's.
 */
// @formatter:on
public interface BatchPrefillDecodeForwardPlanComponents
        extends PrefillDecodeForwardPlanComponents {

    ActivationTaskGraph batchPrefillActivation(int batchSize);

    ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId);

    TransformerLayerTaskGraphs batchDecodeTransformerLayers();

    BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize);
}
