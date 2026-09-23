package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.BatchPrefillDecodeForwardPlan;

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
 *   <li>{@link #batchPrefillActivation(int)} — B×dim embedding → FP32 batch activation (graph 0)
 *   <li>{@link #batchPrefillTransformerLayers(int)} — N batch transformer layer TaskGraphs (graphs
 *       1.N)
 *   <li>{@link #batchDecodeActivation(String)} — single-token decode activation (graph N+1)
 *   <li>{@link #batchDecodeTransformerLayers()} — N decode transformer layer TaskGraphs (graphs
 *       N+2.2N+1)
 *   <li>{@link #decodeLogits(String)} (inherited) — final RMSNorm + vocabulary projection (graph
 *       2N+2)
 * </ul>
 *
 * Note: Consult also the {@link
 * org.beehive.jitllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout}
 */
// @formatter:on
public interface BatchPrefillDecodeForwardPlanComponents
        extends PrefillDecodeForwardPlanComponents {

    ActivationTaskGraph batchPrefillActivation(int batchSize);

    ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId);

    TransformerLayerTaskGraphs batchDecodeTransformerLayers();

    BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize);
}
