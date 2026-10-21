package org.beehive.jitllm.backend.tornado.layers.qwen2;

import org.beehive.jitllm.backend.tornado.kernels.Qwen2BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Qwen2's tensor-core batched prefill: {@link LlamaQ8_0LayersBatchPrefillMMA}, with the Q, K and V
 * biases added to the packed rows of the fused QKV GEMM and Qwen2's split-half RoPE.
 */
public class Qwen2Q8_0LayersBatchPrefillMMA extends LlamaQ8_0LayersBatchPrefillMMA {

    public Qwen2Q8_0LayersBatchPrefillMMA(
            Qwen2State state, Qwen2TornadoWeights weights, Configuration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    // The base builds its graphs in its constructor, before this class's fields would be set,
    // so the hooks read the inherited fields.
    private Qwen2TornadoWeights qwen2Weights() {
        return (Qwen2TornadoWeights) weights;
    }

    @Override
    protected Object[] extraLayerWeights(int layerIndex) {
        return new Object[] {
            qwen2Weights().q_biasLayered[layerIndex].asFloatArray(),
            qwen2Weights().k_biasLayered[layerIndex].asFloatArray(),
            qwen2Weights().v_biasLayered[layerIndex].asFloatArray()
        };
    }

    @Override
    protected void afterQkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_qkv_bias",
                Qwen2BatchPrefillKernels::batchedPackedQKVBias,
                context,
                state.workspace.qkvResultBatch,
                qwen2Weights().q_biasLayered[layerIndex].asFloatArray(),
                qwen2Weights().k_biasLayered[layerIndex].asFloatArray(),
                qwen2Weights().v_biasLayered[layerIndex].asFloatArray(),
                config.dim(),
                config.kvDim(),
                batchSize);
    }

    @Override
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {
        int rows = batchSize * (config.dim() + 2 * config.kvDim());
        scheduler.addWorkerGrid(
                p + "batch_qkv_bias",
                WorkerGridFactory.genericWorker(((rows + 255) / 256) * 256, 256));
    }

    @Override
    protected boolean splitHalfRope() {
        return true;
    }

    @Override
    protected float ropeTheta() {
        return config.ropeTheta();
    }
}
