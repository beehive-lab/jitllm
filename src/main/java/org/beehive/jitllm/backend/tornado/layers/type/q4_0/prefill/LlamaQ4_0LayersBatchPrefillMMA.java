package org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill;

import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.ops.DequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Implementations;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Llama's Q4_0 batched prefill on the tensor cores without native libraries: each Q4_0 projection
 * decoded into FP16 before its GEMM.
 */
public class LlamaQ4_0LayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    public LlamaQ4_0LayersBatchPrefillMMA(
            State state, LlamaTornadoWeights weights, LlamaConfiguration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    /**
     * Declares the layer's weights through the session's prepacker, as the decode layers that bind
     * them expect: the Q4_0 projections the plan prepacks are consumed from their prepacking graph.
     */
    @Override
    protected void uploadLayerWeights(TaskGraph batchPrefillLayer, int layerIndex) {
        LlamaTornadoWeights w = (LlamaTornadoWeights) weights;
        state.workspace.prepacker.configureGraphInputs(
                batchPrefillLayer,
                state.workspace.prepacking,
                w.rms_att_weightLayered[layerIndex].asFloatArray(),
                w.wqLayered[layerIndex],
                w.wkLayered[layerIndex],
                w.wvLayered[layerIndex],
                w.woLayered[layerIndex],
                w.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                w.w1Layered[layerIndex],
                w.w2Layered[layerIndex],
                w.w3Layered[layerIndex],
                w.freq_cis_realFlat.asFloatArray(),
                w.freq_cis_imagFlat.asFloatArray());
    }

    /** Each projection decoded into one scratch, then one GEMM ({@link DequantizingMatMul}). */
    @Override
    protected List<Implementation> candidates() {
        return Implementations.dequantizingTensorCorePrefill();
    }
}
