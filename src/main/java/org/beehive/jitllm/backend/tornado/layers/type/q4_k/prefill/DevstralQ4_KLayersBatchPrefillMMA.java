package org.beehive.jitllm.backend.tornado.layers.type.q4_k.prefill;

import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.ops.DequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Implementations;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;

/**
 * Devstral's Q4_K batched prefill on the tensor cores: each Q4_K (or Q6_K) projection decoded into
 * FP16 before its GEMM. There is no Q4_K tensor-core GEMM.
 */
public class DevstralQ4_KLayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    public DevstralQ4_KLayersBatchPrefillMMA(
            State state, LlamaTornadoWeights weights, DevstralConfiguration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    /** Each projection decoded into one scratch, then one GEMM ({@link DequantizingMatMul}). */
    @Override
    protected List<Implementation> candidates() {
        return Implementations.dequantizingTensorCorePrefill();
    }
}
