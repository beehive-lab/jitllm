package org.beehive.jitllm.backend.tornado.layers.type.q4_k.prefill;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.layers.DequantizingFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Devstral's Q4_K batched prefill on the tensor cores: each Q4_K (or Q6_K) projection decoded into
 * FP16 before its GEMM. There is no Q4_K tensor-core GEMM.
 */
public class DevstralQ4_KLayersBatchPrefillMMA extends DequantizingFP16LayersBatchPrefillMMA {

    public DevstralQ4_KLayersBatchPrefillMMA(
            State state, LlamaTornadoWeights weights, DevstralConfiguration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    /**
     * Decodes {@code w} into the scratch: Q4_K, or Q6_K where the file gave a tensor more bits (a
     * Q4_K_M file holds {@code attn_v} and {@code ffn_down} as Q6_K in half its layers).
     */
    @Override
    protected void decode(TaskGraph layer, String task, TornadoTensor w, int destOffset, int n) {
        switch (w.dataType()) {
            case Q4_K ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            case Q6_K ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            default ->
                    throw new UnsupportedOperationException(
                            "the devstral batched prefill has no decoder for "
                                    + w.dataType()
                                    + " ("
                                    + task
                                    + "); the projections must be Q4_K or Q6_K");
        }
    }
}
