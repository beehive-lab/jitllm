package org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.layers.DequantizingFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Llama's Q4_0 batched prefill on the tensor cores without native libraries: each Q4_0 projection
 * decoded into FP16 before its GEMM.
 */
public class LlamaQ4_0LayersBatchPrefillMMA extends DequantizingFP16LayersBatchPrefillMMA {

    public LlamaQ4_0LayersBatchPrefillMMA(
            State state, LlamaTornadoWeights weights, LlamaConfiguration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    /**
     * Uploads the weights through the session's repack, as the decode layers that bind them expect:
     * on a device that runs the packed int8 kernels the Q4_0 projections are repacked into tiles on
     * upload.
     */
    @Override
    protected void uploadLayerWeights(TaskGraph batchPrefillLayer, int layerIndex) {
        LlamaTornadoWeights w = (LlamaTornadoWeights) weights;
        state.workspace.packedRepack.upload(
                batchPrefillLayer,
                state.workspace.tileLayout,
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

    /**
     * Decodes {@code w}: Q4_0, from the file's blocks or the packed tiles, or Q4_1 where the file
     * kept a tensor in it (some {@code ffn_down}).
     */
    @Override
    protected void decode(TaskGraph layer, String task, TornadoTensor w, int destOffset, int n) {
        if (state.workspace.tileLayout.tiled(w)) {
            layer.task(
                    task,
                    TransformerComputeKernelsQ4_0::dequantizeQ4_0PackedToFP16,
                    context,
                    w.asByteArray(),
                    scratch(),
                    destOffset,
                    n,
                    Math.toIntExact(w.shape().dimension(0)));
            return;
        }
        switch (w.dataType()) {
            case Q4_0 ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ4_0::dequantizeQ4_0ToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            case Q4_1 ->
                    layer.task(
                            task,
                            TransformerComputeKernelsQ4_0::dequantizeQ4_1ToFP16,
                            context,
                            w.asByteArray(),
                            scratch(),
                            destOffset,
                            n);
            default ->
                    throw new UnsupportedOperationException(
                            "the llama Q4_0 batched prefill has no decoder for "
                                    + w.dataType()
                                    + " ("
                                    + task
                                    + "); the projections must be Q4_0 or Q4_1");
        }
    }
}
