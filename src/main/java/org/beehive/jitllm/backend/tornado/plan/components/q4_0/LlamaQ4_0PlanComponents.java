package org.beehive.jitllm.backend.tornado.plan.components.q4_0;

import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.llama.LlamaQ4_0LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.LlamaQ4_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.decode.LlamaQ4_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill.LlamaQ4_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill.LlamaQ4_0LayersBatchPrefillNative;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.StagedForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * Llama's plan when its per-layer weights are retained as Q4_0.
 *
 * <p>Mixed by construction, because the file is. {@code llama-quantize ... Q4_0} writes every
 * {@code blk.*} weight as Q4_0 (and some {@code ffn_down} as Q4_1), with {@code token_embd} as Q4_0
 * or Q6_K, which is kept as it is. So the transformer layers read Q4_0, and the embedding and the
 * logits layer read their tensors in their own representation.
 *
 * <p>Single-token decode, and batched prefill with its projections run by cuBLAS ({@code
 * --with-native-libraries}). There is no sequential prefill/decode plan: asking for one fails here,
 * by name.
 */
// @formatter:on
public class LlamaQ4_0PlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final SchedulerType schedulerType;

    public LlamaQ4_0PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (LlamaConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new LlamaQ4_0FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Batched prefill + decode ──────────────────────────────────────────────

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        // Metal: the host decodes the embeddings into the FP32 batch carrier for the SIMD-group
        // layers. CUDA: a Q8_0 or Q4_0 embedding is decoded on the device for the cuBLAS and
        // tensor-core layers; any other kept embedding (Q6_K) the host decodes, as on Metal.
        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        if (!BatchPrefillGemmPolicy.simdgroup()
                && (embedding == DataType.Q8_0 || embedding == DataType.Q4_0)) {
            return new BatchPrefillQ8DeviceActivation(state, config, batchSize, embedding);
        }
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        // The embedding tensor's own representation, which is what the passes stage.
        return new BatchDecodeActivation(
                state, config, lastBatchLayerId, weights.getTokenEmbeddingTable().dataType());
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        if (BatchPrefillGemmPolicy.simdgroup()) {
            return new org.beehive.jitllm.backend.tornado.layers.llama.LlamaQ4_0FFNLayersDecode(
                    state, weights, config, schedulerType);
        }
        return new LlamaQ4_0FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return BatchPrefillGemmPolicy.simdgroup()
                ? new LlamaQ4_0LayersBatchPrefill(state, weights, config, batchSize)
                : batchPrefillTransformerLayers(batchSize, 0, config.numberOfLayers());
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsQ8_0LayerDecode(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Pipeline stages ───────────────────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer) {
        return new LlamaQ4_0FFNLayers(state, weights, config, schedulerType, firstLayer, endLayer);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        // The whole model: the projections decoded to FP16, then the tensor-core GEMM (cuBLAS
        // under native libraries). A pipeline stage keeps the int8/cuBLAS layers, which build any
        // range of layers.
        if (firstLayer == 0 && endLayer == config.numberOfLayers()) {
            return new LlamaQ4_0LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        if (!NativePrefillSupport.nativeProjections(state.executionPolicy())) {
            throw new UnsupportedOperationException(
                    "the Llama Q4_0 batched prefill of a pipeline stage runs its projections"
                            + " through cuBLAS: add --with-native-libraries");
        }
        return new LlamaQ4_0LayersBatchPrefillNative(
                state, weights, config, batchSize, firstLayer, endLayer);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        return new LlamaQ4_0FFNLayersDecode(
                state, weights, config, schedulerType, firstLayer, endLayer);
    }

    // ── Sequential prefill/decode: not implemented for Q4_0 ──────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw noSequentialPrefill();
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw noSequentialPrefill();
    }

    private static UnsupportedOperationException noSequentialPrefill() {
        return new UnsupportedOperationException(
                "Llama Q4_0 has no sequential prefill/decode plan; use --batch-prefill-size N"
                        + " with --with-native-libraries, or the default single-token plan");
    }
}
