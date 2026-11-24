package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.inference.state.State;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * Plan components that can build one stage of a model split by layers across several devices: the
 * layers {@code [firstLayer, endLayer)} on that stage's state.
 *
 * <p>The pipeline plan builds every stage from these, and from the family's own activation and
 * logits components for the first and last stage; it names no family. A family whose components do
 * not implement this is not split, and says so.
 *
 * <p>The first layer of a range takes the hidden state from the graph before it in the stage's
 * plan, which on a later stage is the hand-off that received it. Whether a stage's key/value cache
 * holds its own layers only or the whole model's is the family's decision, made where its stage
 * state is built ({@code TornadoPlanProvider.stageState}) and honoured by its layers.
 */
// @formatter:on
public interface StagedForwardPlanComponents extends SingleTokenForwardPlanComponents {

    /** The single-token layers {@code [firstLayer, endLayer)}. */
    TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer);

    /**
     * The batched prefill layers {@code [firstLayer, endLayer)}, for a family that is also {@link
     * BatchPrefillDecodeForwardPlanComponents}.
     */
    default BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        throw new UnsupportedOperationException(
                "this family's batched prefill is not split across devices; drop"
                        + " --batch-prefill-size to split its single-token plan");
    }

    /**
     * What crosses a stage boundary in the single-token and decode plans, on {@code stageState}:
     * the hidden state, unless a family carries more between its layers (glm5next's residual
     * streams).
     */
    default FloatArray stageCarrier(State stageState) {
        return stageState.workspace.wrapX;
    }

    /** What crosses a stage boundary in the batched prefill: a chunk's hidden states by default. */
    default FloatArray batchStageCarrier(State stageState) {
        return stageState.workspace.wrapXBatch;
    }

    /** The decode layers {@code [firstLayer, endLayer)} that follow a staged batched prefill. */
    default TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        throw new UnsupportedOperationException(
                "this family's batched prefill is not split across devices; drop"
                        + " --batch-prefill-size to split its single-token plan");
    }
}
