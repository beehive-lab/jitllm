package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * One expert's gated feed-forward pass, selected by index out of stacked expert tensors.
 *
 * <h2>Why the weights are whole tensors</h2>
 *
 * <p>It is therefore a small composite — it contains a gated feed-forward — and that is the point:
 * what it hides is the index arithmetic, which is exactly the thing that must not become slicing.
 *
 * @param input the activations every expert reads, the feed-forward norm output
 * @param expertIndex which selected slot to execute, an index into the router's output
 * @param selectedIds the router's selected expert identifiers
 * @param gateWeights stacked gate projections for all experts
 * @param upWeights stacked up projections for all experts
 * @param downWeights stacked down projections for all experts
 * @param hidden working buffer for the gate half
 * @param hiddenUp working buffer for the up half
 * @param output this expert's contribution, before it is weighted and accumulated
 * @param expertHiddenDim one expert's hidden width
 * @param modelDim the model dimension
 * @param dataType the representation the expert weights were materialized in
 */
public record ExpertFeedForward(
        OperandRef input,
        int expertIndex,
        OperandRef selectedIds,
        OperandRef.Weight gateWeights,
        OperandRef.Weight upWeights,
        OperandRef.Weight downWeights,
        OperandRef hidden,
        OperandRef hiddenUp,
        OperandRef output,
        int expertHiddenDim,
        int modelDim,
        DataType dataType)
        implements Operation {

    public ExpertFeedForward {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(selectedIds, "selectedIds");
        Objects.requireNonNull(gateWeights, "gateWeights");
        Objects.requireNonNull(upWeights, "upWeights");
        Objects.requireNonNull(downWeights, "downWeights");
        Objects.requireNonNull(hidden, "hidden");
        Objects.requireNonNull(hiddenUp, "hiddenUp");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (expertIndex < 0) {
            throw new IllegalArgumentException("expertIndex must not be negative: " + expertIndex);
        }
        if (expertHiddenDim <= 0 || modelDim <= 0) {
            throw new IllegalArgumentException(
                    "expertHiddenDim and modelDim must be positive: "
                            + expertHiddenDim
                            + ", "
                            + modelDim);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.EXPERT_FEED_FORWARD;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input, selectedIds, gateWeights, upWeights, downWeights);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(hidden, hiddenUp, output);
    }
}
