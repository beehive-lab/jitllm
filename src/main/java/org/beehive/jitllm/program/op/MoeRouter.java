package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Choosing which experts a token is routed to, and with what weight.
 *
 * <p>One operation rather than four steps. A router projection, a normalization over the scores and
 * a top-k selection are one <b>semantic result</b> — "these experts, these weights" — and naming
 * them separately would pin the sequence as program structure and forbid a backend from doing them
 * in one kernel.
 *
 * <h2>What it produces</h2>
 *
 * <p>Two parallel sequences of length {@code topK}: the selected expert identifiers and their
 * routing weights. The full score vector over every expert is the operation's <b>working
 * buffer</b>, declared because every device array is, but not a program-visible result: nothing
 * downstream reads it, and publishing scratch would make it part of the contract.
 *
 * <h2>Semantics that are not free to vary</h2>
 *
 * <p>Each of these is visible in the floating-point result, because the selection order is the
 * accumulation order ({@link WeightedAccumulate}):
 *
 * <ul>
 *   <li>normalization runs over <b>all</b> experts and <b>precedes</b> selection, and the selected
 *       weights are those probabilities <b>unrescaled</b>;
 *   <li>ties select the <b>lowest expert index</b>;
 *   <li>the selected experts come out in <b>descending weight order</b>.
 * </ul>
 *
 * <p>A backend may fuse projection, normalization and selection into one kernel. It may not change
 * any of the three.
 *
 * <h2>Not the sampler's selection</h2>
 *
 * <p>Deliberately not {@link ArgMax} or {@link Sample}. Those take logits and produce one token
 * identifier; this takes activations, produces {@code k} identifiers <b>and</b> {@code k} weights,
 * and normalizes first. Sharing a type would mean sharing semantics that differ in every one of
 * those respects.
 *
 * @param input the activations the router scores, normally a feed-forward norm output
 * @param routerWeight the router's projection weight
 * @param scores working buffer for the per-expert scores, {@code numberOfExperts} long
 * @param selectedIds where the chosen expert identifiers are written, {@code topK} long
 * @param selectedWeights where their routing weights are written, {@code topK} long
 * @param numberOfExperts how many experts the router scores
 * @param topK how many are selected
 * @param normalization how the scores are turned into weights
 * @param dataType the representation the routing executes at
 */
public record MoeRouter(
        OperandRef input,
        OperandRef.Weight routerWeight,
        OperandRef scores,
        OperandRef selectedIds,
        OperandRef selectedWeights,
        int numberOfExperts,
        int topK,
        RouterNormalization normalization,
        DataType dataType)
        implements Operation {

    /**
     * How router scores become routing weights.
     *
     * <p>One value today, following the rule {@code DataType} follows: a value is added when a
     * model genuinely needs it, not in anticipation. The parameter exists so that the convention is
     * <b>stated</b> rather than inherited — a family that renormalized its top-k would otherwise
     * silently adopt Qwen's.
     */
    public enum RouterNormalization {
        /**
         * Softmax over every expert, then select; the selected weights are those probabilities and
         * are <b>not</b> rescaled to sum to one. Qwen1.5-MoE's {@code norm_topk_prob=false}.
         */
        SOFTMAX_OVER_ALL_EXPERTS
    }

    public MoeRouter {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(routerWeight, "routerWeight");
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(selectedIds, "selectedIds");
        Objects.requireNonNull(selectedWeights, "selectedWeights");
        Objects.requireNonNull(normalization, "normalization");
        Objects.requireNonNull(dataType, "dataType");
        if (numberOfExperts <= 0) {
            throw new IllegalArgumentException(
                    "numberOfExperts must be positive: " + numberOfExperts);
        }
        if (topK <= 0 || topK > numberOfExperts) {
            throw new IllegalArgumentException(
                    "topK must be in 1.." + numberOfExperts + ": " + topK);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.MOE_ROUTER;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input, routerWeight);
    }

    /** The scores buffer is written too: it is scratch, but it is scratch this operation owns. */
    @Override
    public List<OperandRef> outputs() {
        return List.of(scores, selectedIds, selectedWeights);
    }
}
