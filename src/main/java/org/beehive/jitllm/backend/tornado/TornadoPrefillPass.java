package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;

/**
 * The accelerated <b>sequential prefill</b> pass: stage one token's embedding, then run the plan's
 * prefill graphs.
 *
 * <p>The body is the one that ran before, moved rather than rewritten: the same two data-type
 * cases, the same byte arithmetic, the same delegation to {@link
 * TornadoVMMasterPlanPrefillDecode#tornadoVMForwardPrefill}. Sequential prefill stays selectable
 * and behaves identically.
 */
public final class TornadoPrefillPass {

    private TornadoPrefillPass() {}

    /**
     * Stages {@code token}'s embedding into the session's device carrier, then executes the
     * preprocessing and layer graphs — the logits graph is deliberately skipped, because a prefill
     * position's logits are discarded and only its KV entry matters.
     *
     * @param model the model (must carry {@link TornadoWeights})
     * @param state the session's state, whose workspace holds the embedding carrier
     * @param token input token id
     * @param position sequence position being processed
     * @param prefillPlan the prefill/decode plan wrapper
     */
    public static void prefill(
            Model model,
            State state,
            int token,
            int position,
            TornadoVMMasterPlanPrefillDecode prefillPlan) {
        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        TornadoForwardPass.stageEmbeddingRow(weights, configuration, state, token);

        prefillPlan.tornadoVMForwardPrefill(position);
    }
}
