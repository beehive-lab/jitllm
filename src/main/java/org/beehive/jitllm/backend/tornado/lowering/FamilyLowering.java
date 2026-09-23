package org.beehive.jllm.backend.tornado.lowering;

import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.runtime.metrics.MetricsSink;

/**
 * One family's lowering: it validates the programs it claims and builds their compiled form.
 *
 * <p>Extracted when the second family arrived, which is what {@code LlamaLowering}'s own comment
 * said to wait for — an abstraction drawn from one example would have been drawn from a guess. What
 * it shares is deliberately thin: what a lowering <i>is</i>, not how any family is shaped.
 *
 * <p><b>Each implementation states its own layer sequence, in order, as explicit checks.</b> There
 * is no rule table, no pattern language and no fusion-rule engine. Two families that look similar
 * are still written out twice, because the similarity is not a fact about the architecture — it is
 * a fact about these two families today.
 */
interface FamilyLowering {

    /** The architecture identity this lowering's programs carry. */
    org.beehive.jllm.runtime.model.ArchitectureId architecture();

    /** Whether this lowering can handle {@code program}, without throwing. */
    default boolean supports(InferenceProgram program) {
        try {
            validate(program);
            return true;
        } catch (UnsupportedProgramException e) {
            return false;
        }
    }

    /**
     * @throws UnsupportedProgramException naming the first thing that did not match
     */
    void validate(InferenceProgram program);

    /** Validates, then builds the compiled program. */
    TornadoVMMasterPlan lower(InferenceProgram program, State state, Model model, MetricsSink sink);
}
