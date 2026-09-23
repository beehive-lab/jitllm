package org.beehive.jllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * One architecture's <b>legacy plan components</b>, as this backend builds them.
 *
 * <p>The twin of {@code TornadoLoweringProvider}, for the path that is still the default. The
 * lowering answers "build this program"; this answers "build the plan components this family has
 * always used". Both are registered per identity and discovered, and neither is listed centrally.
 *
 * <p>The concrete {@code State} cast that {@code ForwardPlanFactory} used to do per family moves in
 * here, which is where a family's own file can do it safely. That cast is the reason the factory's
 * switch could not simply be deleted (Rule 15's note on {@code ForwardPlanFactory$1}).
 */
public interface TornadoPlanProvider {

    /** The identity these components implement. Two providers claiming one identity is an error. */
    ArchitectureId architecture();

    /**
     * The <b>model-wide</b> weight representations this provider builds plan components for.
     *
     * <p>Provider admission, and nothing else. It answers "given a model whose weights report this
     * representation, can this family build a plan for it", which is the question {@code
     * ForwardPlanFactory} asks. It is deliberately <b>not</b> the set of representations the
     * family's kernels can read: a heterogeneous model reports one representation and holds
     * several, and answering both questions with one set would make one of the two answers false.
     * See {@link #nativeTensorTypes()} for the other half.
     */
    Set<DataType> supportedDataTypes();

    /**
     * The representations this family's tasks read <b>per tensor</b>, in the file's own layout.
     *
     * <p>Read by the memory preflight, which predicts each tensor at the representation it will
     * actually occupy on the device. A family listing {@code Q4_0} has tasks that decode Q4_0
     * blocks, so a Q4_0 tensor stays 4.5 bits per weight instead of becoming 8.5.
     *
     * <p>Defaults to {@link #supportedDataTypes()}, which is right for every family whose model is
     * one representation throughout: there the two questions have one answer. A family with a
     * genuinely mixed model overrides it — {@code qwen35} admits plans for its projections'
     * representation while its tasks also decode Q4_1 down projections, Q5_K recurrent outputs, a
     * Q6_K vocabulary projection and F32 recurrent parameters.
     *
     * <p>Listing a representation here is a claim that some task decodes it natively. It is not a
     * claim that every operation can: an operation with no kernel for a representation must fail by
     * name at plan construction, not be repaired by conversion.
     */
    default Set<DataType> nativeTensorTypes() {
        return supportedDataTypes();
    }

    /** The execution modes this family has plans for. */
    Set<ExecutionMode> supportedModes();

    /**
     * Builds the components for a representation this provider supports.
     *
     * @throws IllegalArgumentException if the state is not this family's, which is the cast that
     *     used to live in the central factory
     */
    SingleTokenForwardPlanComponents components(DataType weights, State state, Model model);
}
