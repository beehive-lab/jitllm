package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.Qwen35MoeStandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import org.beehive.jitllm.model.qwen35moe.Qwen35Moe;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host forward pass of {@code qwen35moe}: the {@code qwen35} pass ({@link Qwen35Forward}) with
 * the mixture-of-experts feed-forward in every layer.
 */
public final class Qwen35MoeForward {

    private Qwen35MoeForward() {}

    public static FloatTensor forward(Model model, State state, int token, int position) {
        Qwen35Experts experts = ((Qwen35Moe) model).experts();
        return Qwen35Forward.forward(
                model,
                state,
                token,
                position,
                (config, weights, s, l) ->
                        mixtureOfExperts(
                                experts,
                                config.dim(),
                                (Qwen35MoeStandardWeights) weights,
                                (Qwen35MoeState) s,
                                l));
    }

    // @formatter:off
    /**
     * The mixture-of-experts feed-forward of layer {@code l}, from {@code xb} into {@code xb2}.
     *
     * <p>A softmax over every expert's score, the top {@code used} kept and their weights
     * renormalized to sum to one; each selected expert's SwiGLU output added in weighted, in
     * descending weight order; then the shared expert's, scaled by the sigmoid of its gate. The
     * same arithmetic as llama.cpp's {@code build_moe_ffn} with a softmax gate and {@code norm_w},
     * plus its shared-expert branch.
     */
    // @formatter:on
    private static void mixtureOfExperts(
            Qwen35Experts experts,
            int dim,
            Qwen35MoeStandardWeights weights,
            Qwen35MoeState state,
            int l) {
        var tensors = weights.experts();

        CpuOperations.moeRouter(
                state.xb,
                tensors.router()[l],
                state.expertScores,
                state.expertIds,
                state.expertWeights,
                experts.count(),
                experts.used(),
                dim);
        float total = 0f;
        for (float weight : state.expertWeights) {
            total += weight;
        }
        for (int k = 0; k < experts.used(); k++) {
            state.expertWeights[k] /= total;
        }

        state.xb2.fillInPlace(0, dim, 0f);
        for (int k = 0; k < experts.used(); k++) {
            CpuOperations.expertFeedForward(
                    state.xb,
                    state.expertIds[k],
                    tensors.gateExperts()[l],
                    tensors.upExperts()[l],
                    tensors.downExperts()[l],
                    state.expertHidden,
                    state.expertHiddenUp,
                    state.expertOut,
                    experts.hiddenDim(),
                    dim);
            CpuOperations.weightedAccumulate(
                    state.xb2, state.expertOut, state.expertWeights[k], dim);
        }

        if (experts.sharedHiddenDim() > 0) {
            int shared = experts.sharedHiddenDim();
            CpuOperations.matVec(
                    tensors.sharedGate()[l], state.xb, state.expertHidden, shared, dim);
            CpuOperations.matVec(
                    tensors.sharedUp()[l], state.xb, state.expertHiddenUp, shared, dim);
            CpuOperations.swiGLU(state.expertHidden, state.expertHiddenUp);
            CpuOperations.matVec(
                    tensors.sharedDown()[l], state.expertHidden, state.expertOut, dim, shared);
            float gate = tensors.sharedGateInput()[l].dot(0, state.xb, 0, dim);
            CpuOperations.weightedAccumulate(
                    state.xb2, state.expertOut, CpuOperations.logistic(gate), dim);
        }
    }
}
