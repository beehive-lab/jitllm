package org.beehive.jitllm.inference.weights;

/**
 * The mixture-of-experts tensors of a {@code qwen35moe} model, per layer, in the tensor type of the
 * backend that holds them ({@code FloatTensor} on the host, {@code TornadoTensor} on the device).
 *
 * <p>The routed experts stay stacked as the file stores them: {@link #gateExperts()}[l] holds every
 * expert's gate matrix, expert {@code e} at element offset {@code e * hiddenDim * dim}, and
 * likewise the up and down projections ({@code e * dim * hiddenDim}). Nothing is split or
 * converted.
 *
 * @param router {@code ffn_gate_inp}: one score row per expert, F32
 * @param gateExperts {@code ffn_gate_exps}
 * @param upExperts {@code ffn_up_exps}
 * @param downExperts {@code ffn_down_exps}
 * @param sharedGate {@code ffn_gate_shexp}
 * @param sharedUp {@code ffn_up_shexp}
 * @param sharedDown {@code ffn_down_shexp}
 * @param sharedGateInput {@code ffn_gate_inp_shexp}: the shared expert's gate vector, F32
 */
public record Qwen35ExpertWeights<T>(
        T[] router,
        T[] gateExperts,
        T[] upExperts,
        T[] downExperts,
        T[] sharedGate,
        T[] sharedUp,
        T[] sharedDown,
        T[] sharedGateInput) {}
