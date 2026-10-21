package org.beehive.jitllm.model.qwen35moe;

/**
 * The mixture-of-experts feed-forward of a {@code qwen35moe} model: in place of the dense SwiGLU,
 * each token is routed to {@link #used()} of {@link #count()} experts, each a SwiGLU of width
 * {@link #hiddenDim()}, and a shared expert of width {@link #sharedHiddenDim()} runs for every
 * token.
 *
 * <p>Routing follows the model: a softmax over all experts, the top {@code used} kept and their
 * weights renormalized to sum to one; the shared expert's output is scaled by the sigmoid of its
 * own gate. That is what llama.cpp runs for this architecture ({@code build_moe_ffn} with a softmax
 * gate and {@code norm_w}).
 */
public record Qwen35Experts(int count, int used, int hiddenDim, int sharedHiddenDim) {

    public Qwen35Experts {
        if (count < 1 || used < 1 || used > count || hiddenDim < 1 || sharedHiddenDim < 0) {
            throw new IllegalArgumentException(
                    "experts: count "
                            + count
                            + ", used "
                            + used
                            + ", width "
                            + hiddenDim
                            + ", shared width "
                            + sharedHiddenDim);
        }
    }

    /** Width of the routed experts' hidden activations for one token: every slot's, end to end. */
    public int routedHiddenDim() {
        return used * hiddenDim;
    }
}
