package org.beehive.jitllm.model.qwen35moe;

import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.qwen35.Qwen35;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.tokenizer.Tokenizer;

// @formatter:off
/**
 * A {@code qwen35moe} model (Qwen3.6-35B-A3B): the {@code qwen35} hybrid stack of gated delta-net
 * and attention layers, whose feed-forward is a mixture of experts instead of a dense SwiGLU.
 *
 * <p>Its own family: its own loader, state, weights, forward passes and plan components, each built
 * on the {@code qwen35} one and adding the experts. The trunk configuration is {@code qwen35}'s,
 * with the routed experts' combined width for one token as its hidden width; {@link #experts()}
 * describes the experts themselves.
 */
// @formatter:on
public final class Qwen35Moe extends Qwen35 {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen35moe");

    private final Qwen35Experts experts;

    public Qwen35Moe(
            Qwen35Configuration configuration,
            Qwen35Experts experts,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(configuration, tokenizer, weights, chatFormat);
        this.experts = experts;
    }

    /** The routed and shared experts every layer's feed-forward is made of. */
    public Qwen35Experts experts() {
        return experts;
    }

    @Override
    protected Qwen35State constructState(int batchsize) {
        return new Qwen35MoeState(configuration(), experts, batchsize);
    }

    @Override
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }
}
