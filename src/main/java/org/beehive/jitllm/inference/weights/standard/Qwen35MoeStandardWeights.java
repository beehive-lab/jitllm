package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The host weights of a {@code qwen35moe} model: the {@code qwen35} trunk, whose dense feed-forward
 * tensors are absent, and the experts that replace it.
 */
public final class Qwen35MoeStandardWeights extends Qwen35StandardWeights {

    private final Qwen35ExpertWeights<FloatTensor> experts;

    public Qwen35MoeStandardWeights(
            Qwen35StandardWeights trunk, Qwen35ExpertWeights<FloatTensor> experts) {
        super(trunk);
        this.experts = experts;
    }

    /** The routed and shared experts, per layer. */
    public Qwen35ExpertWeights<FloatTensor> experts() {
        return experts;
    }
}
