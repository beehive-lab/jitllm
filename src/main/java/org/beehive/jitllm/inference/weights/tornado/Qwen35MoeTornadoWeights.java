package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;

/**
 * The device weights of a {@code qwen35moe} model: the {@code qwen35} trunk, whose dense
 * feed-forward tensors are absent, and the experts that replace it.
 */
public final class Qwen35MoeTornadoWeights extends Qwen35TornadoWeights {

    private final Qwen35ExpertWeights<TornadoTensor> experts;

    public Qwen35MoeTornadoWeights(
            Qwen35TornadoWeights trunk, Qwen35ExpertWeights<TornadoTensor> experts) {
        super(trunk);
        this.experts = experts;
    }

    /** The routed and shared experts, per layer. */
    public Qwen35ExpertWeights<TornadoTensor> experts() {
        return experts;
    }
}
