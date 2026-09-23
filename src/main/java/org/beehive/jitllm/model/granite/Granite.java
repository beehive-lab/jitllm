package org.beehive.jitllm.model.granite;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.tokenizer.GraniteTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

public class Granite extends AbstractModel {

    private final GraniteConfiguration configuration;

    public Granite(
            GraniteConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public GraniteConfiguration configuration() {
        return configuration;
    }

    @Override
    public GraniteTokenizer tokenizer() {
        return (GraniteTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GRANITE;
    }

    @Override
    public State createNewState() {
        State state = new GraniteState(configuration(), -1);
        // Granite uses token 0 (<|end_of_text|>) as BOS - it's multi-purpose
        // Token 0 is the default BOS for Granite
        state.latestToken = 0;
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new GraniteState(configuration(), batchsize);
        // Token 0 is the default BOS for Granite
        state.latestToken = 0;
        return state;
    }

    @Override
    public List<Integer> generateTokens(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {
        return TokenGenerationLoop.generateTokensGranite(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        return TokenGenerationLoop.generateTokensGPUGranite(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated,
                tornadoVMPlan);
    }

    // Convenience accessors for scaling factors (used in forward pass)
    public float embeddingScale() {
        return configuration.embeddingScale();
    }

    public float residualScale() {
        return configuration.residualScale();
    }

    public float attentionScale() {
        return configuration.attentionScale();
    }

    public float logitScale() {
        return configuration.logitScale();
    }

    @Override
    public State createNewState(org.beehive.jitllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new GraniteState(configuration(), -1, lease);
        state.latestToken = 0;
        return state;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jitllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jitllm.runtime.model.ArchitectureId.of("granite");
    }
}
