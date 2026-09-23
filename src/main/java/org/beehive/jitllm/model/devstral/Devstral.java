package org.beehive.jllm.model.devstral;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.TokenGenerationLoop;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.DevstralState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.AbstractModel;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.tokenizer.DevstralTokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

public class Devstral extends AbstractModel {

    DevstralConfiguration configuration;

    public Devstral(
            DevstralConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public DevstralConfiguration configuration() {
        return configuration;
    }

    @Override
    public DevstralTokenizer tokenizer() {
        return (DevstralTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DEVSTRAL_2;
    }

    /** Block-table KV addressing, on the same flag every other family follows. */
    public State createNewState() {
        State state = new DevstralState(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    public State createNewState(int batchsize) {
        State state = new DevstralState(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
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
        return TokenGenerationLoop.generateTokensLlama(
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
        return TokenGenerationLoop.generateTokensGpu(
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

    /** Its layer classes bind the paged kernels and chain their buffers by predecessor name. */
    @Override
    public boolean supportsSharedKvStorage() {
        return true;
    }

    /** A state whose KV lives in the lease's shared storage. */
    @Override
    public State createNewState(org.beehive.jllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new DevstralState(configuration(), -1, lease);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jllm.runtime.model.ArchitectureId.of("devstral");
    }
}
