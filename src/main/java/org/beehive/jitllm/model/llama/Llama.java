package org.beehive.jllm.model.llama;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.TokenGenerationLoop;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.AbstractModel;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.tokenizer.LlamaTokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

public class Llama extends AbstractModel {

    LlamaConfiguration configuration;

    public Llama(
            LlamaConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public LlamaConfiguration configuration() {
        return configuration;
    }

    @Override
    public LlamaTokenizer tokenizer() {
        return (LlamaTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LLAMA_3;
    }

    @Override
    public State createNewState() {
        State state = new LlamaState(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new LlamaState(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(org.beehive.jllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new LlamaState(configuration(), -1, lease);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
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
        return TokenGenerationLoop.generateTokensLlamaForPolicy(
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

    /** Its layer graphs chain the KV buffers by predecessor name, so a shared table stays fresh. */
    @Override
    public boolean supportsSharedKvStorage() {
        return true;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jllm.runtime.model.ArchitectureId.of("llama");
    }
}
