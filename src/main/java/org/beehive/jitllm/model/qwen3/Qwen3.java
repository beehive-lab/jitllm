package org.beehive.jllm.model.qwen3;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.TokenGenerationLoop;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.AbstractModel;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

public class Qwen3 extends AbstractModel {

    Qwen3Configuration configuration;

    public Qwen3(
            Qwen3Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    public Qwen3Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_3;
    }

    public Qwen3Tokenizer tokenizer() {
        return (Qwen3Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Qwen3State(configuration(), -1);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen3State(configuration(), batchsize);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /** No begin of text needed for Qwen models. */
    @Override
    public boolean shouldAddBeginOfText() {
        return false;
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
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE
                && state.executionPolicy().prefillBatchSize() > 1) {
            throw new UnsupportedOperationException(
                    "Batch prefill/decode on CPU not yet implemented for Qwen3");
        }
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "Prefill/decode on CPU not yet implemented for Qwen3");
        }
        return TokenGenerationLoop.generateTokensQwen3(
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
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            // Charging the whole prompt against the budget, which is what this family's decode
            // loop does. Without it a prompt ingested as its own phase produced one token more
            // than the same prompt in STANDARD — 64 rows against 63, measured on Qwen3-0.6B.
            return TokenGenerationLoop.generateTokensGPUPrefillDecode(
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
        return TokenGenerationLoop.generateTokensGPUQwen3(
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

    /** Qwen3's cache is sized by its value-head dimension, not by {@code dim}. */
    @Override
    public int kvCacheDim() {
        Qwen3Configuration config = configuration();
        return config.numberOfHeadsValue() * config.numberOfKeyValueHeads();
    }

    @Override
    public State createNewState(org.beehive.jllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new Qwen3State(configuration(), -1, lease);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /** Its layer graphs chain the KV buffers by predecessor name, so a shared table stays fresh. */
    @Override
    public boolean supportsSharedKvStorage() {
        return true;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jllm.runtime.model.ArchitectureId.of("qwen3");
    }
}
