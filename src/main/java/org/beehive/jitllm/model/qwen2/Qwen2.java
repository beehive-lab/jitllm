package org.beehive.jitllm.model.qwen2;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

public class Qwen2 extends AbstractModel {

    Qwen2Configuration configuration;

    public Qwen2(
            Qwen2Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    public Qwen2Configuration configuration() {
        return configuration;
    }

    @Override
    public Tokenizer tokenizer() {
        return (Qwen3Tokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_2;
    }

    @Override
    public State createNewState() {
        State state = new Qwen2State(configuration(), -1);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen2State(configuration(), batchsize);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /** No <|beginoftext|> needed for Qwen models. */
    @Override
    public boolean shouldAddBeginOfText() {
        return false;
    }

    /**
     * No system prompt for Deepseek-R1-Distill-Qwen. Based on <a
     * href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">Usage
     * Recommendations</a>
     */
    @Override
    public boolean shouldAddSystemPrompt() {
        return !getModelType().isDeepSeekR1();
    }

    /**
     * Force inclusion of <think></think> for Deepseek-R1-Distill-Qwen. Based on <a
     * href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">Usage
     * Recommendations</a>
     */
    @Override
    public boolean shouldIncludeReasoning() {
        return getModelType().isDeepSeekR1();
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
                    "Batch prefill/decode on CPU not yet implemented for Qwen2/Deepseek-R1-Distill-Qwen");
        }
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "Prefill/decode on CPU not yet implemented for Qwen2/Deepseek-R1-Distill-Qwen");
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
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE
                && state.executionPolicy().prefillBatchSize() > 1) {
            throw new UnsupportedOperationException(
                    "Batch prefill/decode on GPU not yet implemented for Qwen2/Deepseek-R1-Distill-Qwen");
        }
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "Prefill/decode on GPU not yet implemented for Qwen2/Deepseek-R1-Distill-Qwen");
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

    @Override
    public State createNewState(org.beehive.jitllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new Qwen2State(configuration(), -1, lease);
        state.latestToken =
                tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jitllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jitllm.runtime.model.ArchitectureId.of("qwen2");
    }
}
