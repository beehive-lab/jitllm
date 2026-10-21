package org.beehive.jitllm.model.deepseek2;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code deepseek2} architecture with latent attention — GLM-4.7-Flash.
 *
 * <p>Its weights are kept in the representations the file holds them in, except Q5_0, which the
 * loader rewrites as Q8_0 exactly.
 */
public class DeepSeek2 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("deepseek2");

    private final DeepSeek2Configuration configuration;

    public DeepSeek2(
            DeepSeek2Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public DeepSeek2Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DEEPSEEK_2;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public State createNewState() {
        return newState(-1);
    }

    @Override
    public State createNewState(int batchsize) {
        return newState(batchsize);
    }

    private State newState(int batchsize) {
        State state = new DeepSeek2State(configuration, batchsize);
        state.latestToken = chatFormat.getBeginOfText();
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
        if (state.executionPolicy().phaseStrategy()
                == org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
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

    @Override
    public boolean shouldAddSystemPrompt() {
        return true;
    }

    @Override
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }
}
