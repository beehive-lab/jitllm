package org.beehive.jitllm.model.kimilinear;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code kimi-linear} architecture (Kimi-Linear-48B-A3B): KDA and latent
 * attention layers, about three to one, with a mixture of experts.
 */
public class KimiLinear extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("kimi-linear");

    private final KimiLinearConfiguration configuration;

    public KimiLinear(
            KimiLinearConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public KimiLinearConfiguration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.KIMI_LINEAR;
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
        State state = new KimiLinearState(configuration, batchsize);
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
                == ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
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
