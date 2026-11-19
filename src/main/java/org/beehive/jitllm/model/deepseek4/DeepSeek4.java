package org.beehive.jitllm.model.deepseek4;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek4TornadoWeights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code deepseek4} architecture (DeepSeek-V4): attention over a sliding
 * window and compressed blocks, a mixture of experts and hyper-connections.
 */
public class DeepSeek4 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("deepseek4");

    private final DeepSeek4Configuration configuration;

    public DeepSeek4(
            DeepSeek4Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public DeepSeek4Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DEEPSEEK_4;
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
        State state = new DeepSeek4State(configuration, batchsize);
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

    /** The hash-routed layers' experts come from tables on the host: staged per token. */
    @Override
    public void stagePerTokenDeviceInputs(State state, int token) {
        if (weights instanceof DeepSeek4TornadoWeights device) {
            ((DeepSeek4State) state)
                    .stageHashExperts(
                            token,
                            device.layers.ffn().tokenExperts(),
                            configuration.expertsUsed(),
                            Math.min(
                                    configuration.hashLayerCount(),
                                    configuration.numberOfLayers()));
        }
    }

    /** The chunk-wide twin of {@link #stagePerTokenDeviceInputs}. */
    @Override
    public void stageBatchDeviceInputs(State state, int[] tokens, int chunkSize) {
        if (weights instanceof DeepSeek4TornadoWeights device) {
            ((DeepSeek4State) state)
                    .stageBatchHashExperts(
                            tokens,
                            chunkSize,
                            device.layers.ffn().tokenExperts(),
                            configuration.expertsUsed(),
                            Math.min(
                                    configuration.hashLayerCount(),
                                    configuration.numberOfLayers()));
        }
    }
}
