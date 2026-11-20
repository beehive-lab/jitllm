package org.beehive.jitllm.model.lfm2;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.tokenizer.Lfm2Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code lfm2moe} architecture — Liquid AI's hybrid convolution/attention
 * mixture of experts, LFM2-8B-A1B and LFM2.5-24B-A2B.
 *
 * <p>Runs on the host and, in single-token and sequential prefill/decode, on an accelerator. Its
 * weights are kept in the representations the file holds them in — Q4_K, Q5_K and Q6_K experts,
 * Q8_0 attention and dense projections, F32 norms, routers and convolution taps — and each
 * matrix–vector product decodes the representation of the tensor it reads.
 */
public class Lfm2 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("lfm2moe");

    private final Lfm2Configuration configuration;

    public Lfm2(
            Lfm2Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public Lfm2Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LFM2_MOE;
    }

    @Override
    public Lfm2Tokenizer tokenizer() {
        return (Lfm2Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        return newState(-1);
    }

    @Override
    public State createNewState(int batchsize) {
        return newState(batchsize);
    }

    /**
     * A session's state, told whether it is being built for a device — which the weights this model
     * holds decide, not a system property.
     */
    private State newState(int batchsize) {
        boolean device = weights instanceof TornadoWeights;
        State state =
                Lfm2State.withDeviceArrays(device, () -> new Lfm2State(configuration, batchsize));
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

    /**
     * Decode on the accelerator, through the shared generation loops: they stage the token's
     * embedding, run the plan's graphs in order and sample. Everything that makes this model what
     * it is lives in the graphs the plan holds. Sequential prefill ingests the prompt through the
     * same layer graphs with the logits graph skipped.
     */
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
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }
}
