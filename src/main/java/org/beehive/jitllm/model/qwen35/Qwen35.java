package org.beehive.jllm.model.qwen35;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.TokenGenerationLoop;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jllm.model.AbstractModel;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.tokenizer.Qwen35Tokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code qwen35} architecture — the hybrid attention/delta-net stack behind
 * the Qwen3.5, 3.6 and 3.8 releases.
 *
 * <p>Runs on the host and, in single-token decode, on an accelerator. Its weights are retained in
 * the representations the file holds them in — Q4_0 projections, Q4_1 down projections on the early
 * blocks, Q5_K recurrent outputs, a Q6_K vocabulary projection, F32 norms and SSM parameters — and
 * each device task decodes the representation of the tensor it reads. Prefill, batched decode and
 * device-side drafting are not implemented, and a request for one fails by name.
 */
public class Qwen35 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen35");

    /**
     * Whether to drive generation through the MTP draft head.
     *
     * <p>Default off, and it stays off until a backend can verify several positions in one forward
     * pass: without that, an accepted draft saves no work and the draft head's own block is added
     * cost. Read once, here, so a session cannot change it halfway through a sequence.
     */
    private static final boolean SPECULATIVE = Boolean.getBoolean("jllm.qwen35.speculative");

    private final Qwen35Configuration configuration;

    public Qwen35(
            Qwen35Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public Qwen35Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_3_5;
    }

    @Override
    public Qwen35Tokenizer tokenizer() {
        return (Qwen35Tokenizer) tokenizer;
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
     * A session's state, told whether it is being built for a device.
     *
     * <p>This family's device arrays are over a gigabyte — the recurrent state and the key/value
     * store — so a host session does not allocate them. What decides is the weights this model
     * actually holds, which is the only thing here that knows: reading a system property instead
     * gave a device session with null buffers whenever a caller loaded device weights without
     * setting it.
     */
    private State newState(int batchsize) {
        boolean device = weights instanceof TornadoWeights;
        State state =
                Qwen35State.withDeviceArrays(
                        device, () -> new Qwen35State(configuration(), batchsize));
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
        if (SPECULATIVE && configuration.numberOfNextnLayers() > 0) {
            return TokenGenerationLoop.generateTokensQwen35(
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

    // @formatter:off
    /**
     * Single-token decode on the accelerator, through the shared generation loop.
     *
     * <p>The loop is the one Qwen3 uses. Nothing in it is family-specific: it stages the token's
     * embedding, runs the plan's graphs in order and samples, and everything that makes this model
     * what it is lives in the graphs the plan holds.
     *
     * <p>Sequential prefill ingests the prompt through the plan's prefill graphs — the same layer
     * computation with the logits graph skipped — and then decodes. Batched prefill is refused by
     * the provider, which does not declare it.
     */
    // @formatter:on
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
                == org.beehive.jllm.runtime.policy.ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            // Sequential and batched both enter here: the loop reads the batch width from the
            // policy and ingests the prompt in chunks of it, one token at a time when it is one.
            // The shared prefill loop, told that this family's decode loop charges the whole
            // prompt against the token budget — which is what makes a prompt produce the same
            // number of tokens here as it does in STANDARD.
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
