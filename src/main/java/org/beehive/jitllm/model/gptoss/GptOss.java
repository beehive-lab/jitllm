package org.beehive.jitllm.model.gptoss;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * OpenAI's gpt-oss (20b and 120b): a mixture-of-experts decoder with grouped-query attention that
 * alternates sliding-window and full layers, learned attention sinks, YaRN RoPE, and the Harmony
 * chat format.
 */
public final class GptOss extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("gpt-oss");

    private final GptOssConfiguration configuration;

    public GptOss(
            GptOssConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public GptOssConfiguration configuration() {
        return configuration;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GPT_OSS;
    }

    @Override
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }

    /** The key/value width: heads times the stated head size, not a fraction of {@code dim}. */
    @Override
    public int kvCacheDim() {
        return configuration.kvDim();
    }

    @Override
    public State createNewState() {
        return seeded(new GptOssState(configuration, -1));
    }

    @Override
    public State createNewState(int batchsize) {
        return seeded(new GptOssState(configuration, batchsize));
    }

    @Override
    public State createNewState(org.beehive.jitllm.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        return seeded(new GptOssState(configuration, -1, lease));
    }

    /** Seeded with the token that opens every Harmony message, which every prompt starts with. */
    private State seeded(State state) {
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    /** The Harmony system message, which the template writes before every conversation. */
    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    @Override
    public boolean shouldAddSystemPrompt() {
        return true;
    }

    /** No {@code <think>} prefix: Harmony's model opens its own analysis channel. */
    @Override
    public boolean shouldIncludeReasoning() {
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
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "gpt-oss has no sequential prefill/decode plan; use batched prefill or the"
                            + " standard path");
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
}
