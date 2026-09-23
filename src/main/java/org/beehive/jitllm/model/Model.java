package org.beehive.jitllm.model;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.tokenizer.Tokenizer;

public interface Model {

    Configuration configuration();

    Tokenizer tokenizer();

    Weights weights();

    ChatFormat chatFormat();

    ModelType getModelType();

    /**
     * Which architecture this model is, as the identity everything downstream keys on.
     *
     * <p><b>Stated by the family, not re-derived.</b> Recognition already happened — a provider
     * chose this identity when it claimed the file — and asking a second registry to work it out
     * again from the configuration is how two answers appear for one model. It is also ambiguous
     * where it matters most: {@code Qwen2Configuration} belongs to both Qwen2 and
     * DeepSeek-R1-Distill-Qwen, so a configuration-type check cannot tell them apart.
     *
     * <p>No central switch: each family returns its own constant (Rule 15). A family with no
     * program description still has an identity — the architecture registry answers "nothing
     * computes this", which is the normal unsupported answer, not an error in the model.
     */
    default org.beehive.jitllm.runtime.model.ArchitectureId architectureId() {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not state an architecture identity yet");
    }

    State createNewState();

    State createNewState(int batchsize);

    /**
     * A state whose KV lives in the storage this lease addresses, rather than in arrays the state
     * allocates for itself.
     *
     * <p>Default: ignore the lease and build an ordinary state. That is the retained legacy path
     * every family except Llama still takes, and it is behaviour-identical to what they did before
     * — the lease is still held by the session, so the accounting and the pinning are real either
     * way.
     *
     * <p>The parameter is a <b>lease</b>, not a cache: the model reads what it was handed and owns
     * nothing [Rule 7].
     */
    default State createNewState(org.beehive.jitllm.runtime.kv.KvLease lease) {
        return createNewState();
    }

    /**
     * KV values stored per token, per layer — the {@code kvDim} the KV kernels index by.
     *
     * <p>Default: {@code dim * nKvHeads / nHeads}, which is what every family except Qwen3 uses.
     * Qwen3 sizes its cache from its own head dimensions instead, which is why this is a question
     * the model answers rather than one the cache derives.
     */
    /**
     * Whether this family may lease <b>shared</b> KV storage, as opposed to merely addressing its
     * own through a block table.
     *
     * <p>Asked of the model rather than kept as a list of families elsewhere [Rule 15].
     */
    default boolean supportsSharedKvStorage() {
        return false;
    }

    default int kvCacheDim() {
        Configuration config = configuration();
        return config.dim() * config.numberOfKeyValueHeads() / config.numberOfHeads();
    }

    /**
     * Host-side staging this family needs before its device graphs run for {@code token}.
     *
     * <p>Default: nothing. A family overrides it when a per-token input is too large to keep
     * resident and has to be gathered on the host each step — Gemma 4's {@code
     * per_layer_token_embd} is 2.35 billion elements, so only the current token's row is staged.
     * The alternative is a family test inside the shared forward pass, which is the central switch
     * Rule 15 forbids.
     */
    default void stagePerTokenDeviceInputs(State state, int token) {}

    /**
     * The chunk-wide twin of {@link #stagePerTokenDeviceInputs}: whatever a family has to put in a
     * device buffer per prompt token before a batched prefill graph runs, for every token of the
     * chunk.
     *
     * <p>A default no-op for the same reason the single-token hook is one, and separate from it
     * because the destination is a row of a chunk-wide buffer rather than a buffer of its own.
     */
    default void stageBatchDeviceInputs(State state, int[] tokens, int chunkSize) {}

    default boolean shouldAddBeginOfText() {
        return true;
    }

    default boolean shouldAddSystemPrompt() {
        return true;
    }

    default boolean shouldIncludeReasoning() {
        return false;
    }

    /** Wrapper for invoking the model-specific {@code TokenGenerationLoop.generateTokens} call. */
    List<Integer> generateTokens(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated);

    List<Integer> generateTokensGPU(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan);

    // ── Transitional generation bridges ───────────────────────────────
    //
    // The loops moved to org.beehive.jitllm.generation.ModelGeneration under Rule 8a: a model
    // that owns a generation loop cannot be an embedding or reranking model, and it drags a CLI
    // options record and System.out into the interface every backend implements.
    //
    // These three remain as thin delegates for callers outside this repository, and because this
    // project deprecates with a documented replacement before removing. They add no behaviour.
    //
    // They are NOT kept for the LangChain4j and Quarkus integrations: both were audited on
    // 2026-09-01 and neither calls them — they use the lower-level engine API directly.
}
