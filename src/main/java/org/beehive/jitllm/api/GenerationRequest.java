package org.beehive.jllm.api;

import java.util.List;
import java.util.function.Consumer;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;

/**
 * What to generate, and how.
 *
 * <p>Immutable and thread-safe once built; the builder is not.
 *
 * <h2>A prompt, not a conversation</h2>
 *
 * <p>v1 carries a {@link Builder#prompt(String) prompt} and an optional {@link
 * Builder#systemPrompt(String) system prompt}. <b>Chat formatting is internal and model-driven</b>:
 * the model applies its own template, exactly as it does today, and the caller does not choose or
 * supply one.
 *
 * <p>There is no {@code messages(List<ChatMessage>)} in v1. It is deferred, not rejected: shipping
 * it would make a conversation type and a template-selection mechanism public API before the
 * multi-turn and tool-calling surfaces are designed, and both are far harder to change afterwards
 * than to add. Multi-turn today is a session that keeps its position between calls.
 */
public final class GenerationRequest {

    private final String prompt;
    private final String systemPrompt;
    private final int maxNewTokens;
    private final float temperature;
    private final float topP;
    private final Long seed;
    private final List<String> stopSequences;
    private final Consumer<String> onToken;
    private final List<ChatMessage> messages;
    private final List<ToolSpec> tools;
    private final java.util.function.Consumer<GenerationEvent> onEvent;

    private GenerationRequest(Builder builder) {
        this.messages = builder.messages == null ? null : List.copyOf(builder.messages);
        this.tools = List.copyOf(builder.tools);
        this.onEvent = builder.onEvent;
        this.prompt = builder.prompt;
        this.systemPrompt = builder.systemPrompt;
        this.maxNewTokens = builder.maxNewTokens;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.seed = builder.seed;
        this.stopSequences = List.copyOf(builder.stopSequences);
        this.onToken = builder.onToken;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Convenience for the common case: everything else defaulted. */
    public static GenerationRequest of(String prompt) {
        return builder().prompt(prompt).build();
    }

    public String prompt() {
        return prompt;
    }

    /** The system prompt, or {@code null} when the caller did not set one. */
    public String systemPrompt() {
        return systemPrompt;
    }

    public int maxNewTokens() {
        return maxNewTokens;
    }

    public float temperature() {
        return temperature;
    }

    public float topP() {
        return topP;
    }

    /**
     * The seed, or {@code null} for a fresh one per request. Sampling at temperature 0 ignores it.
     */
    public Long seed() {
        return seed;
    }

    public List<String> stopSequences() {
        return stopSequences;
    }

    /** Streaming callback, or {@code null}. Called on the thread that called {@code generate}. */
    /**
     * The complete conversation, or {@code null} when this request uses {@link #prompt()}.
     *
     * <p><b>The whole conversation, not the new turns</b> [§1]. "New turns only" would make a
     * request's meaning depend on the session's hidden position, so the same request object would
     * produce different output depending on what came before it.
     *
     * <p>A session may still reuse work: when a request's encoded input extends what the session
     * has already encoded, it keeps the key/value prefix and encodes only the remainder. That is a
     * timing property and never a semantic one.
     */
    @Experimental
    public List<ChatMessage> messages() {
        return messages;
    }

    /**
     * Receives one ordered event per emitted completion token, or {@code null}.
     *
     * <p>Runs <b>before</b> {@link #onToken()} for the same event. A throwing callback propagates
     * out of {@code generate}, and no later callback for that event runs [A5].
     */
    public java.util.function.Consumer<GenerationEvent> onEvent() {
        return onEvent;
    }

    /** Tools the model may call, never {@code null} — empty when none were given. */
    public List<ToolSpec> tools() {
        return tools;
    }

    public Consumer<String> onToken() {
        return onToken;
    }

    /** Not thread-safe; build one request per call. */
    /**
     * Exactly one of the two request forms, and a conversation that says something.
     *
     * <p>The two forms differ in more than convenience: {@code prompt} continues a session's
     * conversation turn by turn, while {@code messages} states the whole of it and lets the session
     * reuse what matches. Accepting both would mean choosing one of them for the caller.
     *
     * <p>On the outer class rather than on {@code Builder}: {@code FacadeSurfaceTest} pins the
     * builder's <b>declared</b> methods, so a private helper there widens a surface pin that is
     * meant to be exact. The test caught this, which is what an exact pin is for — and it is the
     * second time, after {@code ModelOptions}.
     */
    private static void requireExactlyOneRequestForm(
            String prompt, String systemPrompt, List<ChatMessage> messages) {
        boolean hasPrompt = prompt != null || systemPrompt != null;
        if (messages != null && hasPrompt) {
            throw new IllegalArgumentException(
                    DiagnosticCode.REQUEST_INVALID.prefix()
                            + "a request carries either messages(...) or"
                            + " prompt(...)/systemPrompt(...), not both: messages is the whole"
                            + " conversation, so a separate prompt has nowhere to go");
        }
        if (messages == null && prompt == null) {
            throw new IllegalArgumentException(
                    "a request needs either prompt(...) or" + " messages(...)");
        }
        if (messages != null && messages.isEmpty()) {
            throw new IllegalArgumentException(
                    "messages(...) must not be empty;"
                            + " an empty conversation encodes to nothing to answer");
        }
    }

    public static final class Builder {

        private String prompt;
        private String systemPrompt;
        private int maxNewTokens = 512;
        private float temperature = 0.1f;
        private float topP = 0.95f;
        private Long seed;
        private List<String> stopSequences = List.of();
        private Consumer<String> onToken;
        private List<ChatMessage> messages;
        private List<ToolSpec> tools = List.of();
        private java.util.function.Consumer<GenerationEvent> onEvent;

        private Builder() {}

        public Builder prompt(String prompt) {
            this.prompt = prompt;
            return this;
        }

        /** Optional. The model decides how to place it; there is no template control in v1. */
        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        /** 0 is greedy: the highest-probability token every time, and reproducible. */
        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Generation stops when one of these appears in the output. */
        public Builder stopSequences(List<String> stopSequences) {
            this.stopSequences = stopSequences == null ? List.of() : List.copyOf(stopSequences);
            return this;
        }

        /**
         * Receives one ordered event per emitted completion token — its id and the text it
         * completed — on the calling thread of {@code generate(.)}.
         *
         * <p>This is the surface for a consumer that needs token ids. It runs before {@link
         * #onToken(Consumer)}, which is defined in terms of it.
         */
        public Builder onEvent(java.util.function.Consumer<GenerationEvent> onEvent) {
            this.onEvent = onEvent;
            return this;
        }

        /**
         * Receives each token's text as it is produced, on the calling thread of {@code
         * generate(.)}. A callback that throws propagates out of {@code generate}.
         *
         * <p>Kept, and <b>defined in terms of the event stream</b>: invoked with {@code
         * event.text()} for each event whose text is non-empty. Callers that only want text see no
         * change, and the two cannot disagree, because there is only one stream.
         */
        public Builder onToken(Consumer<String> onToken) {
            this.onToken = onToken;
            return this;
        }

        /**
         * The complete conversation as of this request — every turn, including those already sent.
         *
         * <p>Mutually exclusive with {@link #prompt(String)} and {@link #systemPrompt(String)}: a
         * request carrying both is a caller with two ideas about what it is asking, and picking one
         * silently is how the wrong one ships.
         *
         * @param messages a non-empty conversation
         */
        @Experimental
        public Builder messages(List<ChatMessage> messages) {
            this.messages = messages == null ? null : List.copyOf(messages);
            return this;
        }

        /**
         * Tools the model may call. May accompany <b>either</b> request form.
         *
         * <p>Whether the model's format can do tool calling is checked at generation, where the
         * model is known — not here, where it is not.
         */
        public Builder tools(List<ToolSpec> tools) {
            this.tools = tools == null ? List.of() : List.copyOf(tools);
            return this;
        }

        public GenerationRequest build() {
            requireExactlyOneRequestForm(prompt, systemPrompt, messages);
            if (maxNewTokens <= 0) {
                throw new IllegalArgumentException(
                        "maxNewTokens must be positive: " + maxNewTokens);
            }
            if (temperature < 0) {
                throw new IllegalArgumentException(
                        "temperature must not be negative: " + temperature);
            }
            if (topP <= 0 || topP > 1) {
                throw new IllegalArgumentException("topP must be in (0, 1]: " + topP);
            }
            return new GenerationRequest(this);
        }
    }
}
