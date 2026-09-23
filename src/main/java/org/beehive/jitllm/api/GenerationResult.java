package org.beehive.jitllm.api;

import java.util.Objects;

/**
 * What one {@link GenerationSession#generate(GenerationRequest)} produced.
 *
 * <p>Immutable and thread-safe. The text is the whole completion, whether or not the caller also
 * received it token by token through {@link GenerationRequest#onToken()}.
 */
public final class GenerationResult {

    private final String text;
    private final int promptTokens;
    private final int generatedTokens;
    private final FinishReason finishReason;
    private final GenerationTimings timings;
    private final java.util.List<ChatContent.ToolCall> toolCalls;

    /** A result with no tool calls — every path that cannot produce one. */
    @Experimental
    public GenerationResult(
            String text,
            int promptTokens,
            int generatedTokens,
            FinishReason finishReason,
            GenerationTimings timings) {
        this(text, promptTokens, generatedTokens, finishReason, timings, java.util.List.of());
    }

    @Experimental
    public GenerationResult(
            String text,
            int promptTokens,
            int generatedTokens,
            FinishReason finishReason,
            GenerationTimings timings,
            java.util.List<ChatContent.ToolCall> toolCalls) {
        this.text = Objects.requireNonNull(text, "text");
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
        this.finishReason = Objects.requireNonNull(finishReason, "finishReason");
        this.timings = Objects.requireNonNull(timings, "timings");
        this.toolCalls = java.util.List.copyOf(toolCalls);
    }

    public String text() {
        return text;
    }

    /** Tokens the prompt occupied, including whatever the model's template added. */
    public int promptTokens() {
        return promptTokens;
    }

    public int generatedTokens() {
        return generatedTokens;
    }

    @Experimental
    public FinishReason finishReason() {
        return finishReason;
    }

    @Experimental
    public GenerationTimings timings() {
        return timings;
    }

    /**
     * The tool calls the model asked for — <b>immutable</b>, and <b>empty</b> when it asked for
     * none [A4].
     *
     * <p>Empty is also what a caller gets when the model produced tool-shaped text that did not
     * parse. That case is ordinary text with an ordinary stop reason, and {@link
     * FinishReason#TOOL_CALL} is not reported for it: a malformed call reported as a successful one
     * would have the caller execute something the model never asked for.
     */
    public java.util.List<ChatContent.ToolCall> toolCalls() {
        return toolCalls;
    }

    @Override
    public String toString() {
        return "GenerationResult[" + generatedTokens + " tokens, " + finishReason + "]";
    }
}
