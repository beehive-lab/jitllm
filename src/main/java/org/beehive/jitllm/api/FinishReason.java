package org.beehive.jitllm.api;

/** Why generation stopped. */
@Experimental
public enum FinishReason {

    /** The model emitted one of its own stop tokens. */
    STOP_TOKEN,

    /** The request's {@code maxNewTokens} was reached. */
    MAX_TOKENS,

    /** One of the request's stop sequences appeared in the output. */
    STOP_SEQUENCE,

    /** The session ran out of context. */
    CONTEXT_FULL,

    /**
     * The model asked to call a tool.
     *
     * <p><b>Narrow on purpose</b> [A4]. Reported only when a valid tool call was extracted
     * <b>and</b> generation ended through the format's tool-call termination path — the tool-aware
     * stop-token set. Both conditions matter:
     *
     * <ul>
     *   <li><b>Supplying tools does not produce this.</b> Prose that comes back from a request with
     *       tools attached keeps its actual stop reason, because that is what happened.
     *   <li><b>Malformed tool-like output does not produce this.</b> Text that looks like a call
     *       but does not parse is text: it comes back as {@link #STOP_TOKEN} with the response
     *       intact and {@code toolCalls()} empty, so a caller sees what the model actually said
     *       rather than a successful call that never was.
     * </ul>
     */
    TOOL_CALL,

    /**
     * The request's {@link CancellationToken} was cancelled. The result holds the text produced
     * before generation stopped, and the session remains usable.
     */
    CANCELLED
}
