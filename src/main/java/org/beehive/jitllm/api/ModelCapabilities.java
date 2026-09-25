package org.beehive.jitllm.api;

/**
 * What a loaded model's chat format can do, so a caller can check before sending a request rather
 * than find out from the exception the request would throw.
 *
 * <p>Reached as {@link ModelInfo#capabilities()}. Each flag is the answer to "will a request that
 * uses this be accepted":
 *
 * <ul>
 *   <li>{@link #toolCalling()} — a {@link GenerationRequest} with {@link
 *       GenerationRequest.Builder#tools tools} is encoded in the family's native tool format and
 *       its calls are extracted into {@link GenerationResult#toolCalls()}. When {@code false}, such
 *       a request fails with an {@link UnsupportedOperationException}.
 *   <li>{@link #thinkingControl()} — the family has a reasoning phase that {@link
 *       ThinkingMode#ENABLED} and {@link ThinkingMode#DISABLED} can switch. When {@code false},
 *       asking for either fails with an {@link IllegalArgumentException}; {@link
 *       ThinkingMode#DEFAULT} is always accepted.
 * </ul>
 *
 * <p>Immutable and thread-safe.
 *
 * @param toolCalling whether requests carrying tools are supported
 * @param thinkingControl whether an explicit {@link ThinkingMode} is supported
 */
@Experimental
public record ModelCapabilities(boolean toolCalling, boolean thinkingControl) {

    /** Neither tool calling nor thinking control: what a plain text-completion format offers. */
    public static final ModelCapabilities NONE = new ModelCapabilities(false, false);
}
