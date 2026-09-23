package org.beehive.jitllm.api;

/**
 * One piece of a conversation turn.
 *
 * <p><b>Sealed</b>, so a fourth kind — an image, say — is a reviewed decision rather than something
 * that arrives because an interface was open. Multimodal content is deliberately not in this
 * milestone, and sealing is what keeps adding it a decision.
 *
 * <p>The JSON-bearing strings are <b>opaque</b>. This milestone introduces no JSON-schema model:
 * both integrations already hold JSON at this boundary, the engine only splices it into a prompt,
 * and a modelled schema would be a second, worse copy of a specification this project does not own.
 */
public sealed interface ChatContent {

    /** Plain text. */
    record Text(String text) implements ChatContent {
        public Text {
            if (text == null) {
                throw new IllegalArgumentException(
                        "message text must not be null;"
                                + " use an empty string if a turn genuinely has no text");
            }
        }
    }

    /**
     * A tool the assistant asked to run.
     *
     * @param id the call's identifier, non-blank — it is what a {@link ToolResult} is matched
     *     against
     * @param name the tool's name, non-blank
     * @param argumentsJson the arguments as an opaque JSON string, never {@code null}
     */
    record ToolCall(String id, String name, String argumentsJson) implements ChatContent {
        public ToolCall {
            requireNonBlank(id, "a tool call id");
            requireNonBlank(name, "a tool name");
            if (argumentsJson == null) {
                throw new IllegalArgumentException(
                        "tool call arguments must not be null;"
                                + " an argumentless call is an empty JSON object, not null");
            }
        }
    }

    /**
     * What running a tool produced.
     *
     * @param id the {@link ToolCall#id()} this answers, non-blank
     * @param name the tool's name, non-blank
     * @param resultJson the result as an opaque JSON string, never {@code null}
     */
    record ToolResult(String id, String name, String resultJson) implements ChatContent {
        public ToolResult {
            requireNonBlank(id, "a tool result id");
            requireNonBlank(name, "a tool name");
            if (resultJson == null) {
                throw new IllegalArgumentException(
                        "a tool result must not be null;"
                                + " a tool that returns nothing returns an empty JSON object, not null");
            }
        }
    }

    private static void requireNonBlank(String value, String what) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    what
                            + " must not be blank, got "
                            + (value == null ? "null" : "'" + value + "'"));
        }
    }
}
