package org.beehive.jllm.api;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;

/**
 * One turn in a conversation: who spoke, and what they said.
 *
 * <p>Immutable, with a defensive copy of the content list, so a caller mutating the list it passed
 * cannot change a request that was already built.
 *
 * <h2>Role and content are validated here, not by a formatter later</h2>
 *
 * <p>A {@code TOOL} message carrying text, or a {@code USER} message carrying a tool call, is a
 * caller error, and the caller should learn about it where they wrote it — not several layers down
 * as a complaint about tokens from a chat format they never named.
 *
 * <table>
 *   <caption>Legal combinations</caption>
 *   <tr><th>Role</th><th>Content</th></tr>
 *   <tr><td>{@code SYSTEM}, {@code USER}</td><td>{@link ChatContent.Text} only</td></tr>
 *   <tr><td>{@code ASSISTANT}</td><td>{@code Text} and/or {@link ChatContent.ToolCall}</td></tr>
 *   <tr><td>{@code TOOL}</td><td>{@link ChatContent.ToolResult} only</td></tr>
 * </table>
 */
@Experimental
public record ChatMessage(ChatRole role, List<ChatContent> content) {

    public ChatMessage {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(content, "content");
        if (content.isEmpty()) {
            throw new IllegalArgumentException(
                    DiagnosticCode.MESSAGES_INVALID.prefix()
                            + "a "
                            + role
                            + " message must carry content;"
                            + " an empty turn says nothing and encodes to nothing");
        }
        content = List.copyOf(content); // defensive, and rejects a null element
        for (ChatContent piece : content) {
            requireLegal(role, piece);
        }
    }

    /** A single-text turn, which is most of them. */
    public static ChatMessage of(ChatRole role, String text) {
        return new ChatMessage(role, List.of(new ChatContent.Text(text)));
    }

    private static void requireLegal(ChatRole role, ChatContent piece) {
        boolean legal =
                switch (role) {
                    case SYSTEM, USER -> piece instanceof ChatContent.Text;
                    case ASSISTANT ->
                            piece instanceof ChatContent.Text
                                    || piece instanceof ChatContent.ToolCall;
                    case TOOL -> piece instanceof ChatContent.ToolResult;
                };
        if (!legal) {
            throw new IllegalArgumentException(
                    "a "
                            + role
                            + " message may not carry "
                            + piece.getClass().getSimpleName()
                            + " content. "
                            + legalContentFor(role));
        }
    }

    private static String legalContentFor(ChatRole role) {
        return switch (role) {
            case SYSTEM, USER -> role + " carries Text only";
            case ASSISTANT -> "ASSISTANT carries Text and/or ToolCall";
            case TOOL -> "TOOL carries ToolResult only";
        };
    }
}
