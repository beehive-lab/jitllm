package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * Every illegal combination is enumerated rather than sampled, because "the formatter will complain
 * later" is exactly the behaviour this replaces, and a gap in the table is a combination that
 * reaches a chat format and fails there instead.
 */
public class ChatMessageValidationTest {

    private static final ChatContent TEXT = new ChatContent.Text("hello");
    private static final ChatContent CALL = new ChatContent.ToolCall("id-1", "clock", "{}");
    private static final ChatContent RESULT = new ChatContent.ToolResult("id-1", "clock", "{}");

    private static void assertRejected(ChatRole role, ChatContent content) {
        IllegalArgumentException thrown =
                assertThrows(
                        role + " must not carry " + content.getClass().getSimpleName(),
                        IllegalArgumentException.class,
                        () -> new ChatMessage(role, List.of(content)));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains(role.name()));
    }

    @Test
    public void everyIllegalRoleAndContentCombinationIsRejected() {
        assertRejected(ChatRole.SYSTEM, CALL);
        assertRejected(ChatRole.SYSTEM, RESULT);
        assertRejected(ChatRole.USER, CALL);
        assertRejected(ChatRole.USER, RESULT);
        assertRejected(ChatRole.ASSISTANT, RESULT);
        assertRejected(ChatRole.TOOL, TEXT);
        assertRejected(ChatRole.TOOL, CALL);
    }

    @Test
    public void everyLegalCombinationIsAccepted() {
        new ChatMessage(ChatRole.SYSTEM, List.of(TEXT));
        new ChatMessage(ChatRole.USER, List.of(TEXT));
        new ChatMessage(ChatRole.ASSISTANT, List.of(TEXT));
        new ChatMessage(ChatRole.ASSISTANT, List.of(CALL));
        new ChatMessage(ChatRole.ASSISTANT, List.of(TEXT, CALL)); // text and a call together
        new ChatMessage(ChatRole.TOOL, List.of(RESULT));
    }

    @Test
    public void theErrorNamesTheRoleAndWhatThatRoleMayCarry() {
        IllegalArgumentException thrown =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new ChatMessage(ChatRole.TOOL, List.of(TEXT)));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("TOOL"));
        assertTrue(
                "the message must say what is allowed, not only what is not",
                thrown.getMessage().contains("ToolResult"));
    }

    @Test
    public void anEmptyTurnIsRejected() {
        assertThrows(
                IllegalArgumentException.class, () -> new ChatMessage(ChatRole.USER, List.of()));
    }

    @Test
    public void contentIsCopiedSoALaterMutationCannotChangeABuiltMessage() {
        List<ChatContent> mutable = new ArrayList<>();
        mutable.add(TEXT);
        ChatMessage message = new ChatMessage(ChatRole.USER, mutable);
        mutable.add(CALL); // would be illegal for USER, and must not reach the message
        assertEquals(1, message.content().size());
        assertThrows(UnsupportedOperationException.class, () -> message.content().add(TEXT));
    }

    @Test
    public void toolIdentifiersAndPayloadsAreValidated() {
        assertThrows(
                IllegalArgumentException.class, () -> new ChatContent.ToolCall(" ", "clock", "{}"));
        assertThrows(
                IllegalArgumentException.class, () -> new ChatContent.ToolCall("id", "", "{}"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatContent.ToolCall("id", "clock", null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatContent.ToolResult(null, "clock", "{}"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatContent.ToolResult("id", "clock", null));
        assertThrows(IllegalArgumentException.class, () -> new ToolSpec("  ", "d", "{}"));
    }

    /** The JSON strings stay opaque: this milestone introduces no schema model [A3]. */
    @Test
    public void jsonStringsAreNotParsedOrValidated() {
        // Nonsense JSON is accepted, because validating it would be the first half of owning a
        // schema specification this project does not own.
        new ChatContent.ToolCall("id", "clock", "not json at all");
        new ToolSpec("clock", "", "also not json");
    }

    @Test
    public void thereIsNoSingleTextConvenienceAccessor() {
        assertTrue(
                "a text() accessor on ChatMessage was deferred, not forgotten",
                java.util.Arrays.stream(ChatMessage.class.getMethods())
                        .noneMatch(m -> m.getName().equals("text")));
    }
}
