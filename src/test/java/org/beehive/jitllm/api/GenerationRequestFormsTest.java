package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

public class GenerationRequestFormsTest {

    private static final ChatMessage USER = ChatMessage.of(ChatRole.USER, "what time is it?");

    @Test
    public void aPromptRequestStillWorksExactlyAsBefore() {
        GenerationRequest request = GenerationRequest.of("hello");
        assertEquals("hello", request.prompt());
        assertNull("the prompt form carries no conversation", request.messages());
        assertTrue("and no tools unless asked", request.tools().isEmpty());
    }

    @Test
    public void aConversationRequestCarriesTheWholeConversation() {
        GenerationRequest request =
                GenerationRequest.builder()
                        .messages(List.of(ChatMessage.of(ChatRole.SYSTEM, "be brief"), USER))
                        .build();
        assertEquals(2, request.messages().size());
        assertNull(request.prompt());
    }

    @Test
    public void thetwoFormsAreMutuallyExclusive() {
        // Picking one silently is how the wrong one ships.
        IllegalArgumentException both =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                GenerationRequest.builder()
                                        .prompt("hi")
                                        .messages(List.of(USER))
                                        .build());
        assertTrue(both.getMessage(), both.getMessage().contains("not both"));

        assertThrows(
                "a system prompt is part of the prompt form",
                IllegalArgumentException.class,
                () ->
                        GenerationRequest.builder()
                                .systemPrompt("be brief")
                                .messages(List.of(USER))
                                .build());
    }

    @Test
    public void aRequestWithNeitherFormIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> GenerationRequest.builder().build());
    }

    @Test
    public void anEmptyConversationIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().messages(List.of()).build());
    }

    /** Tools may accompany either form; whether the model can serve them is checked at generate. */
    @Test
    public void toolsMayAccompanyEitherForm() {
        ToolSpec clock = new ToolSpec("get_current_time", "the time now", "{}");
        assertEquals(
                1,
                GenerationRequest.builder()
                        .prompt("what time is it?")
                        .tools(List.of(clock))
                        .build()
                        .tools()
                        .size());
        assertEquals(
                1,
                GenerationRequest.builder()
                        .messages(List.of(USER))
                        .tools(List.of(clock))
                        .build()
                        .tools()
                        .size());
    }

    @Test
    public void messagesAndToolsAreCopiedAndImmutable() {
        java.util.List<ChatMessage> mutable = new java.util.ArrayList<>(List.of(USER));
        GenerationRequest request = GenerationRequest.builder().messages(mutable).build();
        mutable.add(USER);
        assertEquals(
                "the request kept the conversation it was built with",
                1,
                request.messages().size());
        assertThrows(UnsupportedOperationException.class, () -> request.messages().add(USER));
        assertThrows(
                UnsupportedOperationException.class,
                () -> request.tools().add(new ToolSpec("t", "", "{}")));
    }
}
