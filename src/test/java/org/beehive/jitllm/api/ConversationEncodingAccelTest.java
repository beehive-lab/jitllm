package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * Against the legacy assembly, because the claim being made is that a single-turn conversation is
 * the same input as the prompt form. That is easier to assert here than through generated text, and
 * it fails with the actual token lists rather than with an out-of-range index thirty frames down.
 */
public class ConversationEncodingAccelTest {

    private static final String TEXT = "Name one colour.";

    @Test
    public void aSingleTurnConversationEncodesLikeThePromptForm() throws Exception {
        Model model = loadOrSkip();
        ChatFormat chatFormat = model.chatFormat();

        List<Integer> legacy = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            legacy.add(chatFormat.getBeginOfText());
        }
        legacy.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, TEXT)));
        legacy.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<Integer> conversation =
                new ConversationEncoder(model, ThinkingMode.DEFAULT)
                        .encode(List.of(ChatMessage.of(ChatRole.USER, TEXT)), List.of());

        assertEquals(
                "a one-turn conversation is the prompt form, token for token",
                legacy,
                conversation);
    }

    @Test
    public void everyEncodedTokenIsInVocabulary() {
        Model model = loadOrSkip();
        int vocabulary = model.configuration().vocabularySize();
        List<Integer> tokens =
                new ConversationEncoder(model, ThinkingMode.DEFAULT)
                        .encode(
                                List.of(
                                        ChatMessage.of(ChatRole.SYSTEM, "be brief"),
                                        ChatMessage.of(ChatRole.USER, TEXT)),
                                List.of());
        for (int token : tokens) {
            org.junit.Assert.assertTrue(
                    "token "
                            + token
                            + " is outside the vocabulary of "
                            + vocabulary
                            + "; an out-of-range id reaches the embedding table as an index",
                    token >= 0 && token < vocabulary);
        }
    }

    /** Acceptance: attaching a tool changes the encoded input, which is why reuse compares it. */
    @Test
    public void toolsChangeTheEncodedPrefix() {
        Model model = loadOrSkip();
        ConversationEncoder encoder = new ConversationEncoder(model, ThinkingMode.DEFAULT);
        List<ChatMessage> conversation = List.of(ChatMessage.of(ChatRole.USER, TEXT));

        List<Integer> without = encoder.encode(conversation, List.of());
        if (!encoder.supportsTools()) {
            return; // a format without tool calling has nothing to assert here
        }
        List<Integer> with =
                encoder.encode(
                        conversation,
                        List.of(new ToolSpec("get_current_time", "the time now", "{}")));

        assertNotEquals(
                "a tool specification changes the encoded system content, so the input"
                        + " diverges from its first tokens while the message list looks unchanged",
                without,
                with);
    }

    private static Model loadOrSkip() {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent", false);
        }
        try {
            return ModelLoader.loadModel(modelPath, 512, true, false);
        } catch (Exception e) {
            throw new AssertionError("the fixture failed to load", e);
        }
    }
}
