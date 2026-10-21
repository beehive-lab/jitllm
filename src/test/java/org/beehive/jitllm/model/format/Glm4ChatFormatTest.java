package org.beehive.jitllm.model.format;

import static org.junit.Assert.assertEquals;

import java.util.List;
import org.beehive.jitllm.model.format.ChatFormat.Message;
import org.beehive.jitllm.model.format.ChatFormat.Role;
import org.beehive.jitllm.tokenizer.Glm4Tokenizer;
import org.beehive.jitllm.tokenizer.TestVocabularies;
import org.beehive.jitllm.tokenizer.Vocabulary;
import org.junit.Test;

/**
 * {@link Glm4ChatFormat#encodeMessage}: an assistant turn loses its reasoning, every other role's
 * text is encoded as written, a literal {@code </think>} included.
 */
public class Glm4ChatFormatTest {

    private static final Vocabulary VOCABULARY = TestVocabularies.glm4Vocabulary();
    private static final Glm4Tokenizer TOKENIZER = TestVocabularies.glm4(VOCABULARY);
    private static final Glm4ChatFormat FORMAT = new Glm4ChatFormat(TOKENIZER, VOCABULARY);

    private static final Role OBSERVATION = new Role("observation");

    private static int id(String token) {
        return VOCABULARY.getIndex(token).orElseThrow();
    }

    /** The role tokens the message opens with, then its text decoded. */
    private static void assertEncodes(Message message, List<Integer> opening, String text) {
        List<Integer> tokens = FORMAT.encodeMessage(message);
        assertEquals(opening, tokens.subList(0, opening.size()));
        assertEquals(text, TOKENIZER.decode(tokens.subList(opening.size(), tokens.size())));
    }

    @Test
    public void aUserMessageKeepsALiteralClosingTag() {
        assertEncodes(
                new Message(Role.USER, "Explain the literal tag </think> in this document."),
                List.of(id("<|user|>")),
                "Explain the literal tag </think> in this document.");
    }

    @Test
    public void aUserMessageKeepsRepeatedTags() {
        String text = "<think>a</think> b </think> c";
        assertEncodes(new Message(Role.USER, text), List.of(id("<|user|>")), text);
    }

    @Test
    public void aSystemMessageKeepsALiteralClosingTag() {
        String text = "Never print </think> on its own.";
        assertEncodes(new Message(Role.SYSTEM, text), List.of(id("<|system|>")), text);
    }

    @Test
    public void anObservationKeepsALiteralClosingTag() {
        String text = "{\"log\": \"line </think> line\"}";
        assertEncodes(new Message(OBSERVATION, text), List.of(id("<|observation|>")), text);
    }

    @Test
    public void anAssistantTurnDropsItsReasoning() {
        assertEncodes(
                new Message(Role.ASSISTANT, "<think>step one</think>\nThe answer is 4."),
                List.of(id("<|assistant|>"), id("</think>")),
                "The answer is 4.");
    }

    @Test
    public void anAssistantTurnDropsEverythingUpToTheLastClosingTag() {
        assertEncodes(
                new Message(Role.ASSISTANT, "<think>a</think>b</think> Done."),
                List.of(id("<|assistant|>"), id("</think>")),
                "Done.");
    }

    @Test
    public void anAssistantTurnWithoutReasoningIsKept() {
        assertEncodes(
                new Message(Role.ASSISTANT, "The answer is 4."),
                List.of(id("<|assistant|>"), id("</think>")),
                "The answer is 4.");
    }
}
