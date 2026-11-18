package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.tokenizer.KimiTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * The Kimi chat template: each turn is a role token, the role's name as text, {@code
 * <|im_middle|>}, the message, and {@code <|im_end|>}, with nothing between turns. The generation
 * prompt is {@code <|im_assistant|>assistant<|im_middle|>}. No token opens the conversation.
 *
 * <p>Generation stops at {@code <|im_end|>} and at {@code [EOS]}.
 */
public class KimiChatFormat implements ChatFormat {

    private final KimiTokenizer tokenizer;
    private final int beginOfText;
    private final int endOfText;
    private final int system;
    private final int user;
    private final int assistant;
    private final int middle;
    private final int end;

    public KimiChatFormat(Tokenizer tokenizer, Vocabulary vocabulary) {
        this.tokenizer = (KimiTokenizer) tokenizer;
        this.beginOfText = id(vocabulary, "[BOS]");
        this.endOfText = id(vocabulary, "[EOS]");
        this.system = id(vocabulary, "<|im_system|>");
        this.user = id(vocabulary, "<|im_user|>");
        this.assistant = id(vocabulary, "<|im_assistant|>");
        this.middle = id(vocabulary, "<|im_middle|>");
        this.end = id(vocabulary, "<|im_end|>");
    }

    private static int id(Vocabulary vocabulary, String token) {
        return vocabulary
                .getIndex(token)
                .orElseThrow(() -> new IllegalStateException("Kimi vocabulary has no " + token));
    }

    /** The template writes no token before the first turn. */
    @Override
    public List<Integer> beginOfTextTokens() {
        return List.of();
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        String role = message.role().name();
        List<Integer> tokens = new ArrayList<>();
        tokens.add(
                switch (role) {
                    case "user" -> user;
                    case "assistant" -> assistant;
                    default -> system;
                });
        tokens.addAll(tokenizer.encodeOrdinaryAsList(role));
        tokens.add(middle);
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeOrdinaryAsList(message.content()));
        tokens.add(end);
        return tokens;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(end, endOfText);
    }
}
