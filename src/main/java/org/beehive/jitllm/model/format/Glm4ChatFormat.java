package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.tokenizer.Glm4Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * The GLM-4 chat template: {@code [gMASK]<sop>}, then each turn as its role token followed directly
 * by its text, and an assistant turn opened as {@code <|assistant|><think>}.
 *
 * <p>No end-of-turn token closes a message; the next role token does. Generation stops at the
 * tokens that open a turn the model must not write ({@code <|user|>}, {@code <|observation|>}) and
 * at {@code <|endoftext|>}.
 */
public class Glm4ChatFormat implements ChatFormat {

    private final Glm4Tokenizer tokenizer;
    private final int gMask;
    private final int sop;
    private final int system;
    private final int user;
    private final int assistant;
    private final int observation;
    private final int endOfText;
    private final int thinkStart;
    private final int thinkEnd;

    public Glm4ChatFormat(Glm4Tokenizer tokenizer, Vocabulary vocabulary) {
        this.tokenizer = tokenizer;
        this.gMask = id(vocabulary, "[gMASK]");
        this.sop = id(vocabulary, "<sop>");
        this.system = id(vocabulary, "<|system|>");
        this.user = id(vocabulary, "<|user|>");
        this.assistant = id(vocabulary, "<|assistant|>");
        this.observation = id(vocabulary, "<|observation|>");
        this.endOfText = id(vocabulary, "<|endoftext|>");
        this.thinkStart = id(vocabulary, "<think>");
        this.thinkEnd = id(vocabulary, "</think>");
    }

    private static int id(Vocabulary vocabulary, String token) {
        return vocabulary
                .getIndex(token)
                .orElseThrow(() -> new IllegalStateException("GLM-4 vocabulary has no " + token));
    }

    @Override
    public int getBeginOfText() {
        return gMask;
    }

    @Override
    public List<Integer> beginOfTextTokens() {
        return List.of(gMask, sop);
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        switch (message.role().name()) {
            case "system" -> tokens.add(system);
            case "user" -> tokens.add(user);
            case "assistant" -> {
                tokens.add(assistant);
                // An empty assistant message is the generation prompt, which the template opens
                // with <think>; a finished message's own text decides what follows.
                if (message.content().isEmpty()) {
                    tokens.add(thinkStart);
                }
            }
            default -> tokens.add(observation);
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = new ArrayList<>();
        String content = message.content();
        switch (message.role().name()) {
            case "system" -> tokens.add(system);
            case "user" -> tokens.add(user);
            case "assistant" -> {
                tokens.add(assistant);
                // The template drops earlier turns' reasoning and writes the turn pre-closed.
                tokens.add(thinkEnd);
                content = withoutReasoning(content);
            }
            default -> tokens.add(observation);
        }
        tokens.addAll(tokenizer.encodeOrdinaryAsList(content.strip()));
        return tokens;
    }

    /**
     * An assistant turn's text after its reasoning: everything up to the last {@code </think>} is
     * the reasoning. Other roles' text is literal, tags included.
     */
    private static String withoutReasoning(String content) {
        int closed = content.lastIndexOf("</think>");
        return closed < 0 ? content : content.substring(closed + "</think>".length());
    }

    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(endOfText, user, observation);
    }

    @Override
    public boolean supportsThinking() {
        return true;
    }

    @Override
    public int reasoningEndToken() {
        return thinkEnd;
    }

    /**
     * The generation prompt already opened {@code <think>}; disabling closes it at once, which is
     * the template's {@code enable_thinking=false} prompt with the opening tag kept.
     */
    @Override
    public List<Integer> encodeThinkingControl(boolean enableThinking) {
        return enableThinking ? List.of() : List.of(thinkEnd);
    }
}
