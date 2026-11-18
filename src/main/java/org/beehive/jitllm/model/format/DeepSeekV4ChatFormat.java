package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.tokenizer.DeepSeekV3Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * DeepSeek-V4's chat layout: {@code <｜begin▁of▁sentence｜>}, the system prompt as plain text, then
 * {@code <｜User｜>} and {@code <｜Assistant｜>} turns, each finished assistant turn closed by {@code
 * <｜end▁of▁sentence｜>}. An assistant turn opens with {@code </think>} when it does not reason and
 * with {@code <think>} when it does.
 *
 * <p>The base models carry no chat template and are used for plain completion; a vocabulary without
 * the turn markers (such as a test model's) encodes the turns as their text alone.
 */
public class DeepSeekV4ChatFormat implements ChatFormat {

    private final DeepSeekV3Tokenizer tokenizer;
    private final int beginOfSentence;
    private final int endOfSentence;
    private final int user;
    private final int assistant;
    private final int thinkStart;
    private final int thinkEnd;

    public DeepSeekV4ChatFormat(
            DeepSeekV3Tokenizer tokenizer, Vocabulary vocabulary, int bos, int eos) {
        this.tokenizer = tokenizer;
        this.beginOfSentence = bos;
        this.endOfSentence = eos;
        this.user = id(vocabulary, "<｜User｜>");
        this.assistant = id(vocabulary, "<｜Assistant｜>");
        this.thinkStart = id(vocabulary, "<think>");
        this.thinkEnd = id(vocabulary, "</think>");
    }

    private static int id(Vocabulary vocabulary, String token) {
        return vocabulary.getIndex(token).orElse(-1);
    }

    private static void add(List<Integer> tokens, int token) {
        if (token >= 0) {
            tokens.add(token);
        }
    }

    @Override
    public int getBeginOfText() {
        return beginOfSentence;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        switch (message.role().name()) {
            case "user" -> add(tokens, user);
            case "assistant" -> add(tokens, assistant);
            default -> {}
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = new ArrayList<>();
        String content = message.content();
        switch (message.role().name()) {
            case "user" -> add(tokens, user);
            case "assistant" -> {
                add(tokens, assistant);
                add(tokens, thinkEnd);
                int closed = content.lastIndexOf("</think>");
                content = closed < 0 ? content : content.substring(closed + "</think>".length());
            }
            default -> {}
        }
        tokens.addAll(tokenizer.encodeOrdinary(content));
        if (message.role().name().equals("assistant")) {
            add(tokens, endOfSentence);
        }
        return tokens;
    }

    @Override
    public Set<Integer> getStopTokens() {
        Set<Integer> stops = new HashSet<>();
        stops.add(endOfSentence);
        if (user >= 0) {
            stops.add(user);
        }
        return stops;
    }

    @Override
    public boolean supportsThinking() {
        return thinkStart >= 0 && thinkEnd >= 0;
    }

    @Override
    public int reasoningEndToken() {
        return thinkEnd;
    }

    @Override
    public List<Integer> encodeThinkingControl(boolean enableThinking) {
        List<Integer> tokens = new ArrayList<>();
        add(tokens, enableThinking ? thinkStart : thinkEnd);
        return tokens;
    }
}
