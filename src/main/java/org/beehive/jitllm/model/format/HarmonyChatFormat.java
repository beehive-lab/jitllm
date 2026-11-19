package org.beehive.jitllm.model.format;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.tokenizer.GptOssTokenizer;

/**
 * OpenAI's Harmony format, as gpt-oss's chat template renders it.
 *
 * <p>Every conversation opens with a system message the template always writes (model identity,
 * knowledge cutoff, today's date, the reasoning effort and the valid channels). A caller's system
 * prompt becomes the developer message, under {@code # Instructions}. Each message is {@code
 * <|start|>role<|message|>content<|end|>}, and the generation prompt is {@code <|start|>assistant}:
 * the model then names a channel itself, reasoning in {@code analysis} and answering in {@code
 * final}, which it ends with {@code <|return|>}.
 */
public class HarmonyChatFormat implements ChatFormat {

    /** The template's default, used when no effort is configured. */
    public static final String DEFAULT_REASONING_EFFORT = "medium";

    private final GptOssTokenizer tokenizer;
    private final int start;
    private final int end;
    private final int message;
    private final int channel;
    private final int returnToken;
    private final int call;
    private final int endOfText;
    private final String reasoningEffort;

    public HarmonyChatFormat(GptOssTokenizer tokenizer) {
        this(
                tokenizer,
                System.getProperty("jitllm.gptoss.reasoningEffort", DEFAULT_REASONING_EFFORT));
    }

    public HarmonyChatFormat(GptOssTokenizer tokenizer, String reasoningEffort) {
        this.tokenizer = tokenizer;
        this.start = tokenizer.special("<|start|>");
        this.end = tokenizer.special("<|end|>");
        this.message = tokenizer.special("<|message|>");
        this.channel = tokenizer.special("<|channel|>");
        this.returnToken = tokenizer.special("<|return|>");
        this.call = tokenizer.special("<|call|>");
        this.endOfText = tokenizer.special("<|endoftext|>");
        this.reasoningEffort = reasoningEffort;
    }

    /** The template's system message for {@code date}: what opens every conversation. */
    public String systemMessage(LocalDate date) {
        return "You are ChatGPT, a large language model trained by OpenAI.\n"
                + "Knowledge cutoff: 2024-06\n"
                + "Current date: "
                + date
                + "\n\n"
                + "Reasoning: "
                + reasoningEffort
                + "\n\n"
                + "# Valid channels: analysis, commentary, final. Channel must be included for every"
                + " message.";
    }

    /** The system message, dated today, as the template writes it. */
    @Override
    public List<Integer> beginOfTextTokens() {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(start);
        tokens.addAll(text("system"));
        tokens.add(message);
        tokens.addAll(text(systemMessage(LocalDate.now())));
        tokens.add(end);
        return tokens;
    }

    /** {@code <|start|>}: Harmony has no beginning-of-text token, every message opens with this. */
    @Override
    public int getBeginOfText() {
        return start;
    }

    @Override
    public List<Integer> encodeHeader(Message msg) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(start);
        tokens.addAll(text(roleName(msg.role())));
        return tokens;
    }

    /**
     * A whole message. A system prompt is written as the developer message. An assistant message is
     * the final-channel answer only, as the template renders an earlier turn; an empty one is the
     * header and {@code <|end|>}, so the tokens that close an assistant turn are {@code <|end|>}.
     */
    @Override
    public List<Integer> encodeMessage(Message msg) {
        List<Integer> tokens = encodeHeader(msg);
        String content = msg.content();
        if (msg.role().equals(Role.SYSTEM)) {
            content = "# Instructions\n\n" + content + "\n\n";
        }
        if (msg.role().equals(Role.ASSISTANT)) {
            if (content.isEmpty()) {
                tokens.add(end);
                return tokens;
            }
            tokens.add(channel);
            tokens.addAll(text("final"));
        }
        tokens.add(message);
        tokens.addAll(text(content));
        tokens.add(end);
        return tokens;
    }

    private static String roleName(Role role) {
        return role.equals(Role.SYSTEM) ? "developer" : role.name();
    }

    private List<Integer> text(String text) {
        return tokenizer.encodeOrdinaryAsList(text);
    }

    /** {@code <|return|>} ends an answer, {@code <|call|>} a tool call. */
    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(returnToken, call, endOfText);
    }

    @Override
    public double defaultTemperature() {
        return 1.0;
    }

    @Override
    public double defaultTopP() {
        return 1.0;
    }
}
