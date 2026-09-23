package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jllm.tokenizer.Tokenizer;

/**
 * Chat format implementation specifically designed for the Phi3 model. This class handles the
 * specific prompt formatting and token management required for Phi3's conversational interface.
 *
 * <p>Phi3 uses a simpler chat format compared to other models:
 *
 * <ul>
 *   <li>Role tokens: {@code <|system|>}, {@code <|user|>}, {@code <|assistant|>}
 *   <li>End token: {@code <|end|>}
 *   <li>No separate begin-of-text or header/content separation
 * </ul>
 */
public class Phi3ChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final int endOfTextFim;
    protected final int end; // End token for the chat format
    protected ChatTokens chatTokens;

    public Phi3ChatFormat(Tokenizer tokenizer, ChatTokens chatTokens) {
        this.tokenizer = tokenizer;
        this.chatTokens = chatTokens;
        this.beginOfText = tokenizer.getSpecialTokens().getOrDefault("", -1);
        this.startHeader = tokenizer.getSpecialTokens().getOrDefault("", -1);
        this.endHeader = tokenizer.getSpecialTokens().getOrDefault("<|end|>", -1);
        this.endOfTurn = tokenizer.getSpecialTokens().getOrDefault("", -1);
        this.endOfText = tokenizer.getSpecialTokens().getOrDefault("", -1);
        this.endOfTextFim = tokenizer.getSpecialTokens().getOrDefault("", -1);
        this.endOfMessage = tokenizer.getSpecialTokens().getOrDefault("", -1);
        // Initialize end token
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.end = specialTokens.get("<|end|>");
    }

    public ChatTokens chatTokens() {
        return chatTokens;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return Set.of(end);
    }

    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        String tokenRole = "<|" + message.role().name() + "|>";
        final Integer idxSpecial = tokenizer.getSpecialTokens().get(tokenRole);
        if (idxSpecial != null) {
            tokens.add(idxSpecial);
        } else {
            tokens.addAll(this.tokenizer.encodeAsList(tokenRole));
        }
        return tokens;
    }

    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(tokenizer.getSpecialTokens().get("<|end|>"));
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        throw new UnsupportedOperationException("Phi3 does not use a begin-of-text token.");
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        for (Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new Message(Role.ASSISTANT, "")));
        }
        return tokens;
    }
}
