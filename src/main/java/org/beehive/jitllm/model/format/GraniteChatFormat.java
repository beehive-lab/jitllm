package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jllm.tokenizer.GraniteTokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

/**
 * Chat format for Granite models.
 *
 * <p>Granite uses a different chat template than Llama:
 * <|start_of_role|>system<|end_of_role|>.<|end_of_text|>
 * <|start_of_role|>user<|end_of_role|>.<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>.
 */
public class GraniteChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int startRole;
    protected final int endRole;
    protected final int endOfText;
    protected final Set<Integer> stopTokens;

    public GraniteChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        this.startRole = specialTokens.getOrDefault("<|start_of_role|>", -1);
        this.endRole = specialTokens.getOrDefault("<|end_of_role|>", -1);

        // Use tokenizer's EOS token instead of hardcoding
        if (tokenizer instanceof GraniteTokenizer graniteTokenizer) {
            this.endOfText = graniteTokenizer.getEosTokenId();
        } else {
            this.endOfText = specialTokens.getOrDefault("<|end_of_text|>", 0);
        }

        this.stopTokens = Set.of(endOfText);
    }

    @Override
    public int getBeginOfText() {
        return endOfText; // For Granite, token 0 is both BOS and EOS
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (startRole >= 0) {
            tokens.add(startRole);
        }
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        if (endRole >= 0) {
            tokens.add(endRole);
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfText);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }
}
