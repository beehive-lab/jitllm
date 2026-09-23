package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;

/**
 * Chat format for Gemma 4 models.
 *
 * <p>Gemma 4 uses a {@code <|turn>{role}\n. <turn|>} turn structure (the assistant role is spelled
 * "model" in the template), starts conversations with {@code <bos>}, and stops generation on {@code
 * <turn|>} (the model's configured EOS token).
 */
public class Gemma4ChatFormat implements ChatFormat {

    protected final Gemma4Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int startTurn;
    protected final int endTurn;

    public Gemma4ChatFormat(Gemma4Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.getOrDefault("<bos>", -1);
        this.startTurn = specialTokens.getOrDefault("<|turn>", -1);
        this.endTurn = specialTokens.getOrDefault("<turn|>", -1);
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startTurn);
        // The chat template spells the assistant role "model".
        String role = Role.ASSISTANT.equals(message.role()) ? "model" : message.role().name();
        tokens.addAll(tokenizer.encodeAsList(role));
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endTurn);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(endTurn);
    }
}
