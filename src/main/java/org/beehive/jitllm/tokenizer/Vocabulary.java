package org.beehive.jllm.tokenizer;

import java.util.Arrays;
import java.util.Map;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {

    // @formatter:off
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(
                vocabulary,
                scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i)));
    }

    // @formatter:on

    /**
     * Tokens only, no scores. GGUF stores {@code tokenizer.ggml.scores} for the tokenizers that
     * need them; the byte-pair ones do not.
     */
    public static Vocabulary fromTokens(Map<String, Object> metadata) {
        return new Vocabulary((String[]) metadata.get("tokenizer.ggml.tokens"), null);
    }

    /** Tokens and their scores, for the tokenizers whose merge order depends on them. */
    public static Vocabulary fromTokensAndScores(Map<String, Object> metadata) {
        return new Vocabulary(
                (String[]) metadata.get("tokenizer.ggml.tokens"),
                (float[]) metadata.get("tokenizer.ggml.scores"));
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }

    /** Only for Mistral. */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Vocabulary:\n");
        sb.append("Tokens: ").append(Arrays.toString(tokens)).append("\n");
        sb.append("Scores: ").append(Arrays.toString(scores)).append("\n");
        sb.append("Token to Index Map:\n");
        tokenToIndex.forEach(
                (token, index) ->
                        sb.append("  ").append(token).append(" -> ").append(index).append("\n"));
        return sb.toString();
    }
}
