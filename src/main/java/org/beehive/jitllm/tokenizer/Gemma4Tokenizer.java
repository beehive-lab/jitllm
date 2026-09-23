package org.beehive.jllm.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * SentencePiece-style BPE tokenizer with byte fallback, used by Gemma 4 models.
 *
 * <p>Spaces are represented with the SentencePiece marker {@code ▁}, and any codepoint missing from
 * the vocabulary falls back to its individual UTF-8 bytes encoded as {@code <0xXX>} tokens. Pairs
 * are greedily merged according to the highest {@code tokenizer.ggml.scores} value, mirroring
 * {@link MistralTokenizer}.
 */
public class Gemma4Tokenizer implements Tokenizer {

    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public Gemma4Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");

        // Special tokens are anything that isn't a regular sub-word (NORMAL, type 1) or a raw
        // byte-fallback token (BYTE, type 6).
        Map<String, Integer> specialTokens =
                IntStream.range(0, vocabulary.size())
                        .filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6)
                        .boxed()
                        .collect(
                                Collectors.toMap(
                                        vocabulary::get, t -> t, (first, second) -> first));

        this.vocabulary = vocabulary;
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenTypes;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    @Override
    public String regexPattern() {
        return null;
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return getTokenType(tokenIndex) != 1;
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        int type = getTokenType(token);
        return type == 1 || type == 6;
    }

    public int getTokenType(int tokenIndex) {
        return tokenType[tokenIndex];
    }

    private List<Integer> encodeImpl(String text) {
        List<Integer> tokens = new ArrayList<>();

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                tokens.add(id);
            } else {
                // byte fallback: encode each UTF-8 byte as a <0xXX> token (offset by the index of
                // <0x00>)
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }

        // greedily merge the highest-scoring adjacent pair until no more merges apply
        while (true) {
            float bestScore = -1e10f;
            int bestId = -1;
            int bestIdx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                String merged = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(merged).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > bestScore) {
                    bestScore = vocabulary.getScore(id);
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) {
                break;
            }

            tokens.set(bestIdx, bestId);
            tokens.remove(bestIdx + 1);
        }

        return tokens;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text, Collections.emptySet());
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // byte-fallback tokens decode back to their raw byte/codepoint
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6
                        && tokenString.startsWith(prefix)
                        && tokenString.endsWith(suffix)) {
                    String code =
                            tokenString.substring(
                                    prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('▁', ' ');
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }
}
