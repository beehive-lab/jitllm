package org.beehive.jllm.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * TikToken-style BPE tokenizer with byte fallback.
 *
 * <p>TikToken-style: A Byte Pair Encoding (BPE) strategy that converts text to UTF-8 bytes.
 * Frequent pairs of bytes (or tokens) are merged according to a learned vocabulary. This reduces
 * long words into common subwords or whole-word tokens. If a word or character isn't found, it
 * falls back to byte-level tokens.
 *
 * <p>Byte fallback: A fail-safe mechanism. It ensures every byte has a token, so any input (even
 * unknown words, misspellings, foreign languages, emojis, or binary) can be tokenized. If a token
 * is not found in the merges or vocabulary, it will fall back to the individual byte. Each byte is
 * wrapped as a special token like <0xF0> — these are part of the tokenizer’s extended vocabulary.
 * This guarantees reversibility: every string can be tokenized and decoded back exactly.
 */
public class MistralTokenizer implements Tokenizer {
    private static final String MISTRAL_PATTERN = "\\S+|\\s+";
    // general fields
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    // model-specific fields
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    // @formatter:off
    public MistralTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // load from metadata
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList =
                IntStream.range(0, vocabulary.size())
                        .filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6)
                        .boxed()
                        .toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(t -> vocabulary.get(t), t -> t));
        // init tokenizer object fields
        this.vocabulary = vocabulary;
        this.compiledPattern = null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenTypes;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
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
        int tokenType = getTokenType(token);
        return tokenType == 1 || tokenType == 6;
    }

    public int getTokenType(int tokenIndex) {
        return tokenType[tokenIndex];
    }

    // @formatter:on

    private List<Integer> encodeImpl(String text) {

        List<Integer> tokens = new ArrayList<>();

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens.add(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +byte0 here to skip all the control and special tokens e.g. <unk>, <s>, </s>
                // so the individual bytes only start at token <0x00>
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer =
                        vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(str_buffer).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocabulary.getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }

    /** Modified original signature from mistral.java: List<Integer> encode(String text); */
    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        // pass an empty set to comply with method signature.
        return encode(text, Collections.emptySet());
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // some tokens designate raw bytes e.g. '<0x10>'
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
