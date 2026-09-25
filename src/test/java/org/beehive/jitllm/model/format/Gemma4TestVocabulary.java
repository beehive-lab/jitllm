package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * A synthetic Gemma 4 vocabulary for deterministic format tests: the special tokens the chat
 * template uses, with the token types the real GGUF gives them (3 control, 4 user-defined), every
 * printable ASCII character as its own token, and the 256 byte-fallback tokens.
 *
 * <p>There are no multi-character ordinary tokens, so nothing merges and every piece of text
 * decodes back to exactly what was encoded — which is what lets a test compare an encoded
 * conversation, decoded, against the reference template's rendering.
 */
public final class Gemma4TestVocabulary {

    private Gemma4TestVocabulary() {}

    /** {@code <eos>} is not here: the Gemma 4 GGUFs type it as an ordinary token (type 1). */
    private static final List<String> CONTROL =
            List.of("<pad>", "<bos>", "<|turn>", "<turn|>", "<|tool>", "<tool|>");

    private static final List<String> USER_DEFINED =
            List.of(
                    "<|tool_call>",
                    "<tool_call|>",
                    "<|tool_response>",
                    "<tool_response|>",
                    "<|\"|>",
                    "<|channel>",
                    "<channel|>");

    /** The full vocabulary, tool markers included. */
    public static Gemma4Tokenizer tokenizer() {
        return tokenizer(true);
    }

    /**
     * @param withToolMarkers whether the tool-calling markers are in the vocabulary; without them a
     *     file was not trained for tool calling
     */
    public static Gemma4Tokenizer tokenizer(boolean withToolMarkers) {
        List<String> tokens = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        for (String control : CONTROL) {
            if (!withToolMarkers && control.contains("tool")) {
                continue;
            }
            tokens.add(control);
            types.add(3);
        }
        for (String userDefined : USER_DEFINED) {
            if (!withToolMarkers && !userDefined.contains("channel")) {
                continue;
            }
            tokens.add(userDefined);
            types.add(4);
        }
        tokens.add("<eos>");
        types.add(1);
        for (int b = 0; b < 256; b++) {
            tokens.add(String.format("<0x%02X>", b));
            types.add(6);
        }
        tokens.add("▁");
        types.add(1);
        tokens.add("\n");
        types.add(1);
        for (char c = 0x21; c < 0x7f; c++) {
            tokens.add(String.valueOf(c));
            types.add(1);
        }
        int[] tokenTypes = types.stream().mapToInt(Integer::intValue).toArray();
        Vocabulary vocabulary =
                new Vocabulary(tokens.toArray(String[]::new), new float[tokens.size()]);
        return new Gemma4Tokenizer(Map.of("tokenizer.ggml.token_type", tokenTypes), vocabulary);
    }

    /** The chat format over {@link #tokenizer()}. */
    public static Gemma4ChatFormat chatFormat() {
        return new Gemma4ChatFormat(tokenizer());
    }
}
