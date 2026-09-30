package org.beehive.jitllm.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Synthetic byte-level vocabularies for deterministic chat-format tests.
 *
 * <p>Each has one token per byte of the GPT-2 byte alphabet, the family's special tokens where its
 * tokenizer looks for them, and no merges — so every piece of text decodes back to exactly what was
 * encoded, which lets a test compare an encoded conversation, decoded, against the reference
 * template's rendering.
 */
public final class TestVocabularies {

    private TestVocabularies() {}

    /** Granite 3.2's tool marker. */
    public static final List<String> GRANITE_3_2_MARKERS = List.of("<|tool_call|>");

    /** Granite 4.0's tool markers. */
    public static final List<String> GRANITE_4_MARKERS =
            List.of(
                    "<tool_call>",
                    "</tool_call>",
                    "<tool_response>",
                    "</tool_response>",
                    "<tools>",
                    "</tools>");

    /** Qwen 2.5 / Qwen 3 tool markers. */
    public static final List<String> QWEN_MARKERS =
            List.of("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>");

    /**
     * {@code <|end_of_text|>} at id 0 (the metadata default for BOS and EOS), the role markers,
     * then the given tool markers.
     */
    public static GraniteTokenizer granite(List<String> toolMarkers) {
        List<String> tokens = new ArrayList<>();
        tokens.add("<|end_of_text|>");
        tokens.add("<|start_of_role|>");
        tokens.add("<|end_of_role|>");
        tokens.addAll(toolMarkers);
        tokens.addAll(bytes(GraniteTokenizer.BYTE_ENCODER));
        Vocabulary vocabulary = new Vocabulary(tokens.toArray(String[]::new), null);
        return new GraniteTokenizer(Map.of("tokenizer.ggml.merges", new String[0]), vocabulary);
    }

    /**
     * The bytes, then {@code <|endoftext|>} — where the Qwen tokenizer's special block starts — and
     * the ChatML, tool and think tokens.
     */
    public static Qwen3Tokenizer qwen(List<String> toolMarkers) {
        return new Qwen3Tokenizer(
                Map.of(
                        "tokenizer.ggml.merges",
                        new String[0],
                        "tokenizer.ggml.token_type",
                        qwenTokenTypes(toolMarkers)),
                new Vocabulary(qwenTokens(toolMarkers), null),
                false);
    }

    private static String[] qwenTokens(List<String> toolMarkers) {
        List<String> tokens = new ArrayList<>(bytes(Qwen3Tokenizer.BYTE_ENCODER));
        tokens.addAll(List.of("<|endoftext|>", "<|im_start|>", "<|im_end|>"));
        tokens.addAll(toolMarkers);
        tokens.addAll(List.of("<think>", "</think>"));
        return tokens.toArray(String[]::new);
    }

    /** Ordinary for the bytes, 3 for the ChatML block, 4 for the tool and think markers. */
    private static int[] qwenTokenTypes(List<String> toolMarkers) {
        List<Integer> types = new ArrayList<>();
        bytes(Qwen3Tokenizer.BYTE_ENCODER).forEach(t -> types.add(1));
        List.of("<|endoftext|>", "<|im_start|>", "<|im_end|>").forEach(t -> types.add(3));
        toolMarkers.forEach(t -> types.add(4));
        types.add(4);
        types.add(4);
        return types.stream().mapToInt(Integer::intValue).toArray();
    }

    /** As {@link #qwen(List)}, under the {@code qwen35} pre-tokenizer: Qwen3.8's tokenizer. */
    public static Qwen35Tokenizer qwen35(List<String> toolMarkers) {
        Qwen3Tokenizer qwen = qwen(toolMarkers);
        return new Qwen35Tokenizer(
                Map.of(
                        "tokenizer.ggml.merges",
                        new String[0],
                        "tokenizer.ggml.token_type",
                        qwenTokenTypes(toolMarkers)),
                new Vocabulary(qwenTokens(toolMarkers), null));
    }

    /** Llama's special tokens start at id 128000, so the bytes are padded up to it. */
    public static LlamaTokenizer llama() {
        List<String> tokens = new ArrayList<>(bytes(LlamaTokenizer.BYTE_ENCODER));
        for (int i = tokens.size(); i < 128000; i++) {
            tokens.add("<filler_" + i + ">");
        }
        tokens.addAll(
                Arrays.asList(
                        "<|begin_of_text|>",
                        "<|end_of_text|>",
                        "<|start_header_id|>",
                        "<|end_header_id|>",
                        "<|eom_id|>",
                        "<|eot_id|>",
                        "<|python_tag|>"));
        Vocabulary vocabulary = new Vocabulary(tokens.toArray(String[]::new), null);
        return new LlamaTokenizer(Map.of("tokenizer.ggml.merges", new String[0]), vocabulary);
    }

    private static List<String> bytes(Map<Integer, Integer> byteEncoder) {
        List<String> tokens = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            tokens.add(Character.toString(byteEncoder.get(b)));
        }
        return tokens;
    }
}
