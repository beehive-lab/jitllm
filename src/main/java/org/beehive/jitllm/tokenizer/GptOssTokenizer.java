package org.beehive.jitllm.tokenizer;

import java.util.Map;

/**
 * The {@code o200k_harmony} tokenizer of gpt-oss, declared by {@code tokenizer.ggml.pre =
 * "gpt-4o"}.
 *
 * <p>Byte-level BPE with the GPT-2 byte encoder and the file's merges, as Qwen 3's is. What differs
 * is the pre-tokenizer split: letter runs are split by case (an optional run of upper-case letters
 * followed by lower-case ones, or the reverse), English contractions attach to the word before
 * them, and digits group in threes. The pattern is the original from {@code tokenizer.json}, which
 * llama.cpp carries for {@code LLAMA_VOCAB_PRE_TYPE_GPT4O}.
 *
 * <p>Every token from {@code <|endoftext|>} on is special: the Harmony control tokens ({@code
 * <|start|>}, {@code <|channel|>}, {@code <|message|>}, {@code <|end|>}, {@code <|return|>}, {@code
 * <|call|>}, {@code <|constrain|>}) and the reserved block.
 */
public class GptOssTokenizer extends Qwen3Tokenizer {

    private static final String GPT4O_PATTERN =
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
                    + "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                    + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
                    + "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    public GptOssTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        super(metadata, vocabulary, false, GPT4O_PATTERN);
    }

    /** The id of a special token by its text, such as {@code <|start|>}. */
    public int special(String token) {
        Integer id = getSpecialTokens().get(token);
        if (id == null) {
            throw new IllegalStateException("o200k_harmony vocabulary has no " + token);
        }
        return id;
    }
}
