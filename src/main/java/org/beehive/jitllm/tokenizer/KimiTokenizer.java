package org.beehive.jitllm.tokenizer;

import java.util.Map;

/**
 * The tiktoken BPE of Kimi vocabularies ({@code tokenizer.ggml.pre = kimi-k2}).
 *
 * <p>Qwen 3's byte-level BPE, whose merge priority is the merged token's id, which for a vocabulary
 * converted from tiktoken is its rank. Three things differ:
 *
 * <ul>
 *   <li>the split, Kimi's {@code pat_str}: runs of Han characters on their own, letter runs that
 *       exclude Han and take an English contraction suffix in any case, digits in runs of up to
 *       three; whitespace is Unicode whitespace, as in the Rust regex tiktoken runs;
 *   <li>a chunk the vocabulary holds whole is that token, without merges, as tiktoken encodes it;
 *   <li>the special tokens are the block from {@code [BOS]} to the end of the vocabulary.
 * </ul>
 *
 * <p>Matches {@code unicode_regex_split_custom_kimi_k2} in llama.cpp.
 */
public class KimiTokenizer extends Qwen3Tokenizer {

    private static final String LETTERS_UPPER =
            "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]";
    private static final String LETTERS_LOWER = "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]";
    private static final String CONTRACTION = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?";

    /** Kimi's {@code pat_str}, in Java escaping, with Unicode whitespace. */
    static final String KIMI_PATTERN =
            "(?U)[\\p{IsHan}]+"
                    + "|[^\\r\\n\\p{L}\\p{N}]?"
                    + LETTERS_UPPER
                    + "*"
                    + LETTERS_LOWER
                    + "+"
                    + CONTRACTION
                    + "|[^\\r\\n\\p{L}\\p{N}]?"
                    + LETTERS_UPPER
                    + "+"
                    + LETTERS_LOWER
                    + "*"
                    + CONTRACTION
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    public KimiTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        super(metadata, vocabulary, "[BOS]", KIMI_PATTERN);
    }

    @Override
    protected boolean ignoreMerges() {
        return true;
    }
}
