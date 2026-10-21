package org.beehive.jitllm.tokenizer;

import java.util.Map;

/**
 * The byte-level BPE of GLM-4 vocabularies ({@code tokenizer.ggml.pre = glm4}).
 *
 * <p>Qwen3's tokenizer with llama.cpp's {@code CHATGLM4} split — digits in runs of up to three —
 * and with merges skipped for a chunk the vocabulary already holds whole.
 */
public class Glm4Tokenizer extends Qwen3Tokenizer {

    private static final String GLM4_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
                    + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    public Glm4Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        super(metadata, vocabulary, false, GLM4_PATTERN);
    }

    @Override
    protected boolean ignoreMerges() {
        return true;
    }
}
