/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.tokenizer;

import java.util.Map;

/**
 * The tokenizer for the {@code qwen35} architecture, declared by {@code tokenizer.ggml.pre =
 * "qwen35"}.
 *
 * <p>Everything but the pre-tokenizer split is Qwen 3's: the same GPT-2 byte encoder, the same BPE
 * merges, the same special-token block ending at {@code <|endoftext|>}, the same {@code <think>}
 * control tokens.
 *
 * <p>The split differs in one respect, and it matters for any script that writes with combining
 * marks. A letter run consumes marks as well as letters, and the punctuation run excludes them:
 *
 * <pre>
 *   qwen2/qwen3   … [^\r\n\p{L}\p{N}]?\p{L}+       …  ?[^\s\p{L}\p{N}]+[\r\n]*      …
 *   qwen35        … [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ …  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* …
 * </pre>
 *
 * <p>Under the Qwen 3 split a combining mark would break its own base letter's run and then be
 * absorbed by the punctuation alternative, producing chunk boundaries the merge table was never
 * trained over — plausible text, subtly wrong token ids. Matches {@code
 * unicode_regex_split_custom_qwen35} in llama.cpp.
 */
public class Qwen35Tokenizer extends Qwen3Tokenizer {

    /** The literal from llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_QWEN35}, in Java escaping. */
    private static final String QWEN35_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
                    + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
                    + "|\\p{N}"
                    + "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    public Qwen35Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        super(metadata, vocabulary, false, QWEN35_PATTERN);
    }
}
