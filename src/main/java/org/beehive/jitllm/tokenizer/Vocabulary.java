/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Portions of this file are derived from Llama3.java
 * (https://github.com/mukel/llama3.java),
 * Copyright (c) 2024 Alfonso² Peterssen and Copyright (c) 2024 Andrej Karpathy.
 * Used under the MIT License; see the LICENSE file.
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
