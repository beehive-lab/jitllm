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

import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

public interface Tokenizer {
    // Utility method for all tokenizers, implemented as static.
    static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    // Utility method for all tokenizers, implemented as static.
    static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    String regexPattern();

    Map<String, Integer> getSpecialTokens();

    boolean isSpecialToken(int tokenIndex);

    /**
     * Determines if a token should be displayed during streaming output. This filters out special
     * tokens, control characters, or other non-displayable content.
     *
     * @param token the token to check
     * @return true if the token should be displayed to the user, false otherwise
     */
    boolean shouldDisplayToken(int token);

    List<Integer> encode(String text, Set<String> allowedSpecial);

    List<Integer> encodeAsList(String text);

    String decode(List<Integer> tokens);
}
