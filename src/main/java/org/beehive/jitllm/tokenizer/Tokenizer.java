package org.beehive.jllm.tokenizer;

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
