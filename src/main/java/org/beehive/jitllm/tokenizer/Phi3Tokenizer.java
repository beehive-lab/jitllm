package org.beehive.jitllm.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.beehive.jitllm.auxiliary.Pair;

public class Phi3Tokenizer implements Tokenizer {

    private static final String SPM_UNDERSCORE = "\u2581";
    private static final String PHI3_PATTERN = "\\S+|\\s+"; // Define appropriate pattern for Phi3
    private static final Pattern P_UTF8_BYTE = Pattern.compile("<0x([0-9A-F]{2})>");

    /** Special token "&lt;s&gt;" */
    private static String TOKEN_BOS = "<s>";

    /** id of token "&lt;s&gt;" */
    private static int TOKEN_BOS_ID = 1;

    private final Vocabulary vocabulary;
    // general fields
    private final Pattern compiledPattern;
    // model-specific fields
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    /** Number of base tokens in the vocabulary; tokens after this index are considered special. */
    private static final int BASE_TOKENS = 32000;

    public Phi3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Pair<Integer, Integer>> merges = Collections.emptyList();

        int allTokens = vocabulary.size();
        int baseTokens = BASE_TOKENS; // assume all tokens after the base ones are special.
        // int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList =
                Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        // @formatter:off
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(
                                Collectors.toMap(
                                        i -> specialTokensList.get(i), i -> baseTokens + i));
        // @formatter:on
        specialTokens.put(TOKEN_BOS, TOKEN_BOS_ID);
        this.specialTokens = specialTokens;
        this.vocabulary = vocabulary;
        this.tokenType = tokenTypes != null ? tokenTypes : new int[vocabulary.size()];
        this.compiledPattern = Pattern.compile(PHI3_PATTERN);
        this.byte0 = 0xE7; // Default byte for special characters, can be adjusted if needed.
    }

    @Override
    public String regexPattern() {
        return "";
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token);
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        return encodeAsList(text);
    }

    @Override
    public List<Integer> encodeAsList(String pText) {
        String text = pText.replace(" ", SPM_UNDERSCORE);
        text = text.startsWith(SPM_UNDERSCORE) ? text : SPM_UNDERSCORE + text;
        final int textLen = text.length();

        final List<Integer> tokens = new ArrayList<>();
        final int vocSize = vocabulary.size();
        int offset = 0;
        while (offset < textLen) {
            String curVoc = null;
            int token = -1;
            for (int j = 0; j < vocSize; j++) {
                final String voc = vocabulary.get(j);
                if (text.startsWith(voc, offset)
                        && (curVoc == null || curVoc.length() < voc.length())) {
                    curVoc = voc;
                    token = j;
                }
            }
            if (curVoc == null) {
                // Try <0xE7>. of character or surrogate (emoji).
                final int len =
                        (offset + 1 < textLen) && Character.isHighSurrogate(text.charAt(offset))
                                ? 2
                                : 1;
                final byte[] bufUtf8 =
                        text.substring(offset, offset + len).getBytes(StandardCharsets.UTF_8);
                for (int i = 0; i < bufUtf8.length; i++) {
                    final String sHex = String.format("<0x%02x>", bufUtf8[i] & 0xff);
                    token = -1;
                    for (int j = 0; j < vocSize; j++) {
                        if (sHex.equalsIgnoreCase(vocabulary.get(j))) {
                            token = j;
                        }
                    }
                    if (token == -1) {
                        throw new RuntimeException(
                                String.format(
                                        "Can't tokenize text at offset %d (%c / (%d, sHex %s)), tokens = %s, text: %s",
                                        offset, text.charAt(offset), i, sHex, tokens, text));
                    }
                    tokens.add(token);
                }
                offset += len;
                continue;
            }
            tokens.add(token);
            offset += curVoc.length();
        }
        return tokens;
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        for (Integer token : tokens) {
            String tokenString = vocabulary.get(token);

            // Check if this is a UTF-8 byte token like <0x0A>
            Matcher matcher = P_UTF8_BYTE.matcher(tokenString);
            if (matcher.matches()) {
                // Extract the hex value and add it to the byte buffer
                int byteValue = Integer.parseInt(matcher.group(1), 16);
                baos.write(byteValue);
            } else {
                // If we have accumulated bytes, convert them to string first
                if (baos.size() > 0) {
                    sb.append(new String(baos.toByteArray(), StandardCharsets.UTF_8));
                    baos.reset();
                }
                // Append the regular token
                sb.append(tokenString);
            }
        }

        // Don't forget any remaining bytes at the end
        if (baos.size() > 0) {
            sb.append(new String(baos.toByteArray(), StandardCharsets.UTF_8));
        }

        // Replace SPM underscore with space
        return sb.toString().replace(SPM_UNDERSCORE, " ");
    }
}
