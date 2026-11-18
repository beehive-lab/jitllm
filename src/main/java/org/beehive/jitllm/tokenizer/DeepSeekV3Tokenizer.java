package org.beehive.jitllm.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * The byte-level BPE of DeepSeek-V3 and V4 vocabularies (llama.cpp's {@code deepseek-v3} and {@code
 * joyai-llm} pre-tokenizers).
 *
 * <p>The pre-tokenizer is three splits applied one after the other, each keeping both the matches
 * and the text between them: digits in runs of up to three, then runs of CJK ideographs and kana,
 * then the main pattern. Applied in sequence, a CJK run is split from letters around it, which one
 * combined pattern would not do. Control tokens (token type 3, such as {@code
 * <｜begin▁of▁sentence｜>} at id 0) and user-defined ones (type 4) are special; they are encoded only
 * when allowed.
 */
public class DeepSeekV3Tokenizer implements Tokenizer {

    private static final Pattern[] SPLITS = {
        Pattern.compile("\\p{N}{1,3}"),
        Pattern.compile("[\\u4e00-\\u9fa5\\u3040-\\u309f\\u30a0-\\u30ff]+"),
        Pattern.compile(
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
                        + "|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
                        + "| ?[\\p{P}\\p{S}]+[\\r\\n]*"
                        + "|\\s*[\\r\\n]+"
                        + "|\\s+(?!\\S)"
                        + "|\\s+")
    };

    private static final int[] BYTE_ENCODER = new int[256];
    private static final Map<Integer, Integer> BYTE_DECODER = new HashMap<>();

    static {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) {
            bs.add(b);
        }
        for (int b = '¡'; b <= '¬'; b++) {
            bs.add(b);
        }
        for (int b = '®'; b <= 'ÿ'; b++) {
            bs.add(b);
        }
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n++);
            }
        }
        for (int i = 0; i < bs.size(); i++) {
            BYTE_ENCODER[bs.get(i)] = cs.get(i);
            BYTE_DECODER.put(cs.get(i), bs.get(i));
        }
    }

    private final Vocabulary vocabulary;
    private final int[] tokenTypes;
    private final Map<String, Integer> mergeRanks;
    private final Map<String, Integer> specialTokens;

    /** UTF-8 bytes of a character the last {@link #decode} call left incomplete. */
    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();

    public DeepSeekV3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        String[] merges = (String[]) metadata.get("tokenizer.ggml.merges");
        this.mergeRanks = HashMap.newHashMap(merges == null ? 0 : merges.length);
        if (merges != null) {
            for (int i = 0; i < merges.length; i++) {
                mergeRanks.put(merges[i], i);
            }
        }
        this.specialTokens = new HashMap<>();
        for (int i = 0; i < vocabulary.size(); i++) {
            if (tokenTypes != null && (tokenTypes[i] == 3 || tokenTypes[i] == 4)) {
                specialTokens.put(vocabulary.get(i), i);
            }
        }
    }

    @Override
    public String regexPattern() {
        return SPLITS[SPLITS.length - 1].pattern();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return tokenTypes != null && (tokenTypes[tokenIndex] == 3 || tokenTypes[tokenIndex] == 4);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        int type = tokenTypes == null ? 1 : tokenTypes[token];
        return type == 1 || type == 4 || type == 6;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        if (allowedSpecial.isEmpty()) {
            return encodeOrdinary(text);
        }
        String specialPattern =
                allowedSpecial.stream()
                        .map(Pattern::quote)
                        .collect(Collectors.joining("|", "(", ")"));
        List<Integer> ids = new ArrayList<>();
        Matcher m = Pattern.compile(specialPattern).matcher(text);
        int last = 0;
        while (m.find()) {
            ids.addAll(encodeOrdinary(text.substring(last, m.start())));
            ids.add(specialTokens.get(m.group()));
            last = m.end();
        }
        ids.addAll(encodeOrdinary(text.substring(last)));
        return ids;
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text, Set.of());
    }

    /** The text's tokens, special-token text encoded as ordinary text. */
    public List<Integer> encodeOrdinary(String text) {
        List<String> pieces = List.of(text);
        for (Pattern split : SPLITS) {
            pieces = splitIsolated(pieces, split);
        }
        List<Integer> ids = new ArrayList<>();
        for (String piece : pieces) {
            for (String token : bpe(byteLevel(piece))) {
                ids.add(vocabulary.getIndex(token).orElseThrow());
            }
        }
        return ids;
    }

    /** Splits every piece into the pattern's matches and the text between them, keeping both. */
    static List<String> splitIsolated(List<String> pieces, Pattern pattern) {
        List<String> out = new ArrayList<>();
        for (String piece : pieces) {
            Matcher m = pattern.matcher(piece);
            int last = 0;
            while (m.find()) {
                if (m.start() > last) {
                    out.add(piece.substring(last, m.start()));
                }
                out.add(m.group());
                last = m.end();
            }
            if (last < piece.length()) {
                out.add(piece.substring(last));
            }
        }
        return out;
    }

    /** Byte-level BPE of one piece: repeatedly merges the adjacent pair with the lowest rank. */
    private List<String> bpe(String word) {
        List<String> parts = new ArrayList<>();
        word.codePoints().forEach(cp -> parts.add(new String(Character.toChars(cp))));
        while (parts.size() > 1) {
            int best = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < parts.size(); i++) {
                Integer rank = mergeRanks.get(parts.get(i) + " " + parts.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            parts.set(best, parts.get(best) + parts.remove(best + 1));
        }
        return parts;
    }

    private static String byteLevel(String raw) {
        StringBuilder sb = new StringBuilder();
        for (byte b : raw.getBytes(StandardCharsets.UTF_8)) {
            sb.appendCodePoint(BYTE_ENCODER[Byte.toUnsignedInt(b)]);
        }
        return sb.toString();
    }

    /**
     * The tokens as text. A special token is its own text; the others are byte-level strings mapped
     * back to bytes. A character whose bytes are split across calls is held until it is complete.
     */
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder out = new StringBuilder();
        for (int token : tokens) {
            String text = vocabulary.get(token);
            if (isSpecialToken(token)) {
                out.append(flush(false)).append(text);
                continue;
            }
            text.codePoints()
                    .forEach(cp -> pending.write(BYTE_DECODER.getOrDefault(cp, (int) '?')));
        }
        return out.append(flush(true)).toString();
    }

    /**
     * The pending bytes as text; with {@code keepIncomplete}, a trailing partial character stays.
     */
    private String flush(boolean keepIncomplete) {
        byte[] bytes = pending.toByteArray();
        int end = keepIncomplete ? completeLength(bytes) : bytes.length;
        pending.reset();
        pending.write(bytes, end, bytes.length - end);
        return new String(bytes, 0, end, StandardCharsets.UTF_8);
    }

    /** The length of the longest prefix that does not end inside a UTF-8 character. */
    private static int completeLength(byte[] b) {
        int i = b.length;
        int back = 0;
        while (i > 0 && back < 4 && (b[i - 1] & 0xC0) == 0x80) {
            i--;
            back++;
        }
        if (i == 0) {
            return b.length;
        }
        int lead = b[i - 1] & 0xFF;
        int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        return b.length - (i - 1) >= need ? b.length : i - 1;
    }
}
