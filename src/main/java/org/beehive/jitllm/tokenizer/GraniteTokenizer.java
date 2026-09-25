package org.beehive.jitllm.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.beehive.jitllm.auxiliary.Pair;

/**
 * GPT-2-style BPE tokenizer for Granite models.
 *
 * <p>Supports both Granite 3.3 (refact pretokenizer, 49K vocab) and Granite 4.0 (dbrx pretokenizer,
 * 100K vocab).
 */
public class GraniteTokenizer implements Tokenizer {
    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER =
            BYTE_ENCODER.entrySet().stream()
                    .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    // Pretokenizer patterns
    private static final String REFACT_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static final String DBRX_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    // Instance fields
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    // Token IDs (version-dependent)
    private final int bosTokenId;
    private final int eosTokenId;
    private final int padTokenId;
    private final String pretokenizerType;

    /** The tool-call markers, displayed although they are special tokens. */
    private final Set<Integer> toolCallMarkers;

    public GraniteTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        this.vocabulary = vocabulary;

        // Detect pretokenizer type and select pattern
        this.pretokenizerType = (String) metadata.getOrDefault("tokenizer.ggml.pre", "refact");
        String pattern =
                switch (pretokenizerType) {
                    case "dbrx" -> DBRX_PATTERN;
                    default -> REFACT_PATTERN;
                };
        this.compiledPattern = Pattern.compile(pattern);

        // Read token IDs from metadata
        this.bosTokenId = getIntFromMetadata(metadata, "tokenizer.ggml.bos_token_id", 0);
        this.eosTokenId = getIntFromMetadata(metadata, "tokenizer.ggml.eos_token_id", 0);
        this.padTokenId = getIntFromMetadata(metadata, "tokenizer.ggml.padding_token_id", 0);

        // Load merges
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> mergeList =
                Arrays.stream(mergeLines)
                        .map(line -> line.split(" "))
                        .map(
                                parts ->
                                        new Pair<>(
                                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                                vocabulary.getIndex(parts[1]).orElseThrow()))
                        .toList();

        // Collect special tokens
        Map<String, Integer> specialTokens = new HashMap<>();
        int allTokens = vocabulary.size();
        for (int i = 0; i < allTokens; i++) {
            String token = vocabulary.get(i);
            if (token.startsWith("<|") && token.endsWith("|>")) {
                specialTokens.put(token, i);
            }
            // Also catch <fim_*> style tokens used in some Granite models
            if (token.startsWith("<") && token.endsWith(">") && !token.contains(" ")) {
                specialTokens.putIfAbsent(token, i);
            }
        }
        this.specialTokens = Map.copyOf(specialTokens);
        this.toolCallMarkers =
                java.util.stream.Stream.of("<|tool_call|>", "<tool_call>", "</tool_call>")
                        .map(this.specialTokens::get)
                        .filter(java.util.Objects::nonNull)
                        .collect(Collectors.toUnmodifiableSet());

        // Build merge map
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : mergeList) {
            String merged = vocabulary.get(pair.first()) + vocabulary.get(pair.second());
            int mergeIndex = vocabulary.getIndex(merged).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private static int getIntFromMetadata(
            Map<String, Object> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number num) {
            return num.intValue();
        }
        return defaultValue;
    }

    // === Token ID accessors ===
    private static List<String> findAll(Pattern pattern, String text) {
        List<String> matches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            matches.add(matcher.group());
        }
        return matches;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newIds = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (i < ids.size() - 1
                    && ids.get(i).equals(pair.first())
                    && ids.get(i + 1).equals(pair.second())) {
                newIds.add(idx);
                i += 2;
            } else {
                newIds.add(ids.get(i));
                i++;
            }
        }
        return newIds;
    }

    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        return IntStream.range(0, bs.size()).boxed().collect(Collectors.toMap(bs::get, cs::get));
    }

    public int getBosTokenId() {
        return bosTokenId;
    }

    // === Tokenizer interface ===

    public int getEosTokenId() {
        return eosTokenId;
    }

    public int getPadTokenId() {
        return padTokenId;
    }

    public String getPretokenizerType() {
        return pretokenizerType;
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    // === Encoding ===

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    /**
     * Ordinary tokens, and the tool-call markers ({@code <|tool_call|>} for Granite 3.2, {@code
     * <tool_call>} / {@code </tool_call>} for Granite 4): a call is read back from the response
     * text, so the markers have to reach it.
     */
    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token) || toolCallMarkers.contains(token);
    }

    public String regexPattern() {
        return compiledPattern != null ? compiledPattern.pattern() : null;
    }

    //    @Override
    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    // === Decoding ===

    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        if (allowedSpecial.isEmpty()) {
            return encodeOrdinary(text);
        }

        assert specialTokens.keySet().containsAll(allowedSpecial);
        String specialPattern =
                allowedSpecial.stream()
                        .map(Pattern::quote)
                        .collect(Collectors.joining("|", "(", ")"));
        String[] specialChunks = text.split(specialPattern);

        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (allowedSpecial.contains(part)) {
                ids.add(specialTokens.get(part));
            } else {
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    public List<Integer> encodeOrdinary(String text) {
        List<String> textChunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            ids.addAll(encodeChunk(chunk));
        }
        return ids;
    }

    // === Helpers ===

    private List<Integer> encodeChunk(String chunk) {
        List<Integer> ids = new ArrayList<>();
        for (char c : chunk.toCharArray()) {
            int tokenIndex = vocabulary.getIndex(String.valueOf(c)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair =
                    stats.keySet().stream()
                            .min(
                                    Comparator.comparingInt(
                                            key -> merges.getOrDefault(key, Integer.MAX_VALUE)))
                            .orElseThrow();
            if (!merges.containsKey(pair)) {
                break;
            }
            ids = merge(ids, pair, merges.get(pair));
        }
        return ids;
    }

    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts =
                decoded.codePoints().map(cp -> BYTE_DECODER.getOrDefault(cp, cp)).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decodedBytesAsInts.length; i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }

    private String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            sb.append(vocabulary.get(token));
        }
        return sb.toString();
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.merge(key, 1, Integer::sum);
        }
        return map;
    }
}
