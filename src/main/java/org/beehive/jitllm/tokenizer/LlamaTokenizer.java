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
 * GPT-2-style BPE tokenizer (even though it's called "llama") with an explicit merges list.
 *
 * <p>BPE (Byte Pair Encoding): A sub-word tokenization algorithm that iteratively merges the most
 * frequent pairs of symbols in a corpus to build a vocabulary of common character sequences.
 *
 * <p>GPT-2-style tokenization: Applies BPE at the byte level, ensuring all UTF-8 inputs are
 * representable and using tokens that preserve leading spaces (e.g., 'Ġthe').
 *
 * <p>Explicit merges list: A fixed sequence of learned merge rules that deterministically
 * reconstructs the tokenizer’s vocabulary during inference without retraining.
 *
 * <p>Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows
 * along the <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2
 * tokenizer</a>
 */
public class LlamaTokenizer implements Tokenizer {
    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER =
            BYTE_ENCODER.entrySet().stream()
                    .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    private static final String LLAMA_3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // general fields
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    // model-specific fields
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    public LlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // load from metadata
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges =
                Arrays.stream(mergeLines)
                        .map(line -> line.split(" "))
                        .map(
                                parts ->
                                        new Pair<>(
                                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                                vocabulary.getIndex(parts[1]).orElseThrow()))
                        .toList();
        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList =
                Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(
                                Collectors.toMap(
                                        i -> specialTokensList.get(i), i -> baseTokens + i));

        // init tokenizer object fields
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(LLAMA_3_PATTERN);
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex =
                    vocabulary
                            .getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex))
                            .orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first())
                    && i < ids.size() - 1
                    && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings. The reversible bpe
     * codes work on unicode strings. This means you need a large # of unicode characters in your
     * vocab if you want to avoid UNKs. When you're at something like a 10B token dataset you end up
     * needing around 5K for decent coverage. This is a significant percentage of your normal, say,
     * 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size()).boxed().collect(Collectors.toMap(bs::get, cs::get));
    }

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
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

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens if
     * none_raise, then an error is raised if any special token is encountered in text this is the
     * default tiktoken behavior right now as well any other behavior is either annoying, or a major
     * footgun.
     */
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern =
                special.stream().map(Pattern::quote).collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    /** Encoding that ignores any special tokens. */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0.255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair =
                    stats.keySet().stream()
                            .min(
                                    Comparator.comparingInt(
                                            key ->
                                                    this.merges.getOrDefault(
                                                            key, Integer.MAX_VALUE)))
                            .orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

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
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return Arrays.stream(encodeImpl(sb.toString())).boxed().toList();
    }

    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}
