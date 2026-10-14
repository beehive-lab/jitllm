package org.beehive.jitllm.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.regex.Pattern;
import org.beehive.jitllm.auxiliary.Pair;

/**
 * The tokenizer for the LFM2 family, declared by {@code tokenizer.ggml.pre = "lfm2"}.
 *
 * <p>A GPT-2 byte-level BPE, like Qwen's, and decoded the same way, so it extends {@link
 * Qwen3Tokenizer} for decoding, the special-token map and the {@code <think>} handling. Encoding
 * differs in three ways, each taken from llama.cpp, which maps {@code lfm2} onto its {@code LLAMA3}
 * pre-tokenizer:
 *
 * <ul>
 *   <li><b>The split is Llama 3's</b>: case-insensitive contractions and digits in runs of up to
 *       three.
 *   <li><b>Merges are applied by rank</b>, their position in {@code tokenizer.ggml.merges}. The
 *       Qwen tokenizer ranks a merge by the id of the token it produces, which is the same thing
 *       for Qwen's vocabulary and not for this one: LFM2's ids do not follow merge order (id 0 is
 *       ten newlines, the product of the very first merges), so ranking by id would apply merges in
 *       an order the model was never trained on.
 *   <li><b>A chunk that is already a token is emitted whole</b> ({@code ignore_merges} in
 *       llama.cpp), without running the merges at all.
 * </ul>
 *
 * <p>Every token from {@code <|pad|>} to the end of the vocabulary is special: the block opens two
 * tokens before {@code <|endoftext|>}, with {@code <|pad|>} and {@code <|startoftext|>}.
 */
public class Lfm2Tokenizer extends Qwen3Tokenizer {

    /** The literal from llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_LLAMA3}, in Java escaping. */
    private static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
                    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private final Pattern split = Pattern.compile(LLAMA3_PATTERN);
    private final Vocabulary vocabulary;

    /** Merge rank by pair: its index in {@code tokenizer.ggml.merges}, lowest applied first. */
    private final Map<Pair<Integer, Integer>, Integer> ranks;

    /** The token a pair merges into. */
    private final Map<Pair<Integer, Integer>, Integer> merged;

    public Lfm2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        super(metadata, vocabulary, "<|pad|>", LLAMA3_PATTERN);
        this.vocabulary = vocabulary;
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        this.ranks = new HashMap<>(mergeLines.length * 2);
        this.merged = new HashMap<>(mergeLines.length * 2);
        for (int rank = 0; rank < mergeLines.length; rank++) {
            String[] parts = mergeLines[rank].split(" ");
            int first = vocabulary.getIndex(parts[0]).orElseThrow();
            int second = vocabulary.getIndex(parts[1]).orElseThrow();
            Pair<Integer, Integer> pair = new Pair<>(first, second);
            // A pair listed twice keeps its first, lowest rank.
            if (ranks.putIfAbsent(pair, rank) == null) {
                merged.put(pair, vocabulary.getIndex(parts[0] + parts[1]).orElseThrow());
            }
        }
    }

    @Override
    public List<Integer> encodeOrdinary(String text) {
        List<Integer> ids = new ArrayList<>();
        for (String chunk : findAll(split, text)) {
            ids.addAll(encodeChunk(byteLevel(chunk)));
        }
        return ids;
    }

    private List<Integer> encodeChunk(String chunk) {
        OptionalInt whole = vocabulary.getIndex(chunk);
        if (whole.isPresent()) {
            return List.of(whole.getAsInt());
        }
        List<Integer> ids = new ArrayList<>(chunk.length());
        for (char c : chunk.toCharArray()) {
            ids.add(vocabulary.getIndex(String.valueOf(c)).orElseThrow());
        }
        while (ids.size() >= 2) {
            Pair<Integer, Integer> best = null;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < ids.size(); i++) {
                Pair<Integer, Integer> pair = new Pair<>(ids.get(i), ids.get(i + 1));
                Integer rank = ranks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    best = pair;
                }
            }
            if (best == null) {
                break;
            }
            ids = merge(ids, best, merged.get(best));
        }
        return ids;
    }
}
