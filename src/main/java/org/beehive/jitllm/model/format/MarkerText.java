package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Encodes text a chat template writes itself — tool instructions, call and result wrappers — whose
 * markers ({@code <tool_call>}, {@code <tool_response>}, …) are single tokens in the vocabulary.
 *
 * <p>llama.cpp and the reference tokenizers read a rendered prompt with special-token parsing on,
 * so a marker the template spells is its token, not its characters. Only format-owned text goes
 * through here: text a caller supplies is encoded as ordinary text, so it cannot turn into a
 * marker.
 */
final class MarkerText {

    private MarkerText() {}

    /**
     * @param text template text
     * @param markers marker spelling to token id; spellings absent from the map stay text
     * @param ordinary the tokenizer's ordinary-text encoding
     */
    static List<Integer> encode(
            String text, Map<String, Integer> markers, Function<String, List<Integer>> ordinary) {
        List<Integer> tokens = new ArrayList<>();
        int from = 0;
        while (from < text.length()) {
            int next = -1;
            String found = null;
            for (String spelling : markers.keySet()) {
                int at = text.indexOf(spelling, from);
                if (at != -1
                        && (next == -1
                                || at < next
                                || (at == next && spelling.length() > found.length()))) {
                    next = at;
                    found = spelling;
                }
            }
            if (found == null) {
                tokens.addAll(ordinary.apply(text.substring(from)));
                break;
            }
            if (next > from) {
                tokens.addAll(ordinary.apply(text.substring(from, next)));
            }
            tokens.add(markers.get(found));
            from = next + found.length();
        }
        return tokens;
    }
}
