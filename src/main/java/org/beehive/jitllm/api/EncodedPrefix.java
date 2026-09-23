package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;

/**
 * The tokens a session has already fed to the model, and what of a new input it can skip.
 *
 * <p>The unit of prefix reuse [A1]. It holds <b>encoded tokens</b>, not messages: attaching a tool
 * changes the encoded system content, so an input diverges from its first tokens while the message
 * list looks unchanged. A message-level comparison would reuse a prefix that no longer exists and
 * produce an answer conditioned on the wrong context — silently, and only sometimes.
 *
 * <p>Its contents track the session's position exactly, because position is what the key/value
 * cache holds.
 */
final class EncodedPrefix {

    private final List<Integer> tokens = new ArrayList<>();

    /** What is retained. Package-private for tests and for the session's accounting. */
    int size() {
        return tokens.size();
    }

    /**
     * Whether anything is retained at all.
     *
     * <p>Distinct from divergence, and the distinction matters: a fresh session has nothing to
     * diverge <i>from</i>, so it must not be reset. Resetting one anyway was a defect — it
     * discarded the start token the model seeded the state with, and the first token fed was then
     * the "not yet set" sentinel.
     */
    boolean isEmpty() {
        return tokens.isEmpty();
    }

    /** Records tokens the model has now consumed — prompt tokens, then generated tokens. */
    void append(List<Integer> consumed) {
        tokens.addAll(consumed);
    }

    void clear() {
        tokens.clear();
    }

    /** Forgets everything from {@code size} on: the cache is rewound to that position. */
    void truncate(int size) {
        tokens.subList(size, tokens.size()).clear();
    }

    /** The retained tokens, in order. */
    List<Integer> tokens() {
        return List.copyOf(tokens);
    }

    /**
     * What of {@code full} still has to be encoded.
     *
     * @param full the complete model input for the request
     * @return the tokens to feed, or {@code null} if the retained prefix diverges — in which case
     *     the caller resets and encodes {@code full} from the start. Divergence is not an error and
     *     not something the caller has to explain; refusing to reuse is the whole of the
     *     correctness argument
     */
    List<Integer> reusableSuffixOf(List<Integer> full) {
        if (tokens.isEmpty()) {
            return null; // nothing retained: not divergence, just no reuse
        }
        if (full.size() < tokens.size()) {
            // Shorter than what is retained, so it cannot extend it. A caller that dropped a turn
            // lands here.
            return null;
        }
        for (int i = 0; i < tokens.size(); i++) {
            if (!tokens.get(i).equals(full.get(i))) {
                return null;
            }
        }
        // An exact prefix. Note the empty case is legitimate: re-sending the identical conversation
        // with nothing new appended has nothing left to encode.
        return new ArrayList<>(full.subList(tokens.size(), full.size()));
    }
}
