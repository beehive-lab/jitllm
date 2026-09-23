package org.beehive.jllm.api;

import org.beehive.jllm.inference.GenerationCursor;

/**
 * The generation cursor a session owns, and everything else that varies independently by session.
 *
 * <p>Session history, not device state. The seed is read at the entry to generation and the value
 * is updated per token — both <b>outside</b> the per-invocation lock — so it cannot live in a
 * workspace shared between sessions without one continuing another's conversation.
 *
 * <p>The full call-graph audit of the lowered Llama FP16 single-token path found <b>exactly
 * these</b> non-device fields carrying session history: the cursor. Everything else in {@code
 * State} is either a device array, fixed at construction ({@code kvSlot}, the block configuration),
 * or host storage the lowered path does not touch.
 */
final class SessionLogicalState implements GenerationCursor {

    /**
     * The family's initial seed — for Llama, {@code <|begin_of_text|>}.
     *
     * <p>Not {@code -1}. A reset session must ingest its prompt exactly as a new one does, and that
     * depends on starting from the same token {@code createNewState} would have put in a fresh
     * state.
     */
    private final int initialToken;

    private int latestToken;

    SessionLogicalState(int initialToken) {
        this.initialToken = initialToken;
        this.latestToken = initialToken;
    }

    @Override
    public int seed() {
        return latestToken;
    }

    @Override
    public void advance(int token) {
        latestToken = token;
    }

    /** Back to the family's initial seed, so the next turn ingests as a new session would. */
    void reset() {
        latestToken = initialToken;
    }

    int initialToken() {
        return initialToken;
    }
}
