package org.beehive.jllm.inference;

import org.beehive.jllm.inference.state.State;

/**
 * Where a generation loop reads its continuation seed and records the token it produced.
 *
 * <p>Deliberately <b>not</b> routed through the compiled program's invocation boundary. A cursor is
 * not a per-kernel input and belongs to neither the program nor the binding domain.
 */
public interface GenerationCursor {

    /** The token a continued conversation resumes from; the family's initial seed for a new one. */
    int seed();

    /** Records the token just produced. */
    void advance(int token);

    /**
     * The legacy adapter: the cursor is the {@code State}'s own field, exactly as before.
     *
     * <p>Every path but the lowered one uses this, so their behaviour is unchanged — including the
     * families whose initial seed is not repeated in the prompt, which depend on the seed being
     * whatever {@code createNewState} put there.
     */
    static GenerationCursor forState(State state) {
        return new GenerationCursor() {
            @Override
            public int seed() {
                return state.latestToken;
            }

            @Override
            public void advance(int token) {
                state.latestToken = token;
            }
        };
    }
}
