package org.beehive.jitllm.inference;

import java.util.List;
import org.beehive.jitllm.inference.state.State;

/**
 * Where a generation loop starts feeding: the first token, and the prompt index to continue from.
 *
 * <h2>The duplicate this exists to remove</h2>
 *
 * <p>Every generation loop used to begin {@code currentToken = state.latestToken} and then ingest
 * the prompt from index 0. For families whose state is seeded with the beginning-of-text token
 * <b>and</b> whose prompt already opens with it — Llama, Mistral, Devstral, Gemma4 — that fed the
 * model <b>BOS twice</b> before a single real token. Every answer this project has produced on
 * those families was conditioned on the doubled token.
 *
 * <p>It could not be fixed by dropping the seed. Qwen2, Qwen3, DeepSeek and Phi3 override {@code
 * shouldAddBeginOfText()} and seed {@code latestToken} with a <i>start-header</i> token their
 * prompts do not repeat, so for them the seed is a real first token and removing it would corrupt
 * the prompt. The duplication is specific, so the fix is specific: skip the seed only when it is
 * literally the token the prompt already starts with, and only at the start of a conversation.
 *
 * <h2>Why the start position matters</h2>
 *
 * <p>In a continued conversation {@code state.latestToken} is the last token generated, and the
 * prompt is the new turn — which does not repeat it. Feeding the seed there is correct, and a
 * coincidental equality between the last generated token and the new turn's first token must not be
 * mistaken for the duplication above. So the check is gated on {@code startPosition == 0}: at
 * position zero the prompt is the entire context, and nothing precedes it.
 */
public record PromptIngestion(int firstToken, int firstIndex) {

    /**
     * @param state holds the seed token the loop would otherwise start from
     * @param promptTokens the prompt about to be ingested
     * @param startPosition 0 for a new conversation; the continuation point otherwise
     */
    public static PromptIngestion of(State state, List<Integer> promptTokens, int startPosition) {
        return of(state.latestToken, promptTokens, startPosition);
    }

    /**
     * As above, but taking the seed explicitly rather than reading it from a {@code State}.
     *
     * <p>The seed is <b>session history</b>: in a workspace shared between sessions, reading it
     * from the state means continuing whoever went last. Taking it as a parameter is what lets a
     * session carry its own.
     *
     * @param seed the token a continued conversation resumes from
     */
    public static PromptIngestion of(int seed, List<Integer> promptTokens, int startPosition) {
        if (startPosition == 0 && !promptTokens.isEmpty() && promptTokens.get(0) == seed) {
            // The prompt already opens with the token the state was seeded with. Feed it once.
            return new PromptIngestion(seed, 1);
        }
        return new PromptIngestion(seed, 0);
    }
}
