package org.beehive.jitllm.inference;

import java.util.function.IntConsumer;

/**
 * A token consumer that can also ask the generation loop to stop. The loops in {@link
 * TokenGenerationLoop} check it after each token they deliver, and leave the loop the same way a
 * stop token does, so the caller's end-of-turn bookkeeping runs as usual.
 */
public interface CancellableTokenConsumer extends IntConsumer {

    /** Whether generation should stop after the token just delivered. */
    boolean cancellationRequested();
}
