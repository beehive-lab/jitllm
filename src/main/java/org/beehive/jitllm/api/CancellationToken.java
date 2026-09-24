package org.beehive.jitllm.api;

/**
 * Asks a generation to stop early.
 *
 * <p>Attach a token to a request with {@link GenerationRequest.Builder#cancellation}, and call
 * {@link #cancel()} from <b>any thread</b> while {@link GenerationSession#generate} runs — a server
 * whose client disconnected, a UI's stop button, a coroutine being cancelled. Generation stops
 * after the token being produced, the session stays usable, and the result carries the text
 * produced so far with {@link FinishReason#CANCELLED}.
 *
 * <p>A token is one-shot: once cancelled it stays cancelled, and a request built with a cancelled
 * token returns immediately without generating. Use a new token per request.
 */
@Experimental
public final class CancellationToken {

    private volatile boolean cancelled;

    /** Asks the generation using this token to stop. Idempotent; safe from any thread. */
    public void cancel() {
        cancelled = true;
    }

    /** Whether {@link #cancel()} has been called. */
    public boolean isCancelled() {
        return cancelled;
    }
}
