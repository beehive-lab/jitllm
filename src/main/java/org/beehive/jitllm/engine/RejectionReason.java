package org.beehive.jllm.engine;

/**
 * Why a request was refused before it ever ran.
 *
 * <p>Rejection is only for requests that can <b>never</b> run, plus the one capacity case that is a
 * policy rather than a shortfall: a full queue. Ordinary capacity shortfall is <b>not</b> rejection
 * — the request queues, because waiting for capacity is a normal state.
 */
public enum RejectionReason {

    /**
     * The queue is at its bound. The engine never blocks in {@code addRequest}, so backpressure is
     * a rejection the caller can see and retry on, not a hidden wait.
     */
    QUEUE_FULL,

    /**
     * The declared budget exceeds what a slot can ever hold. No amount of waiting helps, so this is
     * terminal and immediate rather than queued.
     */
    CANNOT_EVER_FIT,

    /** The request is not well formed — a non-positive budget, or no tokens to run. */
    MALFORMED,

    /** The engine is shutting down and is not accepting work. */
    SHUTDOWN
}
