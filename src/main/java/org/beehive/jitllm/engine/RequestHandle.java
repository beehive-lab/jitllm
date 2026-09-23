package org.beehive.jitllm.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntConsumer;

/**
 * The caller's view of one submitted request: its state, its tokens, and what went wrong.
 *
 * <p>Handed back by {@link LLMEngine#addRequest}. Reads are safe from any thread; delivery happens
 * on the thread running {@code step()}.
 *
 * <p><b>The token is appended before the callback runs</b>. A caller whose callback throws can
 * still read the token that caused it — the failure and the evidence arrive together, which is the
 * difference between a debuggable failure and a mysterious one.
 */
public final class RequestHandle {

    private final ScheduledRequest request;
    private final IntConsumer onToken;

    /**
     * Per-request timing, kept unconditionally.
     *
     * <p>Deliberately not behind the metrics sink's opt-in. That switch exists because collection
     * can be expensive — on the backend it means enabling the profiler, per execution. Two {@code
     * nanoTime} calls per <i>request</i> are not that, and a caller holding a handle should be able
     * to ask how long its own request waited without configuring anything.
     */
    private final long submittedAtNanos = System.nanoTime();

    private volatile long firstTokenAtNanos;
    private volatile long admittedAtNanos;
    private final List<Integer> tokens = new ArrayList<>();
    private final Object completion = new Object();

    RequestHandle(ScheduledRequest request, IntConsumer onToken) {
        this.request = request;
        this.onToken = onToken;
    }

    public RequestState state() {
        return request.state();
    }

    /** Why it was refused, or {@code null}. */
    public RejectionReason rejectionReason() {
        return request.rejectionReason();
    }

    /**
     * What broke, or {@code null}.
     *
     * <p>Retained for a callback failure too: the exception the callback threw is kept here rather
     * than swallowed or rethrown into the step.
     */
    public Throwable failure() {
        return request.failure();
    }

    /** Everything generated so far, in order. Safe to read while the request is still running. */
    public synchronized List<Integer> tokens() {
        return List.copyOf(tokens);
    }

    public synchronized int tokenCount() {
        return tokens.size();
    }

    public boolean isTerminal() {
        return request.isTerminal();
    }

    /** Blocks until this request reaches a terminal state. */
    public void await() throws InterruptedException {
        synchronized (completion) {
            while (!request.isTerminal()) {
                completion.wait();
            }
        }
    }

    ScheduledRequest request() {
        return request;
    }

    /** Nanoseconds this request waited before it was admitted, or {@code 0} if it still waits. */
    public long queueWaitNanos() {
        return admittedAtNanos == 0 ? 0 : admittedAtNanos - submittedAtNanos;
    }

    /**
     * Nanoseconds from submission to the first token — what a caller experiences as latency.
     *
     * <p>Includes the queue wait and the prompt, not just the decode step that produced the token.
     */
    public long timeToFirstTokenNanos() {
        return firstTokenAtNanos == 0 ? 0 : firstTokenAtNanos - submittedAtNanos;
    }

    void markAdmitted() {
        admittedAtNanos = System.nanoTime();
    }

    /** Appends before any callback runs; see the class note. */
    synchronized void appendToken(int token) {
        if (tokens.isEmpty()) {
            firstTokenAtNanos = System.nanoTime();
        }
        tokens.add(token);
    }

    /** The callback, or {@code null}. Invoked by the engine outside every lock. */
    IntConsumer callback() {
        return onToken;
    }

    void signalTerminal() {
        synchronized (completion) {
            completion.notifyAll();
        }
    }
}
