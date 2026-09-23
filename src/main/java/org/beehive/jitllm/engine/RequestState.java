package org.beehive.jitllm.engine;

/**
 * Where a request is in its life. The whole vocabulary — there is no other state.
 *
 * <p>Allowed transitions, and nothing else [engine-contract.md]:
 *
 * <pre>
 * QUEUED  → RUNNING | REJECTED | CANCELLED
 * RUNNING → COMPLETED | FAILED | CANCELLED
 * </pre>
 *
 * <p>Terminal states never transition.
 */
public enum RequestState {

    /** Submitted, waiting for capacity. Waiting is normal, not an error. */
    QUEUED,

    /** Admitted: holds a slot and a lease, and advances one token per step. */
    RUNNING,

    /** Finished generating — stop token, token budget, or context full. */
    COMPLETED(true),

    /** The caller asked to stop, or closed a non-terminal handle. */
    CANCELLED(true),

    /** Never ran. See {@link RejectionReason}. */
    REJECTED(true),

    /** Ran and broke: a callback threw, the backend failed, or an invariant did not hold. */
    FAILED(true);

    private final boolean terminal;

    RequestState() {
        this(false);
    }

    RequestState(boolean terminal) {
        this.terminal = terminal;
    }

    /** Terminal states release their slot and lease, and never transition again. */
    public boolean isTerminal() {
        return terminal;
    }
}
