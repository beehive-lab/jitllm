package org.beehive.jitllm.runtime.batch;

/**
 * The one thing a scheduler needs from a backend: advance every active slot by one token.
 *
 * <p><b>Why it lives in {@code runtime} and speaks only in primitives.</b> The dependency rules
 * leave exactly one place for this interface. {@code tornadovm} may not name {@code engine} [Rule
 * 18], and an adapter in {@code engine} would have to import TornadoVM, which would grow Rule 1's
 * shrink-only allowlist. A neutral seam in {@code runtime} is what both sides may depend on — the
 * engine schedules against it, the backend implements it, and neither has to know the other's
 * types.
 *
 * <p>That constraint improved the design rather than bending it. An executor does not need to know
 * what a request <i>is</i>; it needs to know, per slot, whether the slot is active, which token to
 * feed it and at which position. Everything else was the scheduler's business all along.
 *
 * <p>Called only from inside the engine's single-caller {@code step()}, so an implementation needs
 * no locking of its own.
 */
public interface BatchExecutor {

    /** Slots this executor was built for. Fixed: changing it means building another. */
    int maxBatchSize();

    /**
     * Advances every active slot by one token.
     *
     * @param batch what to run — see {@link BatchSlots}
     * @return one token id per slot, positionally. The value at an inactive slot is meaningless and
     *     the caller ignores it.
     */
    int[] decodeStep(BatchSlots batch);

    /**
     * Whether this token ends a sequence.
     *
     * <p>Asked of the executor because a stop token is the model's vocabulary, not the scheduler's.
     * The budget half of "is it finished" stays with the scheduler, which is the half it owns.
     */
    boolean isStopToken(int token);
}
