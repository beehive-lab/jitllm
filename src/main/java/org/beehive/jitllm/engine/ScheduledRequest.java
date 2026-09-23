package org.beehive.jitllm.engine;

import org.beehive.jitllm.runtime.kv.KvLease;

/**
 * One request as the scheduler sees it: a declared budget, and — once admitted — a slot and a
 * lease.
 *
 * <p>Not thread-safe on its own. The {@link Scheduler} that owns it is the synchronizing point.
 */
public final class ScheduledRequest {

    /** Monotonic, and the tie-break FCFS ordering is defined on. */
    private final long sequenceNumber;

    /**
     * Tokens this request may occupy — prompt plus generation budget.
     *
     * <p>Reservation is against this number and nothing else: admission takes the blocks it needs
     * once, up front, and holds them to a terminal state. There is no incremental growth, which is
     * why a request cannot be starved after it has been admitted.
     */
    private final int declaredBudgetTokens;

    /**
     * The prepared prompt, as exact token ids.
     *
     * <p>Tokens, never text. A caller that has a conversation renders it through the model's own
     * chat template and submits the ids — flattening a history into a string and re-encoding it
     * produces a different token sequence, and the engine would have no way to notice. This field
     * is why the engine needs no notion of a conversation at all.
     */
    private int[] promptTokens = new int[0];

    /** How many tokens have been fed to the model — the next sequence position. */
    private int consumed;

    private int maxNewTokens;

    private RequestState state = RequestState.QUEUED;
    private RejectionReason rejectionReason;
    private Throwable failure;
    private KvLease lease;
    private int slot = -1;

    public ScheduledRequest(long sequenceNumber, int declaredBudgetTokens) {
        this.sequenceNumber = sequenceNumber;
        this.declaredBudgetTokens = declaredBudgetTokens;
    }

    public long sequenceNumber() {
        return sequenceNumber;
    }

    public int declaredBudgetTokens() {
        return declaredBudgetTokens;
    }

    /** The prepared prompt. Never copied on read; the engine does not mutate it. */
    public int[] promptTokens() {
        return promptTokens;
    }

    void promptTokens(int[] tokens) {
        this.promptTokens = tokens;
    }

    /** How many tokens the caller asked to generate, beyond the prompt. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    void maxNewTokens(int maxNewTokens) {
        this.maxNewTokens = maxNewTokens;
    }

    /** Tokens fed so far, which is also the position the next one occupies. */
    public int consumed() {
        return consumed;
    }

    void consume() {
        consumed++;
    }

    /**
     * Starts ingestion past the tokens a shared prefix already covers.
     *
     * <p>The sequence begins at that position rather than at zero, which is the entire saving: the
     * device never runs those positions.
     *
     * <p><b>The last prompt token is always re-fed</b>, even when the prefix covers the whole
     * prompt. The cache holds that position's <i>KV</i>, not the <i>logits</i> the forward
     * produced, and the logits are what the first generated token is sampled from. Skipping it
     * would leave the request with nothing to sample and nothing to feed. Qwen2-MoE's batch path
     * had already discovered this — "keep the final prompt token for the B1 decode graph, which
     * produces the first generation logits".
     */
    void skipPrefilled(int tokens) {
        this.consumed = Math.max(0, Math.min(tokens, promptTokens.length - 1));
    }

    /**
     * Whether the token sampled after feeding position {@code consumed} is real output.
     *
     * <p>Prefill here is decode: the prompt is fed one token per step through the same graphs, and
     * the sample that follows each prompt token predicts the next prompt token — which is already
     * known and therefore discarded. Only the sample after the <b>last</b> prompt token is the
     * first thing the model actually generated.
     */
    public boolean isGenerating() {
        return consumed >= promptTokens.length - 1;
    }

    /** The token to feed this step: the next prompt token, or the last one sampled. */
    public int nextInput(int lastSampled) {
        return consumed < promptTokens.length ? promptTokens[consumed] : lastSampled;
    }

    public RequestState state() {
        return state;
    }

    /** Why it was rejected, or {@code null} if it was not. */
    public RejectionReason rejectionReason() {
        return rejectionReason;
    }

    /** What broke, or {@code null}. Retained on the handle so a caller can see it. */
    public Throwable failure() {
        return failure;
    }

    /** The lease backing this request while it runs, or {@code null}. */
    public KvLease lease() {
        return lease;
    }

    /** Its position in the batch while running, or {@code -1}. */
    public int slot() {
        return slot;
    }

    public boolean isTerminal() {
        return state.isTerminal();
    }

    // ── transitions, all of them ────────────────────────────────────────────────────────────────
    // Package-private on purpose: the scheduler is the only thing allowed to move a request, so
    // the state machine has exactly one implementation rather than one per caller.

    void reject(RejectionReason reason) {
        require(state == RequestState.QUEUED, RequestState.REJECTED);
        this.state = RequestState.REJECTED;
        this.rejectionReason = reason;
    }

    void admit(int slot, KvLease lease) {
        require(state == RequestState.QUEUED, RequestState.RUNNING);
        this.state = RequestState.RUNNING;
        this.slot = slot;
        this.lease = lease;
    }

    void complete() {
        require(state == RequestState.RUNNING, RequestState.COMPLETED);
        this.state = RequestState.COMPLETED;
    }

    void cancel() {
        require(
                state == RequestState.QUEUED || state == RequestState.RUNNING,
                RequestState.CANCELLED);
        this.state = RequestState.CANCELLED;
    }

    void fail(Throwable cause) {
        require(state == RequestState.RUNNING, RequestState.FAILED);
        this.state = RequestState.FAILED;
        this.failure = cause;
    }

    /** Called by the scheduler once the slot and lease have been given back. */
    void releaseResources() {
        this.slot = -1;
        this.lease = null;
    }

    private void require(boolean allowed, RequestState target) {
        if (!allowed) {
            throw new IllegalStateException(
                    "request #"
                            + sequenceNumber
                            + " cannot go from "
                            + state
                            + " to "
                            + target
                            + "; terminal states never transition and the"
                            + " allowed set is QUEUED → RUNNING|REJECTED|CANCELLED,"
                            + " RUNNING → COMPLETED|FAILED|CANCELLED");
        }
    }

    @Override
    public String toString() {
        return "request #"
                + sequenceNumber
                + " "
                + state
                + (slot >= 0 ? " slot=" + slot : "")
                + (rejectionReason != null ? " (" + rejectionReason + ")" : "");
    }
}
