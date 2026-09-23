package org.beehive.jitllm.api;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.runtime.kv.KvLease;

/**
 * A session on the lowered path: it owns logical values and <b>borrows</b> everything physical.
 *
 * <p><b>It constructs no {@code State} of its own.</b> That is the whole point of the slice: a
 * session that kept one would allocate its own activation, projection, attention and temporary
 * arrays — the duplication the shared workspace exists to remove — even while sharing the compiled
 * program. So the execution state it hands out is the <b>domain's workspace</b>, and what it owns
 * is a {@link SessionLogicalState} and a lease.
 *
 * <p>Sharing the workspace is safe because invocation is serialized behind the compiled program's
 * boundary, which holds the domain's lock across staging, execution and copy-out, and releases it
 * before any caller code runs.
 *
 * <p><b>One field of the borrowed workspace is session-logical and must never be read here:</b>
 * {@code latestToken}. The generation loop writes it per token, and on the single-token path never
 * reads it back — its driver is a local. This runtime keeps its own copy so that nothing depends on
 * the shared one, and a test poisons the workspace's value between two sessions' turns to prove the
 * independence rather than assert it.
 */
final class LoweredSessionRuntime implements SessionRuntime {

    private final State workspace;
    private final KvLease lease;
    private final SessionLogicalState logical;
    private final TornadoVMMasterPlan borrowedPlan;

    LoweredSessionRuntime(
            State workspace, KvLease lease, TornadoVMMasterPlan borrowedPlan, int initialToken) {
        this.workspace = workspace;
        this.lease = lease;
        this.borrowedPlan = borrowedPlan;
        this.logical = new SessionLogicalState(initialToken);
    }

    /** This session's own generation cursor. Never the workspace's. */
    SessionLogicalState cursor() {
        return logical;
    }

    /** The domain's workspace, shared with every other session in this domain. */
    @Override
    public State executionState() {
        return workspace;
    }

    @Override
    public KvLease lease() {
        return lease;
    }

    @Override
    public TornadoVMMasterPlan plan() {
        return borrowedPlan;
    }

    @Override
    public boolean hasPlan() {
        return true;
    }

    /**
     * Runs the turn with <b>this session's</b> cursor, never the workspace's.
     *
     * <p>The lowered tuple is Llama FP16 single-token, so it enters the cursor-taking Llama loop
     * directly. That is the one place the borrowed workspace and the session's own history meet,
     * and they meet as separate arguments rather than as one object.
     */
    @Override
    public java.util.List<Integer> generateOnGpu(
            org.beehive.jitllm.model.Model model,
            int startPosition,
            java.util.List<Integer> promptTokens,
            java.util.Set<Integer> stopTokens,
            int budget,
            org.beehive.jitllm.inference.sampler.Sampler sampler,
            java.util.function.IntConsumer onToken) {
        return org.beehive.jitllm.inference.TokenGenerationLoop.generateTokensGpu(
                model,
                workspace,
                logical,
                startPosition,
                promptTokens,
                stopTokens,
                budget,
                sampler,
                false,
                onToken,
                borrowedPlan);
    }

    /** Nothing to restore, and deliberately so. */
    @Override
    public void beginTurn() {}

    @Override
    public void endTurn() {}

    /**
     * Resets this session and nothing else.
     *
     * <p>It deliberately does <b>not</b> touch the workspace's {@code latestToken}: that field is
     * shared, and clearing it here would reach into another session's turn. This session's own
     * logical value is what it resets, and its key/value history is reset through its lease.
     */
    @Override
    public void reset() {
        logical.reset();
    }

    /**
     * Releases this session's lease, and nothing else.
     *
     * <p>The workspace and the compiled program belong to the binding domain and outlive every
     * session that borrowed them. Releasing either here would tear down device memory a sibling
     * session is still executing in — and the model handle releases them once, at close, when
     * guarantees no borrower is left.
     */
    @Override
    public void close() {
        lease.close();
    }
}
