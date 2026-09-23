package org.beehive.jllm.api;

import org.beehive.jllm.auxiliary.metrics.RunMetricsSink;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.kv.KvLease;

/**
 * The session runtime as it has always been: this session owns its {@code State} and its plan.
 *
 * <p>Every path but the lowered Llama FP16 single-token one uses this, unchanged — the CPU path,
 * the legacy GPU path, and every other model family. Its device arrays are allocated once per
 * session, which is the cost the lowered runtime exists to remove and which this one keeps until
 * that is proven.
 */
final class LegacySessionRuntime implements SessionRuntime {

    private final Model model;
    private final State state;
    private final KvLease lease;

    /**
     * The start token the model seeded this state with — beginning-of-text, or a start-header token
     * for the families whose prompts do not repeat it. What {@link #reset()} restores.
     */
    private final int initialToken;

    private TornadoVMMasterPlan plan;

    LegacySessionRuntime(
            Model model,
            KvLease lease,
            org.beehive.jllm.runtime.policy.ExecutionPolicy executionPolicy,
            org.beehive.jllm.runtime.policy.StorageOptions storageOptions) {
        this.model = model;
        this.lease = lease;
        // The model's storage options shape what this state allocates. They are the model's, not
        // the session's: a value that types or sizes an array cannot differ per session on a
        // shared pool.
        // Both are allocation-time inputs, so both are scoped around the construction: the
        // storage options type the arrays, and the prefill batch width sizes them. The policy
        // is only resolved on the finished state, which is too late to allocate for.
        this.state =
                org.beehive.jllm.inference.state.State.withStorageOptions(
                        storageOptions,
                        () ->
                                org.beehive.jllm.inference.state.State.withPrefillBatchSize(
                                        executionPolicy == null
                                                ? 1
                                                : executionPolicy.prefillBatchSize(),
                                        () -> model.createNewState(lease)));
        // Before the plan is built, which is where the layers read it. The state refuses
        // a change after a plan has read it, so this is the one moment it can happen.
        //
        // A model may legitimately have no state here — a CPU-only stand-in in a lifecycle test
        // returns none — and a session that never executes has no policy to carry.
        if (this.state != null) {
            this.state.resolveExecutionPolicy(executionPolicy);
        }
        // Captured before a single token is fed, so it is the seed the model chose rather than
        // whatever the last turn left behind.
        this.initialToken = this.state != null ? this.state.latestToken : -1;
    }

    @Override
    public State executionState() {
        return state;
    }

    @Override
    public KvLease lease() {
        return lease;
    }

    @Override
    public TornadoVMMasterPlan plan() {
        if (plan == null) {
            plan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(
                            state, model, RunMetricsSink.installedOrDisabled());
        }
        return plan;
    }

    @Override
    public boolean hasPlan() {
        return plan != null;
    }

    /** Straight through the model, exactly as before. */
    @Override
    public java.util.List<Integer> generateOnGpu(
            Model model,
            int startPosition,
            java.util.List<Integer> promptTokens,
            java.util.Set<Integer> stopTokens,
            int budget,
            org.beehive.jllm.inference.sampler.Sampler sampler,
            java.util.function.IntConsumer onToken) {
        return model.generateTokensGPU(
                state,
                startPosition,
                promptTokens,
                stopTokens,
                budget,
                sampler,
                false,
                onToken,
                plan());
    }

    /** Nothing to do: this session's state is its own, so its values are already current. */
    @Override
    public void beginTurn() {}

    @Override
    public void endTurn() {}

    /**
     * Returns the sequence to the condition its state was created in.
     *
     * <p><b>It used to set {@code latestToken = -1}, and that was a defect.</b> {@code -1} is
     * {@code State}'s "not yet set" value from its constructor, but every family's {@code
     * createNewState} then seeds a real start token over it — beginning-of-text for Llama, a
     * start-header token for Qwen and Phi3 — and {@code PromptIngestion} reads that seed as the
     * first token to feed. After a reset it read {@code -1}, and {@code -1} reached the embedding
     * table as an index: {@code reset()} followed by {@code generate()} threw an {@code
     * AssertionError} from {@code Q8_0FloatTensor.getFloat} on the CPU path, or read out of bounds
     * with assertions off.
     *
     * <p>Restoring the token the state was created with is right for every family, including the
     * ones whose prompts do not repeat it: reset means "this session is new again", and a new
     * session has that seed.
     */
    @Override
    public void reset() {
        state.latestToken = initialToken;
        // Families with recurrent state clear it here; for everyone else this does nothing.
        // Through the plan when there is one: the device copy of that state persists between
        // executions and is never re-read from the host, so a host-only clear would leave the
        // accelerator continuing the previous sequence.
        if (plan != null) {
            plan.resetSequenceState();
        } else {
            state.resetSequenceState();
        }
    }

    /** This session built its plan, so this session frees it. */
    @Override
    public void close() {
        if (plan != null) {
            plan.freeTornadoExecutionPlan();
            plan = null;
        }
        lease.close();
    }
}
