package org.beehive.jitllm.api;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.runtime.kv.KvLease;

/**
 * How one session gets the state and the plan it executes with.
 *
 * <p>Two implementations, and the difference between them is an ownership difference rather than a
 * mode flag: the legacy runtime **owns** a per-session {@code State} and its plan, while the
 * lowered runtime owns only session-logical values and **borrows** a binding domain's workspace and
 * its shared compiled program.
 *
 * <p>A typed seam on purpose. The alternative — a nullable {@code State} and {@code if (lowered)}
 * scattered through {@code DelegatingSession} — would put the ownership rules in the reader's head
 * instead of in the types, and lifecycle is exactly where that goes wrong: what {@code close()}
 * releases differs between the two, and getting it wrong frees device memory another session is
 * still using.
 */
sealed interface SessionRuntime permits LegacySessionRuntime, LoweredSessionRuntime {

    /**
     * The state to pass to the model's generation entry points.
     *
     * <p>For the legacy runtime this is the session's own. For the lowered runtime it is the
     * <b>domain's workspace</b>, shared with every other session in that domain and safe because
     * invocation is serialized behind the boundary.
     */
    State executionState();

    /** This session's claim on key/value storage. */
    KvLease lease();

    /** The plan this session executes with, built or borrowed on first use. */
    TornadoVMMasterPlan plan();

    /**
     * Runs one GPU turn, with whichever generation cursor this runtime owns.
     *
     * <p>The legacy runtime's cursor is its state's own field; the lowered runtime's is its {@code
     * SessionLogicalState}, so its conversation history travels with it rather than living in a
     * workspace another session will overwrite.
     */
    java.util.List<Integer> generateOnGpu(
            org.beehive.jitllm.model.Model model,
            int startPosition,
            java.util.List<Integer> promptTokens,
            java.util.Set<Integer> stopTokens,
            int budget,
            org.beehive.jitllm.inference.sampler.Sampler sampler,
            java.util.function.IntConsumer onToken);

    /** Whether a GPU plan has been obtained yet. */
    boolean hasPlan();

    /**
     * Called before a turn: makes this session's logical values current in the execution state.
     *
     * <p>The one that matters is {@code latestToken}, which {@code PromptIngestion} reads as the
     * <b>conversation-continuation seed</b>. In a shared workspace that field is written by whoever
     * went last, so a session that did not restore its own would continue someone else's
     * conversation — which is exactly what the poison test caught.
     */
    void beginTurn();

    /** Called after a turn: captures back whatever the turn left in the execution state. */
    void endTurn();

    /** Returns this session's logical state to its starting point. */
    void reset();

    /**
     * Releases what this session owns, and nothing it borrows.
     *
     * <p>The distinction is the point: a lowered session that released the domain's workspace or
     * the shared program would tear down device memory its siblings are still executing in.
     */
    void close();
}
