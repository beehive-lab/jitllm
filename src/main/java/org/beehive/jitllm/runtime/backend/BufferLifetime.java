package org.beehive.jitllm.runtime.backend;

/**
 * How long a device buffer lives, and therefore what releases it.
 *
 * <p><b>An enum, unlike {@link BackendId} and {@link DeviceCapability}.</b> Those are open — a
 * backend or a capability can arrive without this project knowing about it in advance. This one is
 * closed: a fourth lifetime would be a change to the ownership model itself, and it should require
 * editing this file and the matrix together rather than appearing quietly.
 */
public enum BufferLifetime {

    /**
     * Lives as long as the loaded model. Released at model close, and not before.
     *
     * <p>Model weights, and — one level finer in the matrix, but the same release point — the
     * compiled program's fixed device workspace, which its cache entry owns and which the model's
     * program cache releases at close. The workspace's <b>identity is fixed for the life of the
     * program and never rebound</b> [C1]: a captured CUDA graph replays against the addresses it
     * captured, so reallocating one of these mid-life is not a slower path but a wrong one.
     */
    MODEL,

    /**
     * Lives as long as the engine, or as the engineless session runtime that stands in for it.
     *
     * <p>The key/value pool and its device block table, and the engine's batched invocation buffers
     * sized for the maximum batch. Sessions borrow from these through leases and slots; they never
     * own them. A capacity change means a new allocation and a recapture, which is why it is
     * explicit and off the hot path.
     */
    ENGINE,

    /**
     * Lives for one invocation, and is written only inside it.
     *
     * <p>Per-invocation staging and results. The distinction from {@link #ENGINE} is not size but
     * <b>who may be looking</b>: an engine-lifetime buffer is shared by every session in the batch,
     * while an invocation-lifetime one has exactly one reader and no reader that outlives the call.
     *
     * <p>Nothing in this class may be allocated per token. The per-token path allocates nothing — a
     * lease resolves once to a view, and the hot path is an in-kernel block-table walk.
     */
    INVOCATION
}
