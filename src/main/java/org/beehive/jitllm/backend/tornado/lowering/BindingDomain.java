package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Objects;

/**
 * The identity of the physical arrays a compiled program is bound to.
 *
 * <p>The key/value pool, the block table and the captured workspace, taken together. Minted by
 * whichever runtime owns those arrays — the model handle's session runtime, or an engine — and
 * compared by <b>identity, not by description</b>: two domains with identical shapes are still two
 * domains.
 *
 * <p>That is what stops an engine's program being handed to a standalone session. Their pools
 * differ, so their domains differ, so their cache entries differ, and no amount of shape equality
 * can make them share.
 *
 * <p>Sessions of one model on one session runtime are in <b>one</b> domain, which is what lets them
 * share a compiled program and one copy of the device weights.
 */
public final class BindingDomain {

    private final String label;
    private final org.beehive.jitllm.inference.state.State workspace;
    private final Object invocationLock = new Object();

    private BindingDomain(String label, org.beehive.jitllm.inference.state.State workspace) {
        this.label = label;
        this.workspace = workspace;
    }

    /**
     * Mints a domain with no workspace, for keys and tests.
     *
     * @param label a human-readable owner name, for diagnostics only — it takes no part in identity
     */
    public static BindingDomain create(String label) {
        return new BindingDomain(Objects.requireNonNull(label, "label"), null);
    }

    /**
     * Mints a shareable domain around one fixed device workspace.
     *
     * <p><b>The workspace must address shared key/value storage.</b> A domain that combined a
     * shared workspace and program with session-private key/value arrays would let the second
     * session execute against the first session's cache — silently, and under CUDA graph capture as
     * wrong output rather than an error. That combination is unconstructible here rather than
     * merely discouraged: this refuses it (option 1).
     */
    public static BindingDomain shareable(
            String label, org.beehive.jitllm.inference.state.State workspace) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(workspace, "workspace");
        if (workspace.kvLease == null || workspace.kvLease.storage() == null) {
            throw new IllegalStateException(
                    "a shareable binding domain requires shared key/value"
                            + " storage: its workspace addresses session-private key/value arrays, which"
                            + " two sessions sharing one compiled program would corrupt");
        }
        return new BindingDomain(label, workspace);
    }

    /** The fixed device workspace every session borrowing this domain's program executes in. */
    public org.beehive.jitllm.inference.state.State workspace() {
        if (workspace == null) {
            throw new IllegalStateException(
                    "this domain has no workspace; it was minted for a key");
        }
        return workspace;
    }

    /** Whether this domain owns a workspace and can therefore be shared. */
    public boolean isShareable() {
        return workspace != null;
    }

    /** The lock that serializes invocation in this domain's workspace. */
    public Object invocationLock() {
        return invocationLock;
    }

    /** For logs. Never a substitute for the object itself. */
    public String label() {
        return label;
    }

    @Override
    public String toString() {
        return "BindingDomain["
                + label
                + "@"
                + Integer.toHexString(System.identityHashCode(this))
                + "]";
    }

    // equals and hashCode are deliberately Object's: identity, not description.
}
