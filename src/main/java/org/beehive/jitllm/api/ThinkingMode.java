package org.beehive.jllm.api;

/**
 * Whether a model that has a reasoning phase should use it.
 *
 * <p>It exists because a real consumer needs it. The Quarkus extension publishes {@code
 * enable-thinking} as configuration and implemented it by reaching into {@code ChatFormat};
 * migrating it onto the facade without this would have silently dropped a published property.
 *
 * <h2>Where it lives, and where it does not</h2>
 *
 * <p>On {@link ModelOptions} as a model default and on {@link SessionOptions} as a per-session
 * override, resolved once when the session opens — the same shape as execution policy. It is
 * deliberately <b>not</b> on {@code GenerationRequest}: a reasoning mode is configuration, not a
 * per-generation formatting escape hatch, and putting it on the request would make it one.
 */
public enum ThinkingMode {

    /**
     * Whatever the family does on its own.
     *
     * <p>Nothing is encoded, so a family with no reasoning phase is unaffected and one that has a
     * default keeps it. This is the value that changes nothing, and it is the default.
     */
    DEFAULT,

    /** Ask for the reasoning phase. Rejected by a model that cannot represent the control. */
    ENABLED,

    /**
     * Ask the model to skip the reasoning phase. Rejected by a model that cannot represent the
     * control.
     *
     * <p>Rejected rather than ignored: a caller who turned thinking off and silently got it anyway
     * pays for tokens they asked not to generate, and nothing tells them.
     */
    DISABLED;

    /** Whether this mode asks the model for anything, as opposed to leaving it alone. */
    public boolean isExplicit() {
        return this != DEFAULT;
    }
}
