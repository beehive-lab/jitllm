package org.beehive.jllm.api;

import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/** Per-session settings. */
public final class SessionOptions {

    private static final SessionOptions DEFAULTS = builder().build();

    private final int contextLength;
    private final ExecutionPolicy.Overrides executionPolicy;

    /**
     * This session's reasoning mode, or {@code null} to inherit the model's.
     *
     * <p>Null rather than {@code DEFAULT}, because the two differ: {@code DEFAULT} is an explicit
     * "leave the family alone" that overrides a model configured otherwise, while null is "I did
     * not say". Collapsing them would make a session unable to turn thinking back off for itself.
     */
    private final ThinkingMode thinkingMode;

    private SessionOptions(Builder builder) {
        this.contextLength = builder.contextLength;
        this.executionPolicy = builder.executionPolicy;
        this.thinkingMode = builder.thinkingMode;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** The model's own settings, unmodified. */
    public static SessionOptions defaults() {
        return DEFAULTS;
    }

    /**
     * This session's field-by-field override of the model's execution policy.
     *
     * <p>Never null; {@link ExecutionPolicy.Overrides#none()} when the session takes the model's
     * policy unchanged.
     */
    @Experimental
    public ExecutionPolicy.Overrides executionPolicy() {
        return executionPolicy;
    }

    /** Context length for this sequence, or 0 to use the model's. Never more than the model's. */
    /** This session's reasoning mode, or {@code null} to inherit the model's default. */
    public ThinkingMode thinkingMode() {
        return thinkingMode;
    }

    /**
     * The mode this session runs with, given the model's default.
     *
     * <p>Resolved once, when the session opens — the same shape as execution policy, and for the
     * same reason: nothing should re-read a configuration decision per generation.
     */
    ThinkingMode resolveThinkingMode(ThinkingMode modelDefault) {
        return thinkingMode != null ? thinkingMode : modelDefault;
    }

    public int contextLength() {
        return contextLength;
    }

    public static final class Builder {

        private int contextLength;
        private ExecutionPolicy.Overrides executionPolicy = ExecutionPolicy.Overrides.none();
        private ThinkingMode thinkingMode;

        private Builder() {}

        /**
         * Caps this sequence's context, which is how a caller runs many short sessions on a model
         * loaded for a long one.
         *
         * @param contextLength tokens; must not exceed the model's context length
         */
        public Builder contextLength(int contextLength) {
            if (contextLength < 0) {
                throw new IllegalArgumentException(
                        "contextLength must not be negative: " + contextLength);
            }
            this.contextLength = contextLength;
            return this;
        }

        /**
         * Overrides the model's execution policy for this session, field by field.
         *
         * <p>A session with a different policy resolves to a different compiled program rather than
         * changing how the model's own program behaves.
         *
         * @param executionPolicy the overrides, or {@code null} for none
         */
        @Experimental
        public Builder executionPolicy(ExecutionPolicy.Overrides executionPolicy) {
            this.executionPolicy =
                    executionPolicy == null ? ExecutionPolicy.Overrides.none() : executionPolicy;
            return this;
        }

        /**
         * Override the model's reasoning mode for this session.
         *
         * @param thinkingMode the mode, or {@code null} to inherit the model's default
         */
        public Builder thinkingMode(ThinkingMode thinkingMode) {
            this.thinkingMode = thinkingMode;
            return this;
        }

        public SessionOptions build() {
            return new SessionOptions(this);
        }
    }
}
