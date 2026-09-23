package org.beehive.jitllm.api;

import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceSelector;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;

/**
 * Load-time settings.
 *
 * <p>Immutable and thread-safe once built.
 *
 * <h2>Backend and device selection</h2>
 *
 * <p>When neither is set the existing mechanism still decides, so nothing that works today stops
 * working: {@code -Duse.tornadovm=true}, which is what the {@code jitllm} launcher's {@code --gpu}
 * sets.
 *
 * <p><b>What a selector can constrain today is narrower than what it can express.</b> The backend
 * is honoured; a device index, a name fragment or a capability requirement is <b>not</b> — nothing
 * in the tree yet places a plan on a chosen device, and per-task placement is a TornadoVM
 * capability this project has not taken up. Asking for one of those <b>throws</b> rather than being
 * ignored, because a selection that is silently dropped is the failure {@code --cuda} without
 * {@code --gpu} used to be: it runs, it is wrong, and nothing says so.
 */
public final class ModelOptions {

    private static final ModelOptions DEFAULTS = builder().build();

    private final int contextLength;
    private final ExecutionPolicy executionPolicy;
    private final StorageOptions storageOptions;
    private final BackendId backend;
    private final DeviceSelector device;
    private final ThinkingMode thinkingMode;
    private final int maxConcurrentSessions;

    private ModelOptions(Builder builder) {
        this.contextLength = builder.contextLength;
        this.maxConcurrentSessions = builder.maxConcurrentSessions;
        this.backend = builder.backend;
        this.device = builder.device;
        this.thinkingMode = builder.thinkingMode;
        // Resolved here, not left null and resolved at use: a null that means "read the system
        // properties later" is the same deferred, order-dependent read this task removes.
        this.executionPolicy =
                builder.executionPolicy != null
                        ? builder.executionPolicy
                        : ExecutionPolicy.fromSystemProperties();
        this.storageOptions =
                builder.storageOptions != null
                        ? builder.storageOptions
                        : StorageOptions.fromSystemProperties();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Whatever the model file says, with no overrides. */
    public static ModelOptions defaults() {
        return DEFAULTS;
    }

    /**
     * Context length to load with, or 0 for the model's own.
     *
     * <p>This sizes the key/value cache, so it is a memory decision as much as a capability one.
     */
    /**
     * The default execution policy for every session on this model.
     *
     * <p>Resolved from the {@code llama.*} system properties when the caller sets none, which is
     * how this lands without changing any existing behaviour: the value a session gets is the value
     * the class constants held.
     */
    @Experimental
    public ExecutionPolicy executionPolicy() {
        return executionPolicy;
    }

    /**
     * How this model's key/value storage is shaped.
     *
     * <p>Separate from {@link #executionPolicy()} because it is: a value that sizes or types a
     * device array is not a session choice, and two sessions sharing a pool cannot each have their
     * own idea of what the pool is made of.
     */
    @Experimental
    public StorageOptions storageOptions() {
        return storageOptions;
    }

    public int contextLength() {
        return contextLength;
    }

    /**
     * How many {@link GenerationSession}s of the loaded model may be open at once; 1 unless set.
     *
     * <p>A capacity, not a thread count: it bounds open session objects, and each session still
     * generates one request at a time. It is a memory decision like {@link #contextLength()}: where
     * sessions share one key/value pool, the pool is reserved for this many sessions at the full
     * load-time context when the model loads. A session that asks for a shorter context does not
     * shrink that reservation.
     */
    public int maxConcurrentSessions() {
        return maxConcurrentSessions;
    }

    /**
     * Which backend to load for, or {@code null} to let {@code -Duse.tornadovm} decide.
     *
     * <p>{@link BackendId#CPU} is a real choice, not a fallback.
     */
    @Experimental
    public BackendId backend() {
        return backend;
    }

    /**
     * The default reasoning mode for every session on this model.
     *
     * <p>{@link ThinkingMode#DEFAULT} unless set, which leaves the family's own behaviour alone.
     */
    public ThinkingMode thinkingMode() {
        return thinkingMode;
    }

    /** What device was asked for, or {@code null} for the backend's own pick. */
    @Experimental
    public DeviceSelector device() {
        return device;
    }

    /**
     * The backend these options resolve to, or {@code null} for "not stated — use the property".
     *
     * <p>An explicit {@link #backend()} wins; otherwise a selector that names one supplies it. The
     * two are allowed to disagree only by one being absent: stating {@code backend(CPU)} beside
     * {@code device(selector for cuda)} is a contradiction, and answering it with a precedence rule
     * would just be picking one of the caller's two intentions without telling them.
     */
    BackendId resolvedBackend() {
        BackendId fromSelector = device == null ? null : device.backendId().orElse(null);
        if (backend != null && fromSelector != null && !backend.equals(fromSelector)) {
            throw new IllegalArgumentException(
                    "backend("
                            + backend
                            + ") and device("
                            + device
                            + ") name different backends; set one, or make them agree");
        }
        return backend != null ? backend : fromSelector;
    }

    /**
     * Rejects a selector that expresses more than the tree can honour.
     *
     * <p>On the outer class rather than on {@code Builder} deliberately: {@code FacadeSurfaceTest}
     * pins the builder's <b>declared</b> methods, so a private helper there would widen a surface
     * pin that is meant to be exact. The test caught this, which is what an exact pin is for.
     */
    private static void rejectWhatCannotBeHonoured(DeviceSelector device) {
        if (device == null) {
            return;
        }
        if (device.index().isPresent()) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.DEVICE_SELECTOR_UNSUPPORTED.message(
                                    "selecting a device by index is not implemented")
                            + " yet; nothing places a plan on a chosen device. Select a backend, and use"
                            + " TornadoVM's own device properties meanwhile");
        }
        if (device.nameContains().isPresent()) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.DEVICE_SELECTOR_UNSUPPORTED.message(
                                    "selecting a device by name is not implemented")
                            + " yet; nothing places a plan on a chosen device. Select a backend, and use"
                            + " TornadoVM's own device properties meanwhile");
        }
        if (!device.capabilityRequirements().isEmpty()) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.CAPABILITY_UNAVAILABLE.message(
                                    "requiring a device capability is not")
                            + " implemented yet: the capability is resolved from the device that is used,"
                            + " and nothing yet chooses a device to satisfy a requirement");
        }
    }

    public static final class Builder {

        private int contextLength;
        private ExecutionPolicy executionPolicy;
        private StorageOptions storageOptions;
        private BackendId backend;
        private DeviceSelector device;
        private ThinkingMode thinkingMode = ThinkingMode.DEFAULT;
        private int maxConcurrentSessions = 1;

        private Builder() {}

        public Builder contextLength(int contextLength) {
            if (contextLength < 0) {
                throw new IllegalArgumentException(
                        "contextLength must not be negative: " + contextLength);
            }
            this.contextLength = contextLength;
            return this;
        }

        /**
         * How many sessions of the loaded model may be open at once.
         *
         * <p>Opening one more than this throws; closing a session frees its place. Size it to the
         * concurrency the caller actually needs: every place may be reserved up front.
         *
         * @param maxConcurrentSessions at least 1; the default is 1
         */
        public Builder maxConcurrentSessions(int maxConcurrentSessions) {
            if (maxConcurrentSessions < 1) {
                throw new IllegalArgumentException(
                        "maxConcurrentSessions must be at least 1: " + maxConcurrentSessions);
            }
            this.maxConcurrentSessions = maxConcurrentSessions;
            return this;
        }

        /**
         * The model's default execution policy; sessions may override it field by field.
         *
         * @param executionPolicy the policy, or {@code null} for the properties-derived default
         */
        @Experimental
        public Builder executionPolicy(ExecutionPolicy executionPolicy) {
            this.executionPolicy = executionPolicy;
            return this;
        }

        /**
         * How this model's key/value storage is shaped.
         *
         * @param storageOptions the options, or {@code null} for the properties-derived default
         */
        @Experimental
        public Builder storageOptions(StorageOptions storageOptions) {
            this.storageOptions = storageOptions;
            return this;
        }

        /**
         * Which backend to load for.
         *
         * @param backend the backend, or {@code null} to let {@code -Duse.tornadovm} decide
         */
        @Experimental
        public Builder backend(BackendId backend) {
            this.backend = backend;
            return this;
        }

        /**
         * What device to ask the backend for.
         *
         * <p>Only the selector's <b>backend</b> is honoured today; an index, a name fragment or a
         * capability requirement throws at {@link #build()} rather than being quietly ignored.
         *
         * @param device the selector, or {@code null} for the backend's own pick
         */
        @Experimental
        public Builder device(DeviceSelector device) {
            this.device = device;
            return this;
        }

        /**
         * The model's default reasoning mode; sessions may override it.
         *
         * @param thinkingMode the mode, or {@code null} for {@link ThinkingMode#DEFAULT}
         */
        public Builder thinkingMode(ThinkingMode thinkingMode) {
            this.thinkingMode = thinkingMode == null ? ThinkingMode.DEFAULT : thinkingMode;
            return this;
        }

        public ModelOptions build() {
            rejectWhatCannotBeHonoured(device);
            ModelOptions options = new ModelOptions(this);
            // Fails here rather than at load, so a contradiction is reported where it was written.
            options.resolvedBackend();
            return options;
        }
    }
}
