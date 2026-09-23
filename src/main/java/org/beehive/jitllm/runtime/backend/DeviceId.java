package org.beehive.jitllm.runtime.backend;

import java.util.Objects;
import org.beehive.jitllm.api.Experimental;

/**
 * The stable, opaque identity of a resolved device — what equality and the compiled-program cache
 * key compare.
 *
 * <p><b>Opaque on purpose.</b> The handle is whatever the backend uses to name one device to
 * itself, and nothing above the backend may parse it. What matters is only that it is <b>stable for
 * the life of the process</b> and <b>equal exactly when the device is the same one</b>: a cache key
 * component that changes underneath the cache silently produces one compiled program per lookup,
 * which is the defect {@code LoweredPlanSelection}'s device label already had once when it read
 * {@code System.getProperty("tornado.device")} — that property reads {@code "default"} before
 * TornadoVM initializes and {@code "0"} afterwards.
 *
 * <p><b>What is deliberately not here.</b> Capabilities. They are <b>not</b> a function of the
 * device identifier — the scheduler mode is overridable, so two lowerings on one device can differ
 * — so they live in {@link DeviceCapabilities} and enter the cache key separately. Putting them
 * here would make two different lowerings collide because their device happened to match.
 */
@Experimental
public final class DeviceId {

    private final BackendId backend;
    private final String handle;

    private DeviceId(BackendId backend, String handle) {
        this.backend = backend;
        this.handle = handle;
    }

    /**
     * @param backend which implementation the device belongs to
     * @param handle the backend's own stable name for it; opaque above the backend
     */
    public static DeviceId of(BackendId backend, String handle) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(handle, "handle");
        if (handle.isBlank()) {
            throw new IllegalArgumentException("a device handle must not be blank");
        }
        return new DeviceId(backend, handle);
    }

    public BackendId backend() {
        return backend;
    }

    /** The backend's own name for this device. Do not parse it. */
    public String handle() {
        return handle;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DeviceId that
                && backend.equals(that.backend)
                && handle.equals(that.handle);
    }

    @Override
    public int hashCode() {
        return Objects.hash(backend, handle);
    }

    @Override
    public String toString() {
        return backend + ":" + handle;
    }
}
