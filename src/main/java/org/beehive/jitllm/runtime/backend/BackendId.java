package org.beehive.jllm.runtime.backend;

import java.util.Locale;
import java.util.Objects;
import org.beehive.jllm.api.Experimental;

/**
 * Which execution implementation — the CPU, or one of TornadoVM's accelerator backends.
 *
 * <p><b>A value, not an enum</b>. An enum would have to be edited in this package every time a
 * backend is added, which is the shape Rule 15 exists to prevent: adding a backend should mean
 * adding an implementation, not editing a list in the layer above it. The constants below are
 * conveniences for the ones that exist today; {@link #of(String)} accepts any other.
 *
 * <p>The identifier is canonicalized to lower case so that {@code "CUDA"} and {@code "cuda"} are
 * the same backend. It is compared by value and is stable across runs, which is what lets it sit in
 * {@code ProgramCacheKey} — see {@link DeviceId} for the device half of that key.
 *
 * <p><b>Not a selector.</b> A {@code BackendId} says which implementation; a {@link DeviceSelector}
 * says what is being asked for, including this. The two are separate because a selector may leave
 * the backend unspecified, and because a request must never be used as a cache key.
 */
@Experimental
public final class BackendId {

    /** The host CPU. A first-class backend, not a fallback. */
    public static final BackendId CPU = of("cpu");

    /** TornadoVM's OpenCL backend. */
    public static final BackendId OPENCL = of("opencl");

    /** TornadoVM's CUDA backend. */
    public static final BackendId CUDA = of("cuda");

    /** TornadoVM's Metal backend. */
    public static final BackendId METAL = of("metal");

    private final String id;

    private BackendId(String id) {
        this.id = id;
    }

    /**
     * @param id a non-blank identifier; canonicalized to lower case
     */
    public static BackendId of(String id) {
        Objects.requireNonNull(id, "id");
        String canonical = id.trim().toLowerCase(Locale.ROOT);
        if (canonical.isEmpty()) {
            throw new IllegalArgumentException("a backend identifier must not be blank");
        }
        return new BackendId(canonical);
    }

    /** The canonical identifier. Stable, and safe to put in a cache key or a log line. */
    public String id() {
        return id;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof BackendId that && id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return id;
    }
}
