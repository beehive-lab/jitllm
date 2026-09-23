package org.beehive.jitllm.runtime.backend;

import org.beehive.jitllm.api.Experimental;

/**
 * A <b>resolved</b> device: the answer a backend gives to a {@link DeviceSelector}.
 *
 * <p>An interface, not a value, because the descriptor is <b>backend-owned</b> — the Tornado
 * backend has a {@code TornadoDevice} behind it and the CPU backend has nothing behind it at all,
 * and neither of those belongs in {@code runtime.backend}. What crosses the boundary is this: an
 * identity, a capability set, and a name a human can read in an error message.
 */
@Experimental
public interface Device {

    /** Stable opaque identity. This, not the selector, is what a cache key compares. */
    DeviceId id();

    /**
     * What lowering may vary on. Not derivable from {@link #id()} — see {@link DeviceCapabilities}.
     */
    DeviceCapabilities capabilities();

    /**
     * A human-readable name for diagnostics — a platform or product string. Never parsed, and never
     * used as an identity: {@link #id()} is for that.
     */
    String displayName();

    default BackendId backend() {
        return id().backend();
    }

    /**
     * The largest workgroup (threads per block) the device accepts, as the runtime reports it, or 0
     * when unknown. A kernel whose lane mapping assumes an exact local size asks this before it is
     * selected: a runtime that silently shrinks a workgroup does not run that kernel's arithmetic,
     * it runs a different one.
     */
    default long maxWorkGroupSize() {
        return 0L;
    }

    /**
     * Bytes of header this backend's native arrays carry in front of their elements.
     *
     * <p>A <b>layout</b> fact, not a capability: nothing branches on it, and the one caller needs
     * the number itself. {@code GGUF.loadTensorsTornado} maps each tensor starting this many bytes
     * early so the mapped region already has room for the header the backend's array type expects,
     * and it used to read {@code TornadoNativeArray.ARRAY_HEADER} to learn that — which is how a
     * file-format parser came to name a backend runtime type. The fact is the backend's; it now
     * travels the way every other backend fact does, as an answer from a resolved device.
     *
     * <p>Zero by default, which is the honest answer for a backend whose arrays have no header and
     * for a device that could not be resolved. A caller mapping for a backend that needs no header
     * simply reserves nothing.
     */
    default long nativeArrayHeaderBytes() {
        return 0L;
    }
}
