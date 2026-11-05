package org.beehive.jitllm.backend.tornado.ops.nativelib;

import java.util.function.BooleanSupplier;

/** Asks whether a native library loads in this process. */
public final class LibraryProbe {

    private LibraryProbe() {}

    /**
     * Whether {@code probe} answers yes. A library whose classes cannot be linked, or whose probe
     * throws, does not load: for example {@code loads(MlxLibraryProvider::isAvailable)} on a
     * machine without MLX is false, not an error.
     */
    public static boolean loads(BooleanSupplier probe) {
        try {
            return probe.getAsBoolean();
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }
}
