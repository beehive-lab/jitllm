package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * The device ran out of memory while building the execution plan.
 *
 * <pre>
 * Exception in thread "main" uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException:
 *   Unable to allocate 33554448 bytes of memory.
 *     To increase the maximum device memory, use -Dtornado.device.memory=&lt;X&gt;GB
 * </pre>
 *
 * naming a byte count and a TornadoVM-internal property rather than this project's {@code
 * --gpu-memory} flag, with nothing to say which component dominated the budget.
 *
 * <p><b>Where a configuration that does not fit fails.</b> Nothing refuses a load on a prediction
 * of its memory: a plan whose buffers do not fit the configured budget ({@code --gpu-memory}) or
 * the device fails here, as it allocates them.
 *
 * <p><b>The cause chain is preserved deliberately.</b> {@code LocalModels} identifies device
 * exhaustion by walking causes for a TornadoVM out-of-memory type, and then reports {@code
 * InsufficientDeviceMemoryException} with the memory plan attached. Wrapping without the cause
 * would break that, and the public API path would silently lose its diagnostic.
 *
 * <p>Deliberately <b>not</b> {@code api.InsufficientDeviceMemoryException}: {@code backend.tornado}
 * does not depend on {@code api} and must not start, or the layering inverts. Translation to the
 * public type happens in {@code api}, which is where the public vocabulary lives.
 */
public class DeviceMemoryExhaustedException extends RuntimeException {

    DeviceMemoryExhaustedException(String message, Throwable cause) {
        super(message, cause);
    }

    /** The stable code for this failure: {@code GPUL-MEM-001}. */
    public DiagnosticCode code() {
        return DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT;
    }

    /**
     * Runs {@code build}, a step that allocates a plan's device buffers, and reports a device
     * out-of-memory it raises as this exception; any other failure passes unchanged. Both ways a
     * plan is built go through here: {@code TornadoVMMasterPlan.initializeTornadoVMPlan} and the
     * lowered plans a session builds.
     */
    public static <T> T whileAllocating(java.util.function.Supplier<T> build) {
        try {
            return build.get();
        } catch (RuntimeException | OutOfMemoryError failure) {
            if (isDeviceExhaustion(failure)) {
                throw wrap(failure);
            }
            throw failure;
        }
    }

    /**
     * Whether this failure, or anything it wraps, is a device out-of-memory: TornadoVM's, or the
     * host running out of the direct memory a plan stages device copies in (MLX's forms of the
     * projections are made there).
     *
     * <p>Matched on the type name rather than by catching the TornadoVM class, so that this stays
     * compilable against a TornadoVM whose exception hierarchy has moved — which it has, twice, in
     * the version range this project supports.
     */
    private static boolean isDeviceExhaustion(Throwable failure) {
        for (Throwable t = failure; t != null; t = t.getCause()) {
            String type = t.getClass().getName();
            if (type.startsWith("uk.ac.manchester.tornado")
                    && type.toLowerCase(java.util.Locale.ROOT).contains("outofmemory")) {
                return true;
            }
            if (t instanceof OutOfMemoryError
                    && t.getMessage() != null
                    && t.getMessage().contains("direct buffer memory")) {
                return true;
            }
        }
        return false;
    }

    /** The message the CLI user sees: what failed and what to change. */
    private static DeviceMemoryExhaustedException wrap(Throwable cause) {
        String message =
                DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT.prefix()
                        + "The device ran out of memory while building the execution plan.\n"
                        + "  Raise --gpu-memory, reduce the context length, or select a smaller"
                        + " quantization.\n  Backend failure: "
                        + cause;
        return new DeviceMemoryExhaustedException(message, cause);
    }
}
