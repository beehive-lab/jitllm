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
     * {@code cause}, memory running out while a plan was built ({@link #isDeviceExhaustion}), as
     * this exception, worded for the host's direct memory or the device budget. Example: {@code
     * TornadoVMMasterPlan.initializeTornadoVMPlan} catches what building the plan threw and
     * rethrows anything else unchanged.
     */
    public static DeviceMemoryExhaustedException buildingPlan(Throwable cause) {
        return new DeviceMemoryExhaustedException(
                message("building the execution plan", cause), cause);
    }

    // @formatter:off
    /**
     * Whether this failure, or anything it wraps, is memory running out for the accelerator path:
     * TornadoVM's device budget, or the host's direct memory, which the JVM limits separately
     * ({@code -XX:MaxDirectMemorySize}, by default the maximum heap) and where native arrays and
     * MLX's copies of the projections are made. A heap {@code OutOfMemoryError} is neither.
     *
     * <p>Matched on the type name rather than by catching the TornadoVM class, so that this stays
     * compilable against a TornadoVM whose exception hierarchy has moved — which it has, twice, in
     * the version range this project supports.
     */
    // @formatter:on
    public static boolean isDeviceExhaustion(Throwable failure) {
        for (Throwable t = failure; t != null; t = t.getCause()) {
            String type = t.getClass().getName();
            if ((type.startsWith("uk.ac.manchester.tornado")
                            && type.toLowerCase(java.util.Locale.ROOT).contains("outofmemory"))
                    || isDirectMemoryExhaustion(t)) {
                return true;
            }
        }
        return false;
    }

    /** Whether {@code failure} itself is the JVM refusing direct memory past its limit. */
    private static boolean isDirectMemoryExhaustion(Throwable failure) {
        return failure instanceof OutOfMemoryError
                && failure.getMessage() != null
                && failure.getMessage().contains("direct buffer memory");
    }

    // @formatter:off
    /**
     * The {@code GPUL-MEM-001} message for {@code cause}, which ran out while {@code doing}: the
     * host's direct-memory limit, or the device budget, each with what changes it.
     *
     * <p>Example: {@code Cannot reserve 117440528 bytes of direct buffer memory} while building a
     * plan names {@code -XX:MaxDirectMemorySize} and {@code --heap-max}, not {@code --gpu-memory},
     * which does not change that limit.
     */
    // @formatter:on
    public static String message(String doing, Throwable cause) {
        boolean host = false;
        for (Throwable t = cause; t != null; t = t.getCause()) {
            host |= isDirectMemoryExhaustion(t);
        }
        String prefix = DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT.prefix();
        return host
                ? prefix
                        + "The host ran out of direct memory while "
                        + doing
                        + ": the JVM's limit on direct and native memory, not the device budget,"
                        + " so --gpu-memory does not change it.\n"
                        + "  Raise -XX:MaxDirectMemorySize (by default the maximum heap,"
                        + " --heap-max), reduce the context length, run without native libraries"
                        + " (--no-native-libraries), whose copies are made there, or select a"
                        + " smaller quantization.\n  Host failure: "
                        + cause
                : prefix
                        + "The device ran out of memory while "
                        + doing
                        + ".\n  Raise --gpu-memory, reduce the context length, or select a smaller"
                        + " quantization.\n  Backend failure: "
                        + cause;
    }
}
