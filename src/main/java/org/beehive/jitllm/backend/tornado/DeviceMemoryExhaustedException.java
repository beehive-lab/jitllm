package org.beehive.jllm.backend.tornado;

import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;

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
 * <p><b>Why the failure arrives here at all</b>, rather than from the preflight: {@code
 * MemoryPreflight} refuses only at {@code EXACT} confidence, and Metal is deliberately capped at
 * {@code CONSERVATIVE} until its own thresholds are measured (task 13, {@code
 * memory-validation.md}). {@code CONSERVATIVE} is report-only by definition, so the load proceeds
 * and exhaustion is reported by the runtime. That is the designed behaviour of {@code
 * CONSERVATIVE}, not a defect — but it makes this the failure a Metal user actually sees, so it is
 * worth stating well.
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
     * Whether this failure, or anything it wraps, is a device out-of-memory.
     *
     * <p>Matched on the type name rather than by catching the TornadoVM class, so that this stays
     * compilable against a TornadoVM whose exception hierarchy has moved — which it has, twice, in
     * the version range this project supports.
     */
    static boolean isDeviceExhaustion(Throwable failure) {
        for (Throwable t = failure; t != null; t = t.getCause()) {
            String type = t.getClass().getName();
            if (type.startsWith("uk.ac.manchester.tornado")
                    && type.toLowerCase(java.util.Locale.ROOT).contains("outofmemory")) {
                return true;
            }
        }
        return false;
    }

    /**
     * The message the CLI user sees: what failed, why nothing refused it earlier, what to change.
     */
    static DeviceMemoryExhaustedException wrap(Throwable cause) {
        String message =
                DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT.prefix()
                        + "The device ran out of memory while building the execution plan.\n"
                        + "  The preflight did not refuse this beforehand: it reports CONSERVATIVE"
                        + " confidence on this backend, which is report-only.\n"
                        + "  Raise --gpu-memory, reduce the context length, or select a smaller"
                        + " quantization.\n  Backend failure: "
                        + cause;
        return new DeviceMemoryExhaustedException(message, cause);
    }
}
