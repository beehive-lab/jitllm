package org.beehive.jitllm.api;

import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.memory.MemoryPlan;

/**
 * Device memory ran out while loading a model: the configured budget ({@code --gpu-memory}) or the
 * device could not hold an allocation.
 *
 * <p>Carries the memory plan predicted for the configuration when it could be computed, so a caller
 * can see the dominant component without parsing the message. The cause is the backend's own
 * failure, kept so the original byte count and allocation site are not lost.
 */
@Experimental
public class InsufficientDeviceMemoryException extends RuntimeException {

    private final transient MemoryPlan plan;

    InsufficientDeviceMemoryException(String message, Throwable cause, MemoryPlan plan) {
        super(message, cause);
        this.plan = plan;
    }

    /**
     * Runs {@code build}, which builds a session's plan, and reports the backend's device
     * out-of-memory as this public type; the backend's exception stays the cause.
     */
    static <T> T whileBuildingAPlan(java.util.function.Supplier<T> build) {
        try {
            return build.get();
        } catch (org.beehive.jitllm.backend.tornado.DeviceMemoryExhaustedException exhausted) {
            throw new InsufficientDeviceMemoryException(exhausted.getMessage(), exhausted, null);
        }
    }

    /** The stable code for this failure: {@code GPUL-MEM-001}. */
    public DiagnosticCode code() {
        return DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT;
    }

    /**
     * The memory plan predicted for this configuration: components, totals and the configured
     * budget; {@code null} when it could not be computed.
     */
    public MemoryPlan plan() {
        return plan;
    }
}
