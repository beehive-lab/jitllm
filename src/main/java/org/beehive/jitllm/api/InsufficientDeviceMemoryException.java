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
     * {@code failure}, memory running out while a GPU session's state, workspace or plan was built,
     * as this public type, with {@code failure} as the cause. The backend's translation keeps its
     * message; a raw failure gets one naming the host's direct memory or the device budget. Callers
     * ({@code DelegatingModel.newSession}, the legacy runtime's plan) check {@code
     * DeviceMemoryExhaustedException.isDeviceExhaustion} first and rethrow anything else unchanged.
     */
    static InsufficientDeviceMemoryException buildingSession(Throwable failure) {
        String message =
                failure instanceof org.beehive.jitllm.backend.tornado.DeviceMemoryExhaustedException
                        ? failure.getMessage()
                        : org.beehive.jitllm.backend.tornado.DeviceMemoryExhaustedException.message(
                                "building a session", failure);
        return new InsufficientDeviceMemoryException(message, failure, null);
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
