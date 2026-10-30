package org.beehive.jitllm.backend.tornado;

/**
 * Releases a TornadoVM execution plan. {@code close()} frees the plan's device memory; there is
 * nothing more to release, so calling {@code freeDeviceMemory()} first is not needed.
 */
public final class ExecutionPlanRelease {

    private ExecutionPlanRelease() {}

    /** Closes {@code plan}, so its device memory is freed. */
    public static void close(AutoCloseable plan) {
        try {
            plan.close();
        } catch (Exception e) {
            throw new IllegalStateException("failed to close the TornadoVM execution plan", e);
        }
    }

    /**
     * Closes {@code plan} after {@code failure} stopped its initialization. A failure to close is
     * added to {@code failure} as suppressed, so the original cause is the one reported.
     */
    static void closeAfter(Throwable failure, AutoCloseable plan) {
        try {
            plan.close();
        } catch (Exception | Error cleanup) {
            failure.addSuppressed(cleanup);
        }
    }
}
