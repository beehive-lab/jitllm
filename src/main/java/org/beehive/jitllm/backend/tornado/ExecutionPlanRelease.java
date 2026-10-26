package org.beehive.jitllm.backend.tornado;

/** Releases a TornadoVM execution plan whose initialization failed. */
final class ExecutionPlanRelease {

    private ExecutionPlanRelease() {}

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
