package org.beehive.jllm.api;

import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.memory.MemoryPlan;

/**
 * A load refused because the configuration is predicted not to fit the configured device budget.
 *
 * <p>Carries the whole {@link MemoryPlan}, so a caller can act programmatically rather than parse
 * the message.
 */
@Experimental
public class InsufficientDeviceMemoryException extends RuntimeException {

    private final transient MemoryPlan plan;

    InsufficientDeviceMemoryException(MemoryPlan plan) {
        super(buildMessage(plan));
        this.plan = plan;
    }

    /**
     * Device memory ran out during the load itself, rather than being predicted beforehand.
     *
     * <p>The preflight only <b>refuses</b> at {@code EXACT} confidence, and Metal is deliberately
     * capped at {@code CONSERVATIVE} until its own budget thresholds are measured. {@code
     * CONSERVATIVE} is report-only by definition, so on Metal the load proceeds and exhaustion
     * arrives from the runtime instead — which is what this carries, rather than a raw backend
     * exception naming a byte count and a TornadoVM flag.
     *
     * <p>Carries {@code null} for {@link #plan()}: there is no prediction to attach, because this
     * is the case where prediction did not refuse. The cause is the backend's own failure, kept so
     * the original byte count and allocation site are not lost.
     */
    InsufficientDeviceMemoryException(String message, Throwable cause) {
        super(message, cause);
        this.plan = null;
    }

    /** The stable code for this failure: {@code GPUL-MEM-001}. */
    public DiagnosticCode code() {
        return DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT;
    }

    /**
     * The plan that refused the load: components, totals, and the configured budget.
     *
     * <p>{@code null} when the exhaustion was reported by the runtime during the load rather than
     * predicted before it — see the cause for the backend's own failure.
     */
    public MemoryPlan plan() {
        return plan;
    }

    private static String buildMessage(MemoryPlan plan) {
        StringBuilder message =
                new StringBuilder(DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT.prefix())
                        .append("This configuration needs about ")
                        .append(mib(plan.predictedBudgetBytes()))
                        .append(" of device memory but the configured budget is ")
                        .append(mib(plan.configuredBudgetBytes()))
                        .append(" (short by ")
                        .append(mib(plan.predictedBudgetBytes() - plan.configuredBudgetBytes()))
                        .append(").\n");
        plan.largestComponent()
                .ifPresent(
                        c ->
                                message.append("  Dominant component: ")
                                        .append(c.name())
                                        .append(" at ")
                                        .append(mib(c.predictedBytes()))
                                        .append(
                                                c.multiplicity() > 1
                                                        ? ", of which "
                                                                + mib(c.duplicationBytes())
                                                                + " is duplication across "
                                                                + c.multiplicity()
                                                                + " allocation domains"
                                                        : "")
                                        .append('\n'));
        if (plan.duplicationBytes() > 0) {
            message.append("  Duplication across the whole plan: ")
                    .append(mib(plan.duplicationBytes()))
                    .append(
                            ". A configuration with fewer graph families needs less;"
                                    + " batched prefill binds the per-layer weights twice.\n");
        }
        message.append(
                        "  Raise -Dtornado.device.memory, reduce the context length, or select a"
                                + " smaller quantization.\n")
                .append(
                        "  This is a prediction of backend budget consumption, not a measurement of"
                                + " physical free GPU memory.\n\n")
                .append(plan.describe());
        return message.toString();
    }

    private static String mib(long bytes) {
        return String.format("%.1f MiB", bytes / 1048576.0);
    }
}
