package org.beehive.jitllm.backend.tornado.lowering;

import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * Thrown when {@code jitllm.lowering=on} names a combination that has no lowered implementation
 * [D-6].
 *
 * <p>The alternative — selecting legacy and saying nothing — is worse than a failure. A user sets
 * {@code on} precisely to measure or debug the lowered path; handing them the legacy path in
 * silence means their measurement describes the wrong code, and this project has already recorded
 * one whole accelerator gate that passed that way.
 *
 * <p>Not thrown under {@code auto}, where selecting legacy is the configured answer for an
 * unqualified combination rather than a disappointed request.
 */
public final class UnsupportedLoweringException extends RuntimeException {

    private final transient LoweringQualification.Combination combination;

    public UnsupportedLoweringException(LoweringQualification.Combination combination) {
        super(
                DiagnosticCode.COMBINATION_UNSUPPORTED.prefix()
                        + "No lowered implementation for "
                        + combination
                        + ". "
                        + LoweredPlanSelection.ENABLE_PROPERTY
                        + "=on requests the lowered path for an"
                        + " exact architecture/dtype/mode combination and will not silently fall back to"
                        + " the legacy path. Use "
                        + LoweredPlanSelection.ENABLE_PROPERTY
                        + "=auto to select legacy deliberately for combinations that are not implemented,"
                        + " or =off everywhere. Lowering currently implements modes "
                        + LoweredPlanSelection.SELECTABLE_MODES
                        + " with host-resident sampling.");
        this.combination = combination;
    }

    public LoweringQualification.Combination combination() {
        return combination;
    }
}
