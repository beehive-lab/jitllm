package org.beehive.jllm.runtime.backend;

/**
 * Which implementation actually executed a session — the lowered program, or the legacy plan.
 *
 * <p>Neutral by construction: an enum in {@code runtime.backend}, naming neither TornadoVM nor an
 * implementation package, so the generation layer and the metrics seam can both report it.
 */
public enum ExecutionPath {

    /** The lowered program built from the operation vocabulary. */
    LOWERED,

    /** The legacy hand-built plan. */
    LEGACY;

    /** Lower-case name, for metrics payloads and log lines. */
    public String reportName() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
