package org.beehive.jllm.runtime.diagnostics;

import org.beehive.jllm.api.Experimental;

/**
 * A stable identifier for a user-reachable failure.
 *
 * <h2>Codes, not frozen prose</h2>
 *
 * <p>A catalogue test that pinned whole messages would fail on every wording improvement and teach
 * people to stop improving them. The <b>code</b> and the <b>facts</b> are the contract; the
 * sentence around them is free to get better. A caller that must branch on a failure matches the
 * code; a person reading a log gets the sentence.
 *
 * <h2>No new exception hierarchy</h2>
 *
 * <p>The prefix is {@code GPUL}. Groups: {@code CFG} configuration and selection, {@code MEM}
 * capacity, {@code MOD} model files and providers, {@code LIFE} lifecycle, {@code REQ} request
 * validation.
 */
@Experimental
public enum DiagnosticCode {

    // ── configuration and selection ──────────────────────────────────────────
    /** A device selector names a device this build cannot resolve. */
    DEVICE_SELECTOR_UNSUPPORTED("GPUL-CFG-001"),
    /** The architecture, backend, dtype and mode combination has no implementation. */
    COMBINATION_UNSUPPORTED("GPUL-CFG-002"),
    /** A capability was required that the resolved device does not report. */
    CAPABILITY_UNAVAILABLE("GPUL-CFG-003"),

    // ── capacity ─────────────────────────────────────────────────────────────
    /** The predicted device memory exceeds the configured budget. */
    DEVICE_MEMORY_INSUFFICIENT("GPUL-MEM-001"),
    /** A requested context length exceeds what the loaded model was opened with. */
    CONTEXT_LENGTH_EXCEEDED("GPUL-MEM-002"),
    /** The key/value pool has no free blocks for a reservation admission allowed. */
    KV_POOL_EXHAUSTED("GPUL-MEM-003"),

    // ── model files and providers ────────────────────────────────────────────
    /** The file is not a model this build can read, or is truncated. */
    MODEL_MALFORMED("GPUL-MOD-001"),
    /** No provider recognises this model. */
    PROVIDER_MISSING("GPUL-MOD-002"),
    /** Two providers claim one identity. */
    PROVIDER_DUPLICATE("GPUL-MOD-003"),

    // ── lifecycle ────────────────────────────────────────────────────────────
    /** A closed model or session was used. */
    USED_AFTER_CLOSE("GPUL-LIFE-001"),
    /** A model was closed while something still depended on it. */
    CLOSE_WITH_LIVE_DEPENDENTS("GPUL-LIFE-002"),

    // ── request validation ───────────────────────────────────────────────────
    /** A generation request is not valid. */
    REQUEST_INVALID("GPUL-REQ-001"),
    /** A message list is not valid for this model's conversation rules. */
    MESSAGES_INVALID("GPUL-REQ-002"),
    /** Tools were supplied to a model whose format cannot render them. */
    TOOLS_UNSUPPORTED("GPUL-REQ-003");

    private final String id;

    DiagnosticCode(String id) {
        this.id = id;
    }

    /** The stable identifier, e.g. {@code GPUL-MEM-001}. Safe to match on. */
    public String id() {
        return id;
    }

    /** {@code [GPUL-MEM-001] } — the prefix a message carries. */
    public String prefix() {
        return "[" + id + "] ";
    }

    /** Prefixes {@code message} with this code. */
    public String message(String message) {
        return prefix() + message;
    }
}
