package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Locale;

/**
 * What {@code jitllm.lowering} asks for [D-6].
 *
 * <p>It was a boolean opt-in while exactly one tuple was implemented and none was qualified. With a
 * qualification table there are three distinct questions — "use what is proven", "use it anyway, I
 * am gathering evidence", and "give me the old path" — and a boolean can only answer two.
 *
 * <p><b>One input, not two.</b> A second flag would let the two disagree, and a user reading {@code
 * -Djitllm.lowering=true -Djitllm.lowering.force=false} could not tell which wins.
 */
public enum LoweringMode {

    /**
     * Lower the combinations the qualification table names; select legacy for everything else,
     * deliberately and silently. Unqualified is configured behaviour, not a failure.
     */
    AUTO,

    /**
     * Lower an implemented combination even if it is not yet qualified — the diagnostic and
     * evidence-gathering setting, and how a combination earns its way into the table.
     *
     * <p>An <b>unimplemented</b> combination throws instead of quietly running legacy: a user who
     * asked for lowering and silently got the old path would measure the old path and record it as
     * the new one, which is the false-green this project has already paid for.
     */
    ON,

    /** Always legacy. The rollback, for one compatibility window. */
    OFF;

    /**
     * Parses the property value.
     *
     * <p>{@code true} maps to {@link #ON} and {@code false} to {@link #OFF}, so every existing
     * invocation, script and test keeps working unchanged. An unset value is the caller's default.
     *
     * @throws IllegalArgumentException on a value that is neither a mode nor a boolean — a typo
     *     must not silently degrade to a path the user did not choose
     */
    public static LoweringMode parse(String value, LoweringMode ifUnset) {
        if (value == null || value.isBlank()) {
            return ifUnset;
        }
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "auto" -> AUTO;
            case "on", "true" -> ON;
            case "off", "false" -> OFF;
            default ->
                    throw new IllegalArgumentException(
                            "Unrecognised "
                                    + LoweredPlanSelection.ENABLE_PROPERTY
                                    + " value '"
                                    + value
                                    + "'. Expected auto, on or off (true and false are accepted as on and"
                                    + " off). Refusing to guess: the wrong guess silently selects a"
                                    + " different execution path.");
        };
    }
}
