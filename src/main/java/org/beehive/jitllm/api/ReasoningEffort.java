package org.beehive.jitllm.api;

/**
 * How much a model that reasons should reason, for a family whose template can say so.
 *
 * <p>Qwen3.8's chat template writes an instruction at the top of the system turn whenever thinking
 * is on — {@code "Reasoning effort is set to xhigh. Please think carefully through the task, …"} by
 * default, a brief-thinking one for {@code low}, and none for {@code medium}. The instruction
 * changes how long the model reasons several-fold, so a runtime that leaves it out answers
 * differently from one that renders the template as written.
 *
 * <h2>Where it lives</h2>
 *
 * <p>The same shape as {@link ThinkingMode}: on {@link ModelOptions} as a model default and on
 * {@link SessionOptions} as a per-session override, resolved once when the session opens. It only
 * applies while the reasoning phase is on; asking for an explicit effort on a session with thinking
 * {@linkplain ThinkingMode#DISABLED disabled} is rejected, as is asking a family whose template has
 * no such control ({@link ModelCapabilities#reasoningEffortControl()}).
 */
@Experimental
public enum ReasoningEffort {

    /**
     * Whatever the family's template does when it is not told: {@code xhigh} for Qwen3.8, nothing
     * for a family without the control. The default.
     */
    DEFAULT,

    /** Think carefully, validate assumptions and consider alternatives. Qwen3.8's own default. */
    XHIGH,

    /** No instruction: the model's untold behaviour. */
    MEDIUM,

    /** Keep the reasoning brief and move to the conclusion. */
    LOW;

    /** Whether this effort asks the model for anything, as opposed to leaving the family alone. */
    public boolean isExplicit() {
        return this != DEFAULT;
    }
}
