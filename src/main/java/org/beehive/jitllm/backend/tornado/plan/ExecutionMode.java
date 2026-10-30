package org.beehive.jitllm.backend.tornado.plan;

import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * Selects the GPU execution topology for a TornadoVM inference plan.
 *
 * <ul>
 *   <li>{@link #STANDARD} — single forward pass (activation + N layers + logits).
 *   <li>{@link #PREFILL_DECODE} — shared N+2 graph plan; prefill skips logits, decode runs all.
 *   <li>{@link #BATCH_PREFILL_DECODE} — 2N+3 graph plan; N batch-prefill graphs + N decode graphs +
 *       logits.
 * </ul>
 */
public enum ExecutionMode {
    STANDARD,
    PREFILL_DECODE,
    BATCH_PREFILL_DECODE;

    /** The mode a session running {@code policy} executes in. */
    public static ExecutionMode of(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return STANDARD;
        }
        return policy.prefillBatchSize() > 1 ? BATCH_PREFILL_DECODE : PREFILL_DECODE;
    }
}
// @formatter:on
