package org.beehive.jitllm.backend.tornado.plan;

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
    BATCH_PREFILL_DECODE
}
// @formatter:on
