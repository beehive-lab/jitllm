package org.beehive.jitllm.program;

/**
 * A scalar an invocation supplies, by meaning rather than by buffer.
 *
 * <p>Each is delivered by <b>writing into a persistent control array</b> at a declared offset —
 * never by binding a different array. That is what makes an invocation allocation-free and makes
 * rebinding unrepresentable rather than merely discouraged.
 */
public enum ValueId {

    /** The token being processed. */
    TOKEN,

    /** Its position in the sequence. */
    POSITION,

    /** Which block-table slot this sequence occupies. */
    BLOCK_TABLE_SLOT,

    ACTIVE_COUNT,

    /** The sampler's seed. */
    SAMPLING_SEED,

    /** The sampler's temperature. */
    SAMPLING_TEMPERATURE,

    /** The sampler's nucleus threshold. */
    SAMPLING_TOP_P
}
