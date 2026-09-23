package org.beehive.jllm.program;

/**
 * Something the caller may read once an invocation completes.
 *
 * <p>A result names what may be read and from where; it is not a buffer the caller owns. The array
 * it is read from is a program-fixed binding like any other.
 */
public enum ResultId {

    /** The vocabulary distribution. A tensor result: its representation comes from its carrier. */
    LOGITS,

    /** The chosen token, when sampling ran on the device. A scalar result. */
    SAMPLED_TOKEN
}
