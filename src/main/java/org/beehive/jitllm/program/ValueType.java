package org.beehive.jitllm.program;

/**
 * The type of a scalar an invocation supplies or reads back.
 *
 * <p>Separate from {@code DataType} on purpose. {@code DataType} describes how a <b>tensor's</b>
 * values are represented where the engine computes with them; it has no honest value for a token
 * identifier or a random seed, and describing a sampled token as {@code F32} would be simply wrong.
 */
public enum ValueType {

    /** A 32-bit integer: a token identifier, a position, a slot index, an active count. */
    I32,

    /** A 64-bit integer: a sampling seed. */
    I64,

    /** A 32-bit float: a temperature, a top-p threshold. */
    F32
}
