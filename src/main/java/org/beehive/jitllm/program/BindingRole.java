package org.beehive.jllm.program;

/**
 * What a program-fixed device array is for.
 *
 * <p>Every device array bound into a captured graph has one of these — weights and key/value
 * storage as much as workspace, control and result arrays. Inputs and outputs are not exempt: an
 * output array is a device array in a captured graph, and its identity is as fixed as a weight's
 */
public enum BindingRole {

    /** Model weights. */
    WEIGHT,

    /** Retained key/value storage. */
    KV_POOL,

    /** The block table mapping logical positions to pool blocks. */
    BLOCK_TABLE,

    /** Activations and scratch that live for one invocation but are allocated once. */
    WORKSPACE,

    /** Staging buffers a batched phase fills before execution. */
    STAGING,

    /** Persistent arrays invocation values are written into. */
    CONTROL,

    /** Arrays host-visible results are read out of. */
    RESULT
}
