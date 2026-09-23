package org.beehive.jllm.program;

/** A phase of one inference program. */
public enum PhaseId {

    /** Ingesting prompt tokens. Skips the vocabulary projection: no logits are produced. */
    PREFILL,

    /** Producing one token. Runs every component, including the projection and sampling. */
    DECODE
}
