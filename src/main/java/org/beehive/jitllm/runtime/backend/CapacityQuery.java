package org.beehive.jitllm.runtime.backend;

/** What a runtime can be asked about the key/value capacity it owns. */
public interface CapacityQuery {

    /** The pool's totals and what is left of it, at this moment. */
    KvCapacity capacity();

    /**
     * The most tokens one slot can ever hold — the ceiling a request is rejected against when it
     * could not fit even in an empty pool.
     *
     * <p>Distinct from {@link #capacity()} because it is a property of how the pool is shaped
     * rather than of how full it is: a request larger than this is not waiting for room, it is
     * impossible, and telling those two apart is the difference between a queue and a hang.
     */
    int tokensPerSlot();
}
