package org.beehive.jllm.runtime.kv;

/**
 * A session's claim on KV storage: the blocks it may write, and nothing more.
 *
 * <p>A session <b>holds</b> a lease; it does not own storage [Rule 7]. That is what lets blocks
 * outlive a session and be shared between sequences once prefix caching exists, and what keeps the
 * cache off the model.
 *
 * <p>Blocks named by a live lease are <b>pinned</b>: the pool will not hand them to anyone else
 * until the lease is closed. Closing is idempotent, and a closed lease answers nothing — using one
 * is a bug that should surface here rather than as a wrong answer later.
 *
 * <p>Not thread-safe: a lease belongs to one session, and a session is one sequence.
 */
public final class KvLease implements AutoCloseable {

    private final KvCacheManager manager;
    private final int[] blocks;
    private final int tokenCapacity;
    private final int slot;
    private final long generation;
    private final KvStorage storage;
    private int prefilledTokens;
    private boolean closed;

    KvLease(
            KvCacheManager manager,
            int slot,
            long generation,
            int[] blocks,
            int tokenCapacity,
            KvStorage storage) {
        this.manager = manager;
        this.slot = slot;
        this.generation = generation;
        this.blocks = blocks;
        this.tokenCapacity = tokenCapacity;
        this.storage = storage;
    }

    /**
     * This lease's range in the block table: the kernels index it as {@code slot * blocksPerSlot +
     * logicalBlock}.
     *
     * <p><b>An address, not an identity.</b> Slots are reused as leases come and go, so a slot
     * number says where to look and never who is looking — that is {@link #generation()}'s job.
     * Conflating them would let a stale lease read whatever now occupies its old slot.
     */
    public int slot() {
        ensureOpen();
        return slot;
    }

    /**
     * Monotonic per manager, and never reused. Two leases that held the same slot at different
     * times differ here, which is what makes "is this view still valid" answerable.
     */
    public long generation() {
        return generation;
    }

    /** The device storage this lease addresses, or {@code null} on a CPU-only runtime. */
    public KvStorage storage() {
        ensureOpen();
        return storage;
    }

    /** How many tokens this lease has room for. */
    public int tokenCapacity() {
        ensureOpen();
        return tokenCapacity;
    }

    /** How many blocks back this lease. */
    public int blockCount() {
        ensureOpen();
        return blocks.length;
    }

    /** Whether this lease covers the given block. */
    public boolean holds(int block) {
        ensureOpen();
        for (int held : blocks) {
            if (held == block) {
                return true;
            }
        }
        return false;
    }

    /**
     * Tokens whose KV this lease inherited from the prefix cache and must not write again.
     *
     * <p>Writing them again would put identical bytes into blocks another sequence is reading —
     * harmless today, and exactly the assumption that stops being true the moment anything writes a
     * position twice with different content.
     */
    public int prefilledTokens() {
        return prefilledTokens;
    }

    void prefilledTokens(int tokens) {
        this.prefilledTokens = tokens;
    }

    public boolean isClosed() {
        return closed;
    }

    /** Returns the blocks to the pool. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        manager.release(this, slot, blocks);
    }

    int[] blocks() {
        return blocks;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "this KV lease is closed; its blocks belong to the pool again");
        }
    }
}
