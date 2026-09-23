package org.beehive.jitllm.runtime.kv;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import org.beehive.jitllm.runtime.backend.CapacityQuery;
import org.beehive.jitllm.runtime.backend.KvCapacity;

/**
 * Owns KV block storage and leases it to sessions.
 *
 * <p>Thread-safe: admission and release race by design, since sessions are created and closed from
 * wherever the embedder happens to be.
 *
 * <p>Closing with live leases is an error rather than a silent free — a lease is a session, and
 * sessions close first [and the same rule applies to the model].
 */
public final class KvCacheManager implements CapacityQuery, AutoCloseable {

    private final BlockPool pool;
    private final Set<KvLease> live = Collections.newSetFromMap(new IdentityHashMap<>());
    private PrefixCache prefixCache;
    private KvStorage storage;
    private long generations;
    private boolean closed;

    public KvCacheManager(BlockPool pool) {
        this.pool = pool;
    }

    /**
     * Attaches the device storage this manager's leases address.
     *
     * @throws IllegalStateException if storage is already attached, or leases are live
     */
    /**
     * Whether device storage has been attached — that is, whether leases from this manager address
     * a shared pool rather than each session allocating its own arrays.
     *
     * <p>Read by the lowered path, for which shared storage is not an option but the precondition
     * of sharing a compiled program at all.
     */
    public synchronized boolean hasStorage() {
        return storage != null;
    }

    public synchronized void attach(KvStorage storage) {
        if (this.storage != null) {
            throw new IllegalStateException(
                    "this manager already has storage attached;"
                            + " swapping it under live leases is forbidden");
        }
        if (!live.isEmpty()) {
            throw new IllegalStateException(
                    "cannot attach storage with "
                            + live.size()
                            + " live lease(s): their views would not see it");
        }
        this.storage = storage;
        storage.publishBlockTable(pool.blockTable());
    }

    /** The attached device storage, or {@code null} when this manager only does the accounting. */
    public synchronized KvStorage storage() {
        return storage;
    }

    /**
     * Sizes a manager for a context length, one sequence's worth per session.
     *
     * @param maxSessions how many sequences the pool should hold at once
     * @param contextLength tokens one sequence may reach
     * @param blockSizeTokens tokens per block
     * @param bytesPerBlock device bytes one block occupies across all layers
     */
    public static KvCacheManager sizedFor(
            int maxSessions, int contextLength, int blockSizeTokens, long bytesPerBlock) {
        int slots = Math.max(1, maxSessions);
        int blocksPerSequence = ceilDiv(contextLength, blockSizeTokens);
        // Slots and blocks are sized together here — one sequence's worth per slot — which is the
        // conservative shape. Paging's whole point is that a pool may hold fewer blocks than
        // slots * blocksPerSlot, and nothing above this method assumes otherwise.
        return new KvCacheManager(
                new BlockPool(
                        totalBlocks(slots, blocksPerSequence),
                        blocksPerSequence,
                        slots,
                        blockSizeTokens,
                        bytesPerBlock));
    }

    /**
     * Blocks for {@code sessions} sequences of {@code blocksPerSequence} each, refused rather than
     * wrapped when the product does not fit: once the session count is the caller's, an overflow
     * would silently size the pool, and every array built from it, for a nonsense capacity.
     *
     * @throws IllegalArgumentException if the block count exceeds {@link Integer#MAX_VALUE}
     */
    public static int totalBlocks(int sessions, int blocksPerSequence) {
        try {
            // The scratch block the storage adds beyond the leasable range must fit as well.
            Math.addExact(Math.multiplyExact(sessions, blocksPerSequence), 1);
            return sessions * blocksPerSequence;
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException(
                    "a key/value pool for "
                            + sessions
                            + " sessions of "
                            + blocksPerSequence
                            + " blocks exceeds the addressable block count; reduce"
                            + " maxConcurrentSessions or the context length",
                    overflow);
        }
    }

    /**
     * Leases enough blocks for {@code tokens} tokens.
     *
     * @throws IllegalStateException if the manager is closed, or the pool cannot satisfy it
     */
    public synchronized KvLease acquire(int tokens) {
        if (closed) {
            throw new IllegalStateException("this KV cache manager is closed");
        }
        int blocksNeeded = ceilDiv(tokens, pool.blockSizeTokens());
        int slot = pool.reserveSlot();
        if (slot == BlockPool.NO_SLOT) {
            throw new IllegalStateException(
                    "no free KV slot: all "
                            + pool.maxSlots()
                            + " are held. Slots are addresses in one persistent block table, so there are"
                            + " exactly as many as the table was built with");
        }
        KvLease lease;
        try {
            lease =
                    new KvLease(
                            this,
                            slot,
                            ++generations,
                            pool.reserve(slot, blocksNeeded),
                            blocksNeeded * pool.blockSizeTokens(),
                            storage);
        } catch (RuntimeException e) {
            pool.release(slot, new int[0]); // the slot must not leak when the blocks refuse
            throw e;
        }
        live.add(lease);
        publish();
        return lease;
    }

    /** What the pool has and what is left of it (D5). */
    @Override
    public synchronized KvCapacity capacity() {
        return pool.capacity();
    }

    /** The most tokens one slot can ever hold. */
    @Override
    public synchronized int tokensPerSlot() {
        return pool.blocksPerSlot() * pool.capacity().blockSizeTokens();
    }

    /** How many leases are live. */
    public synchronized int liveLeases() {
        return live.size();
    }

    /**
     * Turns on prefix sharing: sequences that begin with the same tokens read the same blocks
     * instead of prefilling them again.
     *
     * <p>Off unless enabled. It trades pool capacity for prefill: a remembered prefix keeps its
     * blocks whether or not anyone is using them, which is a good trade for served traffic with
     * repeated openings and a bad one for a pool that is already tight.
     */
    public synchronized void enablePrefixCache(int maxEntries) {
        if (prefixCache != null) {
            throw new IllegalStateException("this manager already has a prefix cache");
        }
        this.prefixCache = new PrefixCache(pool, maxEntries);
    }

    /** The prefix cache, or {@code null} if sharing was never enabled. */
    public synchronized PrefixCache prefixCache() {
        return prefixCache;
    }

    /**
     * Leases blocks for {@code tokens} tokens, reusing any cached prefix of {@code promptTokens}.
     *
     * <p>Returns a lease whose first blocks may be shared with other sequences. The count of tokens
     * already filled is on the lease as {@link KvLease#prefilledTokens()}, and the caller must skip
     * exactly that many prompt positions — feeding them again would write the same KV over blocks
     * another sequence is reading.
     */
    public synchronized KvLease acquireWithPrefix(int tokens, int[] promptTokens) {
        if (prefixCache == null) {
            return acquire(tokens);
        }
        int[] cached = prefixCache.lookup(promptTokens);
        KvLease lease = acquire(tokens);
        if (cached != null) {
            // Overwrite the freshly reserved mappings for the covered blocks with the shared ones.
            // The blocks they replace go back: they were reserved for KV that already exists.
            int[] reserved = lease.blocks();
            for (int i = 0; i < cached.length && i < reserved.length; i++) {
                pool.releaseOne(reserved[i]);
                reserved[i] = cached[i];
            }
            prefixCache.attach(lease.slot(), cached);
            lease.prefilledTokens(cached.length * pool.blockSizeTokens());
            publish();
        }
        return lease;
    }

    /**
     * Remembers this lease's prefix, so the next sequence starting with the same tokens can share
     * it. Called once the prompt has actually been prefilled.
     */
    public synchronized void rememberPrefix(KvLease lease, int[] promptTokens, int coveredTokens) {
        if (prefixCache != null) {
            prefixCache.remember(promptTokens, coveredTokens, lease.blocks());
        }
    }

    /** The pool this manager owns; its block table identity is stable for the manager's life. */
    public BlockPool pool() {
        return pool;
    }

    /**
     * Evicts a block, refusing while a live lease pins it.
     *
     * <p>The refusal is the point. Eviction that re-points a block a captured graph holds fails at
     * <b>replay</b>, not at eviction, and with {@code tornado.recover.bailout} at its default that
     * arrives as wrong output rather than an error [C1]. Refusing here is the last place the
     * problem is still reportable.
     *
     * <p>Republishes the table afterwards, since an eviction that succeeds changes what the device
     * should see.
     *
     * @throws IllegalStateException if the block is pinned by a live lease, or the manager is
     *     closed
     */
    public synchronized void evict(int block) {
        if (closed) {
            throw new IllegalStateException("this KV cache manager is closed");
        }
        pool.evict(block);
        publish();
    }

    synchronized void release(KvLease lease, int slot, int[] blocks) {
        if (live.remove(lease)) {
            pool.release(slot, blocks);
            publish();
        }
    }

    /**
     * Pushes the table to the device after it changes. Acquire and release only — never per token;
     * the hot path is the in-kernel walk.
     */
    private void publish() {
        if (storage != null) {
            storage.publishBlockTable(pool.blockTable());
        }
    }

    /**
     * @throws IllegalStateException if any lease is still live
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        if (!live.isEmpty()) {
            throw new IllegalStateException(
                    "cannot close the KV cache manager with "
                            + live.size()
                            + " live lease(s): a lease is a session, and sessions close first");
        }
        closed = true;
        if (prefixCache != null) {
            prefixCache.clear();
        }
        if (storage != null) {
            storage.close();
        }
    }

    private static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }
}
