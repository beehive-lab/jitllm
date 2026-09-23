package org.beehive.jitllm.runtime.kv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Blocks of KV that several sequences can read, because they begin with the same tokens.
 *
 * <p>A system prompt, a few-shot preamble, a tool schema — served traffic repeats its openings, and
 * every repetition is prefill the device has already done. This remembers the blocks holding it so
 * the next sequence starting with the same tokens can point at them instead.
 *
 * <h2>What identity means here</h2>
 *
 * <p>A cached entry is keyed by the <b>exact token sequence of a whole number of blocks</b>. Not by
 * a hash of the text, not by a prefix of arbitrary length: a block is the unit the block table can
 * point at, so a partial block is not shareable, and the tokens must match exactly because the KV
 * is a function of them.
 *
 * <p>A cached block is referenced by the cache itself, so it survives the lease that created it —
 * that is the whole point. It is released when the entry is evicted <b>and</b> no live lease still
 * names it, which the pool's reference counts decide. A block a lease holds is pinned and cannot be
 * evicted: re-pointing it would break replay of any captured graph reading it, and would do so as
 * wrong output rather than an error [C1].
 *
 * <p>Not thread-safe on its own; {@link KvCacheManager} is the synchronizing owner.
 */
public final class PrefixCache {

    /** One remembered prefix: the tokens it covers, and the blocks holding their KV. */
    private record Entry(int[] tokens, int[] blocks) {}

    private final BlockPool pool;
    private final int maxEntries;

    /** Access-ordered, so eviction takes the least recently useful entry. */
    private final Map<Long, Entry> entries;

    private long hits;
    private long misses;
    private long blocksReused;

    public PrefixCache(BlockPool pool, int maxEntries) {
        if (maxEntries <= 0) {
            throw new IllegalArgumentException(
                    "a prefix cache holds at least one entry, got " + maxEntries);
        }
        this.pool = pool;
        this.maxEntries = maxEntries;
        this.entries = new LinkedHashMap<>(16, 0.75f, true);
    }

    /**
     * The longest cached prefix of these tokens, or {@code null}.
     *
     * <p>Only whole blocks are considered, longest first: a longer match is always at least as good
     * as a shorter one, because prefill saved is monotonic in the tokens matched.
     */
    public int[] lookup(int[] tokens) {
        int blockSize = pool.blockSizeTokens();
        for (int blocks = Math.min(tokens.length / blockSize, pool.blocksPerSlot());
                blocks > 0;
                blocks--) {
            int covered = blocks * blockSize;
            Entry entry = entries.get(key(tokens, covered));
            if (entry != null && Arrays.equals(entry.tokens, Arrays.copyOf(tokens, covered))) {
                hits++;
                blocksReused += entry.blocks.length;
                return entry.blocks.clone();
            }
        }
        misses++;
        return null;
    }

    /**
     * Remembers the blocks holding the KV for a whole number of blocks' worth of tokens.
     *
     * <p>Takes a reference, so the entry outlives the lease that filled it. A prefix nobody is
     * using still costs its blocks — which is why the cache is bounded and evicts.
     */
    public void remember(int[] tokens, int coveredTokens, int[] blocks) {
        int blockSize = pool.blockSizeTokens();
        if (coveredTokens < blockSize || coveredTokens % blockSize != 0) {
            return; // partial blocks are not addressable, so not shareable
        }
        int blockCount = coveredTokens / blockSize;
        if (blockCount > blocks.length) {
            return;
        }
        long key = key(tokens, coveredTokens);
        if (entries.containsKey(key)) {
            return;
        }
        int[] shared = Arrays.copyOf(blocks, blockCount);
        for (int block : shared) {
            pool.retain(block); // the cache is itself a holder
        }
        entries.put(key, new Entry(Arrays.copyOf(tokens, coveredTokens), shared));
        evictIfNeeded();
    }

    /** Hands the cached blocks to a slot, taking a reference for the new lease. */
    public void attach(int slot, int[] blocks) {
        for (int logical = 0; logical < blocks.length; logical++) {
            pool.share(slot, logical, blocks[logical]);
        }
    }

    public long hits() {
        return hits;
    }

    public long misses() {
        return misses;
    }

    /** Blocks handed out from the cache rather than filled — the saving, in the pool's own unit. */
    public long blocksReused() {
        return blocksReused;
    }

    public int size() {
        return entries.size();
    }

    /** Drops every entry, releasing the cache's own references. Live leases keep theirs. */
    public void clear() {
        for (Entry entry : List.copyOf(entries.values())) {
            releaseEntry(entry);
        }
        entries.clear();
    }

    private void evictIfNeeded() {
        while (entries.size() > maxEntries) {
            var oldest = entries.entrySet().iterator().next();
            releaseEntry(oldest.getValue());
            entries.remove(oldest.getKey());
        }
    }

    /**
     * Gives up the cache's references. A block a live lease still names stays leased — the pool's
     * counts decide, which is what stops eviction pulling storage from under a running sequence.
     */
    private void releaseEntry(Entry entry) {
        for (int block : entry.blocks) {
            pool.releaseOne(block);
        }
    }

    /** A content key over the covered tokens. Collisions are caught by the exact comparison. */
    private static long key(int[] tokens, int covered) {
        long h = 1125899906842597L;
        h = h * 31 + covered;
        for (int i = 0; i < covered; i++) {
            h = h * 31 + tokens[i];
        }
        return h;
    }

    /** Snapshot of what the cache is holding, for metrics and tests. */
    public List<int[]> cachedBlockSets() {
        List<int[]> out = new ArrayList<>();
        for (Entry entry : entries.values()) {
            out.add(entry.blocks.clone());
        }
        return out;
    }
}
