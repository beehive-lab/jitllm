package org.beehive.jllm.runtime.kv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Class A. What makes this testable without a device is that sharing is entirely a question of who
 * references which block — the KV inside them is the backend's business.
 */
public class PrefixCacheTest {

    private static final int BLOCK_TOKENS = 4;

    private static KvCacheManager manager(int slots, int blocksPerSlot) {
        return new KvCacheManager(
                new BlockPool(slots * blocksPerSlot, blocksPerSlot, slots, BLOCK_TOKENS, 1024));
    }

    private static int[] tokens(int... ids) {
        return ids;
    }

    /**
     * Two sequences with the same opening read the same blocks, and the pool knows both hold them.
     */
    @Test
    public void aSharedPrefixIsTheSameBlocksInTwoSlots() {
        try (KvCacheManager manager = manager(2, 4)) {
            manager.enablePrefixCache(4);
            int[] prompt = tokens(1, 2, 3, 4, 5, 6, 7, 8);

            KvLease first = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            assertEquals("nothing cached yet", 0, first.prefilledTokens());
            manager.rememberPrefix(first, prompt, BLOCK_TOKENS); // one block's worth

            int firstBlock = manager.pool().mapped(first.slot(), 0);
            assertEquals(
                    "held by the lease and by the cache", 2, manager.pool().references(firstBlock));

            KvLease second = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            assertEquals(
                    "the shared prefix is already filled", BLOCK_TOKENS, second.prefilledTokens());
            assertEquals(
                    "and it is literally the same block",
                    firstBlock,
                    manager.pool().mapped(second.slot(), 0));
            assertEquals(
                    "three holders now: two leases and the cache",
                    3,
                    manager.pool().references(firstBlock));

            first.close();
            second.close();
            assertEquals(
                    "the cache still holds it after both leases go",
                    1,
                    manager.pool().references(firstBlock));
            assertTrue(manager.pool().isLeased(firstBlock));
        }
    }

    /** A different opening shares nothing, however similar it looks. */
    @Test
    public void aDifferentPrefixSharesNothing() {
        try (KvCacheManager manager = manager(2, 4)) {
            manager.enablePrefixCache(4);
            int[] first = tokens(1, 2, 3, 4, 9, 9, 9, 9);
            int[] other = tokens(1, 2, 3, 5, 9, 9, 9, 9); // differs at the last token

            KvLease a = manager.acquireWithPrefix(BLOCK_TOKENS * 4, first);
            manager.rememberPrefix(a, first, BLOCK_TOKENS);
            KvLease b = manager.acquireWithPrefix(BLOCK_TOKENS * 4, other);

            assertEquals("no prefix reused", 0, b.prefilledTokens());
            assertNotEquals(manager.pool().mapped(a.slot(), 0), manager.pool().mapped(b.slot(), 0));
            a.close();
            b.close();
        }
    }

    /** Only whole blocks are shareable: a partial block is not something the table can point at. */
    @Test
    public void partialBlocksAreNotRemembered() {
        BlockPool pool = new BlockPool(8, 4, 2, BLOCK_TOKENS, 1024);
        PrefixCache cache = new PrefixCache(pool, 4);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease lease = manager.acquire(BLOCK_TOKENS * 4);
            cache.remember(tokens(1, 2, 3), 3, lease.blocks()); // less than one block
            assertEquals(0, cache.size());
            assertNull(cache.lookup(tokens(1, 2, 3)));
            lease.close();
        }
    }

    /** The longest cached prefix wins: more tokens matched is more prefill saved. */
    @Test
    public void theLongestMatchIsChosen() {
        try (KvCacheManager manager = manager(3, 4)) {
            manager.enablePrefixCache(8);
            int[] prompt = tokens(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

            KvLease seed = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            manager.rememberPrefix(seed, prompt, BLOCK_TOKENS); // one block
            manager.rememberPrefix(seed, prompt, BLOCK_TOKENS * 2); // two blocks

            KvLease next = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            assertEquals(
                    "the two-block entry was preferred", BLOCK_TOKENS * 2, next.prefilledTokens());

            seed.close();
            next.close();
        }
    }

    @Test
    public void aCachedBlockUnderALiveLeaseCannotBeEvicted() {
        try (KvCacheManager manager = manager(2, 4)) {
            manager.enablePrefixCache(4);
            int[] prompt = tokens(1, 2, 3, 4, 5, 6, 7, 8);

            KvLease lease = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            manager.rememberPrefix(lease, prompt, BLOCK_TOKENS);
            int shared = manager.pool().mapped(lease.slot(), 0);

            IllegalStateException refused =
                    assertThrows(IllegalStateException.class, () -> manager.evict(shared));
            assertTrue(refused.getMessage(), refused.getMessage().contains("pinned"));
            assertTrue("and it names how many hold it", refused.getMessage().contains("2 live"));

            lease.close();
            // Still held by the cache alone, so still not evictable.
            assertThrows(IllegalStateException.class, () -> manager.evict(shared));

            manager.prefixCache().clear();
            manager.evict(shared); // now nobody holds it
            assertEquals(0, manager.pool().references(shared));
        }
    }

    /** Eviction is bounded and least-recently-used, and gives the pool its blocks back. */
    @Test
    public void theCacheIsBoundedAndReleasesWhatItDrops() {
        try (KvCacheManager manager = manager(2, 4)) {
            manager.enablePrefixCache(1);
            int[] first = tokens(1, 1, 1, 1, 0, 0, 0, 0);
            int[] second = tokens(2, 2, 2, 2, 0, 0, 0, 0);

            KvLease a = manager.acquireWithPrefix(BLOCK_TOKENS * 4, first);
            manager.rememberPrefix(a, first, BLOCK_TOKENS);
            int firstBlock = manager.pool().mapped(a.slot(), 0);
            a.close();
            assertEquals("cache holds it", 1, manager.pool().references(firstBlock));

            KvLease b = manager.acquireWithPrefix(BLOCK_TOKENS * 4, second);
            manager.rememberPrefix(b, second, BLOCK_TOKENS);

            assertEquals("bounded at one entry", 1, manager.prefixCache().size());
            assertEquals(
                    "and the evicted entry's block went back to the pool",
                    0,
                    manager.pool().references(firstBlock));
            b.close();
        }
    }

    /** Hits, misses and blocks reused are countable — the saving in the pool's own unit. */
    @Test
    public void theCacheCountsWhatItSaved() {
        try (KvCacheManager manager = manager(2, 4)) {
            manager.enablePrefixCache(4);
            int[] prompt = tokens(1, 2, 3, 4, 5, 6, 7, 8);

            KvLease a = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);
            manager.rememberPrefix(a, prompt, BLOCK_TOKENS * 2);
            KvLease b = manager.acquireWithPrefix(BLOCK_TOKENS * 4, prompt);

            PrefixCache cache = manager.prefixCache();
            assertEquals(1, cache.hits());
            assertEquals("the first lookup found nothing", 1, cache.misses());
            assertEquals("two blocks of prefill not repeated", 2, cache.blocksReused());
            a.close();
            b.close();
        }
    }

    /** Sharing a block nobody holds would hand out storage whose contents nothing guarantees. */
    @Test
    public void afreeBlockCannotBeShared() {
        BlockPool pool = new BlockPool(4, 2, 2, BLOCK_TOKENS, 1024);
        IllegalStateException refused =
                assertThrows(IllegalStateException.class, () -> pool.share(0, 0, 3));
        assertTrue(refused.getMessage(), refused.getMessage().contains("free"));
    }
}
