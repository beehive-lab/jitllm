package org.beehive.jllm.runtime.kv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class PagedPoolTest {

    private static final int BLOCK_TOKENS = 16;
    private static final long BYTES_PER_BLOCK = 4096;

    private static BlockPool pool(int totalBlocks, int blocksPerSlot, int maxSlots) {
        return new BlockPool(totalBlocks, blocksPerSlot, maxSlots, BLOCK_TOKENS, BYTES_PER_BLOCK);
    }

    /**
     * The scratch block sits one past the leasable range. An inactive slot still runs the KV
     * kernels every step, so its writes need somewhere harmless: block 0 would corrupt whoever
     * holds it, and {@code UNMAPPED} would index out of bounds.
     */
    @Test
    public void theScratchBlockIsPastTheLeasableRangeAndIsNeverHandedOut() {
        BlockPool pool = pool(4, 2, 2);

        assertEquals("one past the leasable blocks", 4, pool.scratchBlock());
        assertEquals("storage allocates it too", 5, pool.allocatedBlocks());
        assertFalse("it is not a leased block", pool.isLeased(pool.scratchBlock()));

        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease first = manager.acquire(BLOCK_TOKENS * 2);
            KvLease second = manager.acquire(BLOCK_TOKENS * 2);
            for (int block = 0; block < 4; block++) {
                assertTrue("every leasable block went to a lease", pool.isLeased(block));
            }
            assertFalse("but never the scratch block", first.holds(pool.scratchBlock()));
            assertFalse("nor to the other lease", second.holds(pool.scratchBlock()));
            first.close();
            second.close();
        }
    }

    /** It is not evictable either: every unmapped table entry is published pointing at it. */
    @Test
    public void theScratchBlockCannotBeEvicted() {
        BlockPool pool = pool(4, 2, 2);
        IllegalArgumentException refused =
                assertThrows(IllegalArgumentException.class, () -> pool.evict(pool.scratchBlock()));
        assertTrue(refused.getMessage(), refused.getMessage().contains("scratch block"));
    }

    @Test
    public void evictingAPinnedBlockIsRefused() {
        BlockPool pool = pool(4, 2, 2);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease lease = manager.acquire(BLOCK_TOKENS * 2);
            int held = pool.mapped(lease.slot(), 0);
            assertTrue("the lease really holds it", pool.isLeased(held));

            IllegalStateException refused =
                    assertThrows(IllegalStateException.class, () -> manager.evict(held));
            assertTrue(refused.getMessage(), refused.getMessage().contains("pinned"));
            assertTrue("and it is still leased afterwards", pool.isLeased(held));

            lease.close();
            // Once released it is no longer pinned, and eviction is a no-op rather than an error.
            manager.evict(held);
        }
    }

    /**
     * The published table is what the kernels read, and it must never carry {@code UNMAPPED}: that
     * value is host accounting, and as a device index it is a negative write. The translation
     * happens in the store; this asserts the contract it satisfies.
     */
    @Test
    public void unmappedEntriesArePublishedAsTheScratchBlock() {
        BlockPool pool = pool(4, 2, 2);
        TranslatingStorage storage = new TranslatingStorage(pool.scratchBlock());
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            manager.attach(storage);
            assertEquals(
                    "an empty pool publishes nothing but scratch",
                    storage.published.length,
                    countEqualTo(storage.published, pool.scratchBlock()));

            KvLease lease = manager.acquire(BLOCK_TOKENS * 2);
            int base = lease.slot() * pool.blocksPerSlot();
            assertNotEquals(
                    "the leased entries name real blocks",
                    pool.scratchBlock(),
                    storage.published[base]);
            for (int i = 0; i < storage.published.length; i++) {
                assertNotEquals(
                        "no published entry is ever UNMAPPED",
                        BlockPool.UNMAPPED,
                        storage.published[i]);
            }

            lease.close();
            assertEquals(
                    "releasing puts the slot back to scratch",
                    storage.published.length,
                    countEqualTo(storage.published, pool.scratchBlock()));
        }
    }

    private static int countEqualTo(int[] values, int wanted) {
        int n = 0;
        for (int v : values) {
            if (v == wanted) {
                n++;
            }
        }
        return n;
    }

    /** Applies the same UNMAPPED → scratch translation the device store does, and records it. */
    private static final class TranslatingStorage implements KvStorage {
        private final int scratch;
        private int[] published = new int[0];

        TranslatingStorage(int scratch) {
            this.scratch = scratch;
        }

        @Override
        public void publishBlockTable(int[] blockTable) {
            int[] copy = new int[blockTable.length];
            for (int i = 0; i < blockTable.length; i++) {
                copy[i] = blockTable[i] == BlockPool.UNMAPPED ? scratch : blockTable[i];
            }
            published = copy;
        }

        @Override
        public int blockSizeTokens() {
            return BLOCK_TOKENS;
        }

        @Override
        public int blocksPerSlot() {
            return 2;
        }

        @Override
        public long bytesPerBlock() {
            return BYTES_PER_BLOCK;
        }

        @Override
        public void close() {}
    }
}
