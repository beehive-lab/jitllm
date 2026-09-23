package org.beehive.jitllm.runtime.kv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.runtime.backend.CapacityQuery;
import org.beehive.jitllm.runtime.backend.KvCapacity;
import org.junit.Test;

public class KvCacheManagerTest {

    private static final int BLOCK_TOKENS = 16;
    private static final long BYTES_PER_BLOCK = 4096;

    private static KvCacheManager manager(int totalBlocks) {
        return new KvCacheManager(new BlockPool(totalBlocks, BLOCK_TOKENS, BYTES_PER_BLOCK));
    }

    @Test
    public void severalLeasesAreLiveAtOnce() {
        try (KvCacheManager manager = manager(16)) {
            KvLease first = manager.acquire(32);
            KvLease second = manager.acquire(32);
            KvLease third = manager.acquire(32);

            assertEquals(3, manager.liveLeases());
            assertEquals("each lease got its own blocks", 6, manager.capacity().usedBlocks());

            first.close();
            second.close();
            third.close();
            assertEquals(0, manager.liveLeases());
        }
    }

    /** Blocks a live lease names are pinned: nobody else may be given them. */
    @Test
    public void aLeasedBlockIsNeverHandedToAnotherLease() {
        try (KvCacheManager manager = manager(4)) {
            KvLease held = manager.acquire(BLOCK_TOKENS * 2);
            KvLease other = manager.acquire(BLOCK_TOKENS * 2);

            for (int block = 0; block < 4; block++) {
                assertNotEquals(
                        "a block cannot be in both leases at once",
                        held.holds(block),
                        other.holds(block));
            }
            held.close();
            other.close();
        }
    }

    /**
     * The block table is one persistent array [C1]: updating it must not replace it, or every
     * captured CUDA graph would be replaying against a stale address.
     */
    @Test
    public void theBlockTableKeepsItsIdentityAcrossUpdates() {
        BlockPool pool = new BlockPool(8, BLOCK_TOKENS, BYTES_PER_BLOCK);
        int[] table = pool.blockTable();

        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease lease = manager.acquire(BLOCK_TOKENS * 4);
            assertSame("acquiring must not reallocate the table", table, pool.blockTable());
            lease.close();
            assertSame("releasing must not reallocate the table", table, pool.blockTable());
        }
        assertSame("closing must not reallocate the table", table, pool.blockTable());
    }

    @Test
    public void theTableIsSlotMajorAndALeaseOccupiesItsOwnRange() {
        BlockPool pool = new BlockPool(8, 4, 2, BLOCK_TOKENS, BYTES_PER_BLOCK);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease first = manager.acquire(BLOCK_TOKENS * 4);
            KvLease second = manager.acquire(BLOCK_TOKENS * 4);

            assertNotEquals("two live leases never share a slot", first.slot(), second.slot());
            for (int logical = 0; logical < 4; logical++) {
                assertTrue(
                        "every logical block of a live lease is mapped",
                        pool.mapped(first.slot(), logical) != BlockPool.UNMAPPED);
                assertNotEquals(
                        "and to a block the other lease does not hold",
                        pool.mapped(first.slot(), logical),
                        pool.mapped(second.slot(), logical));
            }

            int slot = first.slot();
            first.close();
            for (int logical = 0; logical < 4; logical++) {
                assertEquals(
                        "a released slot names no block",
                        BlockPool.UNMAPPED,
                        pool.mapped(slot, logical));
                assertTrue(
                        "while the other lease is untouched",
                        pool.mapped(second.slot(), logical) != BlockPool.UNMAPPED);
            }
            second.close();
        }
    }

    /**
     * A slot is an address and a generation is an identity. Slots are reused; generations are not,
     * which is what makes "is this view still the one that owns these blocks" answerable.
     */
    @Test
    public void slotsAreReusedAndGenerationsAreNot() {
        try (KvCacheManager manager = manager(16)) {
            KvLease first = manager.acquire(BLOCK_TOKENS);
            int slot = first.slot();
            long generation = first.generation();
            first.close();

            KvLease second = manager.acquire(BLOCK_TOKENS);
            assertEquals("the freed slot is handed out again", slot, second.slot());
            assertNotEquals("but it is not the same lease", generation, second.generation());
            second.close();
        }
    }

    /** Storage is attached once, and never swapped under live leases. */
    @Test
    public void storageIsAttachedOnceAndSeesEveryTableChange() {
        RecordingStorage storage = new RecordingStorage();
        try (KvCacheManager manager = manager(16)) {
            manager.attach(storage);
            assertEquals("attaching publishes the initial table", 1, storage.publishes);

            KvLease lease = manager.acquire(BLOCK_TOKENS);
            assertEquals("acquiring publishes", 2, storage.publishes);
            assertSame("the lease carries the storage it addresses", storage, lease.storage());

            lease.close();
            assertEquals("releasing publishes", 3, storage.publishes);

            assertThrows(IllegalStateException.class, () -> manager.attach(new RecordingStorage()));
        }
    }

    /** Counts publishes; the real store copies the table into a device array. */
    private static final class RecordingStorage implements org.beehive.jitllm.runtime.kv.KvStorage {
        int publishes;

        @Override
        public void publishBlockTable(int[] blockTable) {
            publishes++;
        }

        @Override
        public int blockSizeTokens() {
            return BLOCK_TOKENS;
        }

        @Override
        public int blocksPerSlot() {
            return 1;
        }

        @Override
        public long bytesPerBlock() {
            return BYTES_PER_BLOCK;
        }

        @Override
        public void close() {}
    }

    /**
     * Growing capacity invalidates captured graphs, so the pool refuses rather than reallocating
     * silently. Recapture is an explicit act, not something a lease can trigger.
     */
    @Test
    public void exhaustingThePoolIsRefusedRatherThanGrown() {
        BlockPool pool = new BlockPool(2, BLOCK_TOKENS, BYTES_PER_BLOCK);
        int[] table = pool.blockTable();
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            KvLease all = manager.acquire(BLOCK_TOKENS * 2);

            IllegalStateException refused =
                    assertThrows(IllegalStateException.class, () -> manager.acquire(BLOCK_TOKENS));
            assertTrue(refused.getMessage(), refused.getMessage().contains("exhausted"));
            assertSame("a refused acquire must not have grown the pool", table, pool.blockTable());
            assertEquals(2, pool.totalBlocks());

            all.close();
        }
    }

    @Test
    public void capacityIsReportedInBlocksAndBytes() {
        try (KvCacheManager manager = manager(10)) {
            KvCapacity empty = manager.capacity();
            assertEquals(10, empty.totalBlocks());
            assertEquals(10, empty.freeBlocks());
            assertEquals(0, empty.usedBlocks());
            assertEquals(10 * BYTES_PER_BLOCK, empty.totalBytes());
            assertEquals(0, empty.usedBytes());
            assertEquals(10L * BLOCK_TOKENS, empty.totalTokens());

            KvLease lease = manager.acquire(BLOCK_TOKENS * 3);
            KvCapacity used = manager.capacity();
            assertEquals(3, used.usedBlocks());
            assertEquals(7, used.freeBlocks());
            assertEquals(3 * BYTES_PER_BLOCK, used.usedBytes());

            lease.close();
            assertEquals("closing a lease returns its blocks", 10, manager.capacity().freeBlocks());
        }
    }

    /** A partial block is still a block: 17 tokens needs two of them. */
    @Test
    public void aPartialBlockStillCostsAWholeBlock() {
        try (KvCacheManager manager = manager(8)) {
            KvLease lease = manager.acquire(BLOCK_TOKENS + 1);
            assertEquals(2, lease.blockCount());
            assertEquals(2 * BLOCK_TOKENS, lease.tokenCapacity());
            lease.close();
        }
    }

    /** A lease is a session; sessions close first [mirroring for the model]. */
    @Test
    public void closingWithLiveLeasesIsAnError() {
        KvCacheManager manager = manager(4);
        KvLease live = manager.acquire(BLOCK_TOKENS);

        IllegalStateException failure = assertThrows(IllegalStateException.class, manager::close);
        assertTrue(failure.getMessage(), failure.getMessage().contains("live lease"));

        live.close();
        manager.close(); // now fine
    }

    @Test
    public void closingALeaseIsIdempotentAndAClosedLeaseAnswersNothing() {
        try (KvCacheManager manager = manager(4)) {
            KvLease lease = manager.acquire(BLOCK_TOKENS);
            lease.close();
            lease.close(); // idempotent

            assertTrue(lease.isClosed());
            assertEquals(
                    "a double close must not return the blocks twice",
                    4,
                    manager.capacity().freeBlocks());
            assertThrows(IllegalStateException.class, lease::blockCount);
        }
    }

    @Test
    public void sizingCoversOneSequencePerSession() {
        try (KvCacheManager manager =
                KvCacheManager.sizedFor(4, 512, BLOCK_TOKENS, BYTES_PER_BLOCK)) {
            // 512 tokens is 32 blocks; four sessions is 128.
            assertEquals(128, manager.capacity().totalBlocks());
            assertFalse("a fresh pool has nothing leased", manager.capacity().usedBlocks() > 0);
        }
    }

    /**
     * {@code tokensPerSlot} moved here from {@code Scheduler}, which assembled it from two separate
     * reads of the manager and its pool. It is a property of the pool's shape, so it must not move
     * when blocks are leased out — that is the difference between "there is no room now" and "this
     * could never fit", and admission tells a queue from a rejection with it.
     */
    @Test
    public void theManagerAnswersTheCapacityContractAndTokensPerSlotIsShapeNotOccupancy() {
        BlockPool pool = new BlockPool(8, 4, 2, 16, 1024);
        try (KvCacheManager manager = new KvCacheManager(pool)) {
            CapacityQuery query = manager;

            int before = query.tokensPerSlot();
            assertEquals(4 * 16, before);
            assertEquals(8, query.capacity().totalBlocks());
            assertEquals(8, query.capacity().freeBlocks());

            KvLease lease = manager.acquire(32);
            assertEquals("occupancy changed", 6, query.capacity().freeBlocks());
            assertEquals("shape did not", before, query.tokensPerSlot());
            lease.close();
        }
    }

    @Test
    public void totalBlocksRefusesAnOverflowingSessionCount() {
        assertEquals(8 * 512, KvCacheManager.totalBlocks(8, 512));
        assertThrows(
                IllegalArgumentException.class,
                () -> KvCacheManager.totalBlocks(Integer.MAX_VALUE / 2, 3));
        // The scratch block beyond the leasable range must fit as well.
        assertThrows(
                IllegalArgumentException.class,
                () -> KvCacheManager.totalBlocks(1, Integer.MAX_VALUE));
        assertThrows(
                IllegalArgumentException.class,
                () -> KvCacheManager.sizedFor(Integer.MAX_VALUE, 8192, 16, 1024));
    }
}
