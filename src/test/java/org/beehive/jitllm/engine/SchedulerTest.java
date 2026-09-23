package org.beehive.jitllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.beehive.jitllm.runtime.kv.BlockPool;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.junit.Test;

public class SchedulerTest {

    private static final int BLOCK_TOKENS = 16;
    private static final long BYTES_PER_BLOCK = 4096;

    /** A pool of {@code slots × blocksPerSlot} blocks, sized like the manager's own factory. */
    private static KvCacheManager manager(int slots, int blocksPerSlot) {
        return new KvCacheManager(
                new BlockPool(
                        slots * blocksPerSlot,
                        blocksPerSlot,
                        slots,
                        BLOCK_TOKENS,
                        BYTES_PER_BLOCK));
    }

    /** A pool whose block count is not a multiple of the slot length — the interesting shape. */
    private static KvCacheManager pool(int totalBlocks, int blocksPerSlot, int slots) {
        return new KvCacheManager(
                new BlockPool(totalBlocks, blocksPerSlot, slots, BLOCK_TOKENS, BYTES_PER_BLOCK));
    }

    /**
     * Terminates everything still running, so the manager can close.
     *
     * <p>Closing with a live lease is an error by design — a lease is a session and sessions close
     * first — so a test that leaves one has not finished its own scenario.
     */
    private static void drain(Scheduler scheduler) {
        for (ScheduledRequest request : scheduler.activeSlots()) {
            if (request != null) {
                scheduler.complete(request);
            }
        }
    }

    // ── construction ────────────────────────────────────────────────────────────────────────────

    /**
     * The queue bound is required and positive. A wrong default is worse than an absent one, so the
     * library refuses to have one at all.
     */
    @Test
    public void constructionRequiresAPositiveQueueBound() {
        KvCacheManager manager = manager(2, 4);

        IllegalArgumentException missing =
                assertThrows(IllegalArgumentException.class, () -> new Scheduler(manager, 2, 0));
        assertTrue(missing.getMessage(), missing.getMessage().contains("maxQueuedRequests"));
        assertTrue(
                "says why there is no default",
                missing.getMessage().contains("not one the library guesses"));

        assertThrows(IllegalArgumentException.class, () -> new Scheduler(manager, 2, -1));
        assertThrows(IllegalArgumentException.class, () -> new Scheduler(manager, 0, 8));
    }

    // ── rejection, and what is deliberately not a rejection ─────────────────────────────────────

    /** Capacity shortfall queues; it is a normal state, not an error. */
    @Test
    public void capacityShortfallQueuesRatherThanRejecting() {
        try (KvCacheManager manager = manager(1, 4)) {
            Scheduler scheduler = new Scheduler(manager, 1, 8);

            ScheduledRequest first = scheduler.submit(BLOCK_TOKENS * 4);
            ScheduledRequest second = scheduler.submit(BLOCK_TOKENS * 4);
            scheduler.admit();

            assertEquals(RequestState.RUNNING, first.state());
            assertEquals(
                    "the second waits rather than being refused",
                    RequestState.QUEUED,
                    second.state());
            assertNull(second.rejectionReason());

            scheduler.complete(first);
        }
    }

    /** A budget no slot can ever hold is refused at once — waiting would never help. */
    @Test
    public void aRequestThatCanNeverFitIsRejectedImmediately() {
        try (KvCacheManager manager = manager(2, 4)) {
            Scheduler scheduler = new Scheduler(manager, 2, 8);

            ScheduledRequest tooBig = scheduler.submit(scheduler.tokensPerSlot() + 1);

            assertEquals(RequestState.REJECTED, tooBig.state());
            assertEquals(RejectionReason.CANNOT_EVER_FIT, tooBig.rejectionReason());
            assertEquals("and it never entered the queue", 0, scheduler.queueDepth());
        }
    }

    /** The queue bound is backpressure the caller can see, not a hidden wait. */
    @Test
    public void theQueueIsBoundedAndOverflowIsQueueFull() {
        try (KvCacheManager manager = manager(1, 4)) {
            Scheduler scheduler = new Scheduler(manager, 1, 2);

            ScheduledRequest a = scheduler.submit(BLOCK_TOKENS);
            ScheduledRequest b = scheduler.submit(BLOCK_TOKENS);
            ScheduledRequest overflow = scheduler.submit(BLOCK_TOKENS);

            assertEquals(RequestState.QUEUED, a.state());
            assertEquals(RequestState.QUEUED, b.state());
            assertEquals(RequestState.REJECTED, overflow.state());
            assertEquals(RejectionReason.QUEUE_FULL, overflow.rejectionReason());
            assertEquals("the bound is a bound", 2, scheduler.queueDepth());
        }
    }

    @Test
    public void malformedAndShutdownAreRejectedToo() {
        try (KvCacheManager manager = manager(1, 4)) {
            Scheduler scheduler = new Scheduler(manager, 1, 4);

            assertEquals(RejectionReason.MALFORMED, scheduler.submit(0).rejectionReason());
            assertEquals(RejectionReason.MALFORMED, scheduler.submit(-5).rejectionReason());

            ScheduledRequest queued = scheduler.submit(BLOCK_TOKENS);
            List<ScheduledRequest> rejected = scheduler.shutdown();

            assertEquals("shutdown rejects what was waiting", List.of(queued), rejected);
            assertEquals(RejectionReason.SHUTDOWN, queued.rejectionReason());
            assertEquals(
                    RejectionReason.SHUTDOWN, scheduler.submit(BLOCK_TOKENS).rejectionReason());
        }
    }

    // ── the scheduling rules themselves ──────────────────────────────────────────────────────────

    /**
     * <b>Strict FCFS.</b> A head request that does not fit keeps a later one that would fit waiting
     * behind it — <i>even when a slot is free</i>. This is the rule that costs throughput and buys
     * a starvation guarantee, and it is asserted rather than assumed precisely because the tempting
     * implementation — scan on to fill the idle slot — looks better and defers the large request
     * indefinitely under sustained load.
     */
    @Test
    public void aHeadRequestThatDoesNotFitBlocksASmallerOneBehindIt() {
        // 5 blocks, slots up to 4 blocks each, B = 2. Deliberately not a multiple: the scenario
        // needs a free slot and insufficient blocks at the same time.
        try (KvCacheManager manager = pool(5, 4, 2)) {
            Scheduler scheduler = new Scheduler(manager, 2, 8);

            ScheduledRequest occupier = scheduler.submit(BLOCK_TOKENS * 4);
            scheduler.admit();
            assertEquals(RequestState.RUNNING, occupier.state());
            assertEquals("one block left", 1, manager.capacity().freeBlocks());

            ScheduledRequest big = scheduler.submit(BLOCK_TOKENS * 4); // needs 4, only 1 free
            ScheduledRequest small = scheduler.submit(BLOCK_TOKENS); // needs 1 — would fit

            List<ScheduledRequest> admitted = scheduler.admit();

            assertTrue("a slot was free and still nothing was admitted", admitted.isEmpty());
            assertEquals("the head waits for capacity", RequestState.QUEUED, big.state());
            assertEquals(
                    "and the one behind it waits too, though it would have fit",
                    RequestState.QUEUED,
                    small.state());
            assertEquals(1, scheduler.runningCount());

            // Once the head can be served, the queue drains in order.
            scheduler.complete(occupier);
            scheduler.admit();
            assertEquals(RequestState.RUNNING, big.state());
            assertEquals(RequestState.RUNNING, small.state());
            assertTrue("and in submission order", big.slot() != small.slot());

            drain(scheduler);
        }
    }

    /**
     * B is fixed: never more than B running, and empty slots are inactive rather than errors.
     *
     * <p>Note what {@code activeSlots()} is indexed by — the <b>KV slot</b>, not an independent
     * batch counter. The batched kernels read the block table at {@code batchIndex *
     * blocksPerSlot}, so the two have to be the same number; a second allocator would be a second
     * chance to disagree, and the disagreement reads as another sequence's KV rather than as an
     * error.
     */
    @Test
    public void neverMoreThanBRunAtOnce() {
        try (KvCacheManager manager = manager(4, 2)) {
            Scheduler scheduler = new Scheduler(manager, 2, 8); // B = 2, pool holds 4 sequences

            for (int i = 0; i < 4; i++) {
                scheduler.submit(BLOCK_TOKENS * 2);
            }
            List<ScheduledRequest> admitted = scheduler.admit();

            assertEquals("B bounds the batch, not the pool", 2, admitted.size());
            assertEquals(2, scheduler.runningCount());
            assertEquals(2, scheduler.queueDepth());

            // The slot array is indexed by KV slot, so it is as wide as the pool, not as B.
            // What B bounds is how many of those slots may be occupied at once.
            ScheduledRequest[] slots = scheduler.activeSlots();
            assertEquals("indexed by KV slot", 4, slots.length);
            int occupied = 0;
            for (ScheduledRequest request : slots) {
                if (request != null) {
                    occupied++;
                }
            }
            assertEquals("and B bounds how many are occupied", 2, occupied);

            drain(scheduler);
        }
    }

    /** A slot and its blocks come back at a terminal state, and the next request takes them. */
    @Test
    public void aTerminalRequestGivesBackItsSlotAndBlocks() {
        try (KvCacheManager manager = manager(2, 2)) {
            Scheduler scheduler = new Scheduler(manager, 1, 8);

            ScheduledRequest first = scheduler.submit(BLOCK_TOKENS * 2);
            ScheduledRequest second = scheduler.submit(BLOCK_TOKENS * 2);
            scheduler.admit();

            int slot = first.slot();
            assertEquals(RequestState.QUEUED, second.state());
            assertEquals(1, manager.liveLeases());

            scheduler.complete(first);
            assertEquals("the lease went back", 0, manager.liveLeases());
            assertEquals(-1, first.slot());
            assertNull(first.lease());

            scheduler.admit();
            assertEquals(RequestState.RUNNING, second.state());
            assertEquals("and the slot was reused", slot, second.slot());

            drain(scheduler);
        }
    }

    /**
     * The reservation is whole and durable: an admitted request keeps its blocks until it
     * terminates, so nothing that arrives later can take them.
     */
    @Test
    public void anAdmittedRequestKeepsItsBlocksUntilItTerminates() {
        try (KvCacheManager manager = manager(2, 4)) {
            Scheduler scheduler = new Scheduler(manager, 2, 8);

            ScheduledRequest running = scheduler.submit(BLOCK_TOKENS * 4);
            scheduler.admit();
            int reservedBlocks = running.lease().blockCount();

            for (int i = 0; i < 4; i++) {
                scheduler.submit(BLOCK_TOKENS * 4);
                scheduler.admit();
            }

            assertEquals(
                    "its reservation is untouched by later arrivals",
                    reservedBlocks,
                    running.lease().blockCount());
            assertEquals(RequestState.RUNNING, running.state());

            drain(scheduler);
        }
    }

    // ── failure ─────────────────────────────────────────────────────────────────────────────────

    /** Post-admission exhaustion is a broken invariant, and says so. */
    @Test
    public void exhaustionAfterAdmissionIsAnInvariantViolationNotATruncation() {
        try (KvCacheManager manager = manager(1, 4)) {
            Scheduler scheduler = new Scheduler(manager, 1, 4);
            ScheduledRequest request = scheduler.submit(BLOCK_TOKENS * 4);
            scheduler.admit();

            scheduler.failOnExhaustion(request, "pool reported 0 free while running");

            assertEquals(RequestState.FAILED, request.state());
            assertTrue(request.failure() instanceof IllegalStateException);
            assertTrue(
                    request.failure().getMessage(),
                    request.failure().getMessage().contains("broken invariant"));
            assertEquals("and it still released what it held", 0, manager.liveLeases());
        }
    }

    /** A failing request releases its resources like any other terminal one. */
    @Test
    public void aFailedRequestReleasesItsLeaseAndKeepsItsCause() {
        try (KvCacheManager manager = manager(1, 2)) {
            Scheduler scheduler = new Scheduler(manager, 1, 4);
            ScheduledRequest request = scheduler.submit(BLOCK_TOKENS * 2);
            scheduler.admit();

            RuntimeException thrown = new RuntimeException("callback threw");
            scheduler.fail(request, thrown);

            assertEquals(RequestState.FAILED, request.state());
            assertSame("the cause is retained for the handle", thrown, request.failure());
            assertEquals(0, manager.liveLeases());
        }
    }

    // ── cancellation ────────────────────────────────────────────────────────────────────────────

    @Test
    public void cancellingIsIdempotentFromEitherState() {
        try (KvCacheManager manager = manager(2, 2)) {
            Scheduler scheduler = new Scheduler(manager, 1, 4);

            ScheduledRequest queued = scheduler.submit(BLOCK_TOKENS * 2);
            ScheduledRequest running = scheduler.submit(BLOCK_TOKENS * 2);
            scheduler.admit();
            assertEquals(RequestState.RUNNING, queued.state()); // first submitted, first admitted

            scheduler.cancel(running); // still queued
            assertEquals(RequestState.CANCELLED, running.state());
            assertEquals(0, scheduler.queueDepth());

            scheduler.cancel(queued); // running
            assertEquals(RequestState.CANCELLED, queued.state());
            assertEquals(0, manager.liveLeases());

            scheduler.cancel(queued); // again: not an error
            assertEquals(RequestState.CANCELLED, queued.state());
        }
    }

    // ── the state machine itself ────────────────────────────────────────────────────────────────

    /** Terminal states never transition, and the forbidden moves throw rather than corrupt. */
    @Test
    public void forbiddenTransitionsThrow() {
        try (KvCacheManager manager = manager(1, 2)) {
            Scheduler scheduler = new Scheduler(manager, 1, 4);

            ScheduledRequest request = scheduler.submit(BLOCK_TOKENS * 2);
            scheduler.admit();
            scheduler.complete(request);

            assertThrows(IllegalStateException.class, () -> scheduler.complete(request));
            assertThrows(
                    IllegalStateException.class,
                    () -> scheduler.fail(request, new RuntimeException()));

            ScheduledRequest rejected = scheduler.submit(0);
            assertThrows(IllegalStateException.class, () -> scheduler.complete(rejected));
        }
    }

    /** There is no PREEMPTED: the vocabulary is exactly six states. */
    @Test
    public void thereIsNoPreemptedState() {
        for (RequestState state : RequestState.values()) {
            assertNotEquals("PREEMPTED", state.name());
        }
        assertEquals(6, RequestState.values().length);
        assertTrue(RequestState.COMPLETED.isTerminal());
        assertTrue(RequestState.REJECTED.isTerminal());
        assertTrue(RequestState.CANCELLED.isTerminal());
        assertTrue(RequestState.FAILED.isTerminal());
        assertTrue(!RequestState.QUEUED.isTerminal() && !RequestState.RUNNING.isTerminal());
    }
}
