package org.beehive.jllm.engine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import org.beehive.jllm.runtime.backend.CapacityQuery;
import org.beehive.jllm.runtime.backend.KvCapacity;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.kv.KvLease;

/**
 * Admission and slot assignment for the engine: who runs next, and against which blocks.
 *
 * <p><b>Strict FCFS, and strictly means strictly</b>. Admission looks at the head of the queue and
 * stops there. It never scans past a request it cannot yet fit to find a smaller one it can — that
 * is what defers a large request indefinitely under sustained load. The cost is head-of-line
 * blocking, which is visible and measurable; the alternative is starvation, which is neither.
 *
 * <p><b>Whole-budget reservation</b>. An admitted request takes the blocks its declared budget
 * needs, once, and holds them until it reaches a terminal state. There is no incremental growth and
 * no revisiting, so a request cannot be starved after admission — and any exhaustion that does
 * happen afterwards is an invariant violation rather than a capacity event.
 *
 * <p><b>Fixed B</b>. The slot count is set at construction; each step may have 0…B running and
 * empty slots are inactive, not errors. Changing B means constructing again.
 *
 * <p>Thread-safety follows the contract: {@link #submit} and {@link #cancel} are safe from any
 * thread; {@link #admit} runs inside the single caller's {@code step()}.
 */
public final class Scheduler {

    private final KvCacheManager manager;

    /**
     * The capacity seam admission reads. It is the same object as {@code manager} today — what
     * changed is that admission now depends on the <b>contract</b> rather than on the
     * implementation, so a scheduler test can state a capacity instead of building a pool to imply
     * one.
     *
     * <p>Logical capacity only: blocks the pool was sized for and blocks free, never device bytes
     * free.
     */
    private final CapacityQuery capacityQuery;

    private final int maxBatchSize;
    private final int maxQueuedRequests;

    private final Deque<ScheduledRequest> queue = new ArrayDeque<>();

    /**
     * Indexed by <b>KV slot</b>, not by an independent batch position.
     *
     * <p>The two must be the same number. The batched kernels read the block table at {@code
     * batchIndex * blocksPerSlot + …}, so a request whose batch position differed from its lease's
     * slot would read another sequence's mapping — silently, because both are valid indices.
     * Letting the lease choose and following it removes the second allocator that could disagree.
     *
     * <p>Sized by the pool's slot count, which is therefore required to be at least B.
     */
    private final ScheduledRequest[] slots;

    private long nextSequenceNumber;
    private boolean shuttingDown;

    /**
     * @param manager the KV runtime admission reserves against
     * @param maxBatchSize B — slots, fixed for this scheduler's life
     * @param maxQueuedRequests the queue bound. **Required and positive**; there is no default,
     *     because a wrong one is worse than an absent one — too small rejects load the deployment
     *     could have absorbed, too large hides saturation behind unbounded queue wait, and neither
     *     announces itself. The caller supplies its own
     */
    public Scheduler(KvCacheManager manager, int maxBatchSize, int maxQueuedRequests) {
        if (manager == null) {
            throw new IllegalArgumentException("a scheduler needs a KV runtime to reserve against");
        }
        this.capacityQuery = manager;
        if (maxBatchSize <= 0) {
            throw new IllegalArgumentException(
                    "maxBatchSize must be positive, got "
                            + maxBatchSize
                            + ". B is fixed at construction and sizes the persistent buffers");
        }
        if (maxQueuedRequests <= 0) {
            throw new IllegalArgumentException(
                    "maxQueuedRequests must be given and positive, got "
                            + maxQueuedRequests
                            + ". There is no default: the queue bound is a backpressure"
                            + " policy the deployment owns, not one the library guesses");
        }
        if (manager.pool().maxSlots() < maxBatchSize) {
            throw new IllegalArgumentException(
                    "the KV pool addresses "
                            + manager.pool().maxSlots()
                            + " slots but B is "
                            + maxBatchSize
                            + ". A running request occupies the batch"
                            + " position its lease's slot names, so the pool must have at least one slot"
                            + " per batch position");
        }
        this.manager = manager;
        this.maxBatchSize = maxBatchSize;
        this.maxQueuedRequests = maxQueuedRequests;
        this.slots = new ScheduledRequest[manager.pool().maxSlots()];
    }

    /**
     * Queues a request, or rejects it terminally. Non-blocking, and never runs model work.
     *
     * <p>Capacity shortfall is <b>not</b> a rejection — that is what the queue is for. The four
     * rejections are the ones no amount of waiting fixes, plus the queue bound itself.
     */
    public synchronized ScheduledRequest submit(int declaredBudgetTokens) {
        ScheduledRequest request = new ScheduledRequest(nextSequenceNumber++, declaredBudgetTokens);
        if (shuttingDown) {
            request.reject(RejectionReason.SHUTDOWN);
            return request;
        }
        if (declaredBudgetTokens <= 0) {
            request.reject(RejectionReason.MALFORMED);
            return request;
        }
        if (declaredBudgetTokens > tokensPerSlot()) {
            request.reject(RejectionReason.CANNOT_EVER_FIT);
            return request;
        }
        if (queue.size() >= maxQueuedRequests) {
            request.reject(RejectionReason.QUEUE_FULL);
            return request;
        }
        queue.addLast(request);
        return request;
    }

    /**
     * Admits from the head of the queue while slots and blocks allow. Called at a step boundary.
     *
     * <p>Stops at the first request it cannot fit — it does not look past it. See the class note on
     * why that is the whole point rather than a simplification.
     *
     * @return the requests that moved to {@code RUNNING}, in admission order
     */
    public synchronized List<ScheduledRequest> admit() {
        List<ScheduledRequest> admitted = new ArrayList<>();
        while (!queue.isEmpty()) {
            if (runningCount() >= maxBatchSize) {
                break; // B is full; nothing to admit into
            }
            ScheduledRequest head = queue.peekFirst();
            if (!canReserve(head.declaredBudgetTokens())) {
                break; // head-of-line blocking, deliberately
            }
            // Reuses a cached prefix when the manager has one for these tokens. Without a
            // prefix cache this is exactly acquire(): the lease comes back with nothing prefilled.
            KvLease lease =
                    manager.acquireWithPrefix(head.declaredBudgetTokens(), head.promptTokens());
            // The lease's slot is the batch position. See the field note: two allocators would be
            // two chances to disagree, and the disagreement reads as another sequence's KV.
            int slot = lease.slot();
            queue.removeFirst();
            head.admit(slot, lease);
            // Positions the prefix already covers are not fed again: their KV is in blocks another
            // sequence may be reading, and re-writing them is at best redundant work.
            head.skipPrefilled(lease.prefilledTokens());
            slots[slot] = head;
            admitted.add(head);
        }
        return admitted;
    }

    /** Finished generating: stop token, budget reached, or context full. */
    public synchronized void complete(ScheduledRequest request) {
        request.complete();
        release(request);
    }

    /**
     * Failed: a callback threw, the backend errored, or an invariant did not hold.
     *
     * <p>The cause is retained on the request so the handle can show it.
     */
    public synchronized void fail(ScheduledRequest request, Throwable cause) {
        request.fail(cause);
        release(request);
    }

    /**
     * Cancels from any thread. A queued request leaves the queue at once; a running one gives back
     * its slot and lease here, which the engine calls at the next step boundary.
     */
    public synchronized void cancel(ScheduledRequest request) {
        if (request.isTerminal()) {
            return; // idempotent: cancelling twice is not an error
        }
        boolean wasQueued = request.state() == RequestState.QUEUED;
        request.cancel();
        if (wasQueued) {
            queue.remove(request);
        } else {
            release(request);
        }
    }

    /**
     * Post-admission exhaustion is an invariant violation, never a truncation.
     *
     * <p>Admission reserved the whole declared budget and holds it to a terminal state, so a
     * running request cannot legitimately run out of blocks. If it does, the reservation logic is
     * wrong, and saying so is more useful than absorbing it as a capacity event and quietly
     * producing a shorter answer.
     */
    public synchronized void failOnExhaustion(ScheduledRequest request, String detail) {
        fail(
                request,
                new IllegalStateException(
                        "KV exhausted after admission for "
                                + request
                                + ": "
                                + detail
                                + ". Admission reserves the whole declared budget and retains it to a terminal"
                                + " state, so this cannot happen unless the reservation is wrong — it is reported"
                                + " as a broken invariant rather than truncating the response"));
    }

    /** Stops admission. Queued requests are rejected; running ones are left to the caller. */
    public synchronized List<ScheduledRequest> shutdown() {
        shuttingDown = true;
        List<ScheduledRequest> rejected = new ArrayList<>(queue);
        for (ScheduledRequest request : rejected) {
            request.reject(RejectionReason.SHUTDOWN);
        }
        queue.clear();
        return rejected;
    }

    /** The running requests by slot; {@code null} where a slot is inactive. 0…B of them. */
    public synchronized ScheduledRequest[] activeSlots() {
        return slots.clone();
    }

    public synchronized int runningCount() {
        int n = 0;
        for (ScheduledRequest request : slots) {
            if (request != null) {
                n++;
            }
        }
        return n;
    }

    public synchronized int queueDepth() {
        return queue.size();
    }

    public int maxBatchSize() {
        return maxBatchSize;
    }

    public int maxQueuedRequests() {
        return maxQueuedRequests;
    }

    /** Tokens one slot can ever hold — the ceiling {@link RejectionReason#CANNOT_EVER_FIT} uses. */
    public int tokensPerSlot() {
        return capacityQuery.tokensPerSlot();
    }

    private boolean canReserve(int tokens) {
        KvCapacity capacity = capacityQuery.capacity();
        int blocksNeeded = ceilDiv(tokens, capacity.blockSizeTokens());
        return blocksNeeded <= capacity.freeBlocks();
    }

    private void release(ScheduledRequest request) {
        int slot = request.slot();
        KvLease lease = request.lease();
        if (slot >= 0) {
            slots[slot] = null;
        }
        if (lease != null) {
            lease.close();
        }
        request.releaseResources();
    }

    private static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }
}
