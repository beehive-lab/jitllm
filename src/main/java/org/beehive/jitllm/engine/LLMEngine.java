package org.beehive.jllm.engine;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntConsumer;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.batch.BatchExecutor;
import org.beehive.jllm.runtime.batch.BatchSlots;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.metrics.MetricKey;
import org.beehive.jllm.runtime.metrics.MetricsSink;

/**
 * Continuous batching over one loaded model: submit requests, drive {@code step()}, collect tokens.
 *
 * <p><b>The engine borrows the model</b>. {@link #close()} never closes it. One engine per (loaded
 * model, device, execution configuration).
 *
 * <p><b>Construction refuses rather than degrades.</b> A missing queue bound is an error; a model
 * that cannot share KV storage is an error. Engine-batched execution <i>is</i> the shared pool, so
 * falling back to private per-session storage would hand the operator a memory profile they did not
 * choose and would not be told about.
 *
 * <p>Thread-safety: {@link #addRequest} and {@link #cancel} are safe from any thread; {@link
 * #step()} is <b>single-caller</b> and concurrent calls throw.
 */
public final class LLMEngine implements AutoCloseable {

    private final Model model;
    private final Scheduler scheduler;
    private final BatchExecutor executor;
    private final KvCacheManager manager;

    private final Map<ScheduledRequest, RequestHandle> handles = new IdentityHashMap<>();
    private final Map<ScheduledRequest, Integer> generated = new IdentityHashMap<>();

    /**
     * Submission time per request, for queue wait and TTFT.
     *
     * <p>Kept here rather than on the handle because it is a measurement, not part of the caller's
     * view: a handle that carried timing would invite it into the public surface later.
     */
    private final Map<ScheduledRequest, Long> submittedAtNanos = new IdentityHashMap<>();

    /** Rule 17: written from below, read from above. Disabled by default, and then free. */
    private final MetricsSink metrics;

    /** Whichever thread is inside {@code step()}; the re-entrancy and single-caller guard. */
    private volatile Thread stepping;

    private boolean closed;

    /**
     * @param model borrowed, never closed by this engine
     * @param manager the shared KV runtime this engine's requests lease from
     * @param executor advances the batch by one token; the seam a fake stands in for. Its {@code
     *     maxBatchSize()} must match B — a mismatch means the persistent buffers and the slot table
     *     disagree about the batch
     * @param maxBatchSize B, fixed here
     * @param maxQueuedRequests required and positive; no library default
     * @throws IllegalArgumentException if the model cannot back shared KV storage, or the bounds
     *     are not positive
     */
    public LLMEngine(
            Model model,
            KvCacheManager manager,
            BatchExecutor executor,
            int maxBatchSize,
            int maxQueuedRequests) {
        this(model, manager, executor, maxBatchSize, maxQueuedRequests, MetricsSink.disabled());
    }

    /**
     * @param metrics where serving measurements go. {@link MetricsSink#disabled()} is the default
     *     and costs nothing; collection is opt-in because it is not free [Rule 17]
     */
    public LLMEngine(
            Model model,
            KvCacheManager manager,
            BatchExecutor executor,
            int maxBatchSize,
            int maxQueuedRequests,
            MetricsSink metrics) {
        if (model == null || manager == null || executor == null) {
            throw new IllegalArgumentException(
                    "an engine needs a model, a KV runtime and an executor");
        }
        if (!model.supportsSharedKvStorage()) {
            throw new IllegalArgumentException(
                    "model "
                            + model.getModelType()
                            + " cannot back shared KV storage: its layer graphs do not consume the block"
                            + " table from a named predecessor, so a shared, mutable table would go stale."
                            + " Engine-batched execution is the shared pool — there is no engine that"
                            + " quietly runs on private per-session storage");
        }
        if (executor.maxBatchSize() != maxBatchSize) {
            throw new IllegalArgumentException(
                    "the executor was built for a batch of "
                            + executor.maxBatchSize()
                            + " but this engine is being constructed with B="
                            + maxBatchSize
                            + ". B sizes the persistent buffers and the captured plan, so"
                            + " the two cannot disagree");
        }
        this.model = model;
        this.manager = manager;
        this.executor = executor;
        this.metrics = metrics == null ? MetricsSink.disabled() : metrics;
        this.scheduler = new Scheduler(manager, maxBatchSize, maxQueuedRequests);
    }

    /** The model this engine borrows. It is not this engine's to close. */
    public Model model() {
        return model;
    }

    public int maxBatchSize() {
        return scheduler.maxBatchSize();
    }

    public int maxQueuedRequests() {
        return scheduler.maxQueuedRequests();
    }

    public int queueDepth() {
        return scheduler.queueDepth();
    }

    public int runningCount() {
        return scheduler.runningCount();
    }

    /**
     * Submits a request. Non-blocking, thread-safe, and never runs model work.
     *
     * <p>A rejected request comes back as a handle already in {@link RequestState#REJECTED} with a
     * reason — the engine does not throw for backpressure, because a caller that must distinguish
     * "full" from "broken" should not have to read an exception type to do it.
     */
    public RequestHandle addRequest(int[] promptTokens, int maxNewTokens, IntConsumer onToken) {
        if (promptTokens == null || promptTokens.length == 0) {
            throw new IllegalArgumentException(
                    "a request needs prepared prompt tokens. The engine"
                            + " takes token ids, not text: rendering a conversation is the caller's job,"
                            + " through the model's own chat template");
        }
        // The reservation must cover the prompt as well as the generation — the prompt occupies KV
        // too, and admission promises the whole budget up front.
        int declaredBudgetTokens = promptTokens.length + maxNewTokens;
        ScheduledRequest request;
        RequestHandle handle;
        synchronized (this) {
            if (closed) {
                request = new ScheduledRequest(-1, declaredBudgetTokens);
                handle = new RequestHandle(request, onToken);
                scheduler.cancel(request); // no-op; keeps the handle terminal and consistent
                return handle;
            }
            request = scheduler.submit(declaredBudgetTokens);
            request.promptTokens(promptTokens);
            request.maxNewTokens(maxNewTokens);
            handle = new RequestHandle(request, onToken);
            handles.put(request, handle);
            if (metrics.isEnabled()) {
                submittedAtNanos.put(request, System.nanoTime());
                if (request.state() == RequestState.REJECTED) {
                    metrics.record(MetricKey.REQUESTS_REJECTED, 1);
                }
            }
        }
        if (request.isTerminal()) {
            handle.signalTerminal();
        }
        return handle;
    }

    /** Cancels from any thread. Queued leaves at once; running gives back its slot here. */
    public void cancel(RequestHandle handle) {
        synchronized (this) {
            scheduler.cancel(handle.request());
        }
        if (handle.isTerminal()) {
            handle.signalTerminal();
        }
    }

    /**
     * One batched iteration: admit, execute, append, release the locks, then call back.
     *
     * <p><b>That order is the contract</b>, not an implementation detail:
     *
     * <ul>
     *   <li>appending before the callback keeps the token readable after a throw;
     *   <li>releasing the locks before calling back is what lets a callback submit or cancel
     *       without deadlocking against the machinery that produced its token.
     * </ul>
     *
     * @return how many requests advanced this step; {@code 0} means there was nothing to do
     * @throws IllegalStateException if called concurrently, or re-entered from a callback
     */
    public int step() {
        Thread current = Thread.currentThread();
        Thread other = stepping;
        if (other != null) {
            throw new IllegalStateException(
                    other == current
                            ? "step() was re-entered from a callback. Callbacks may submit or cancel, but"
                                    + " not drive the engine — step() is single-caller"
                            : "step() is single-caller and "
                                    + other.getName()
                                    + " is already inside it."
                                    + " Concurrent step() is a caller bug");
        }
        stepping = current;

        List<Runnable> callbacks = new ArrayList<>();
        List<RequestHandle> terminal = new ArrayList<>();
        int advanced = 0;
        try {
            synchronized (this) {
                if (closed) {
                    return 0;
                }
                for (ScheduledRequest admitted : scheduler.admit()) {
                    RequestHandle admittedHandle = handles.get(admitted);
                    if (admittedHandle != null) {
                        admittedHandle.markAdmitted();
                    }
                    if (metrics.isEnabled()) {
                        metrics.record(MetricKey.REQUESTS_ADMITTED, 1);
                        Long submitted = submittedAtNanos.get(admitted);
                        if (submitted != null) {
                            metrics.record(
                                    MetricKey.QUEUE_WAIT_TIME, System.nanoTime() - submitted);
                        }
                        int prefilled = admitted.lease().prefilledTokens();
                        if (prefilled > 0) {
                            metrics.record(
                                    MetricKey.PREFIX_BLOCKS_REUSED,
                                    prefilled / manager.capacity().blockSizeTokens());
                        }
                    }
                }
                ScheduledRequest[] slots = scheduler.activeSlots();
                if (isEmpty(slots)) {
                    return 0;
                }

                if (metrics.isEnabled()) {
                    metrics.record(MetricKey.ENGINE_STEPS, 1);
                    metrics.record(MetricKey.BATCH_OCCUPANCY, scheduler.runningCount());
                    var capacity = manager.capacity();
                    metrics.record(MetricKey.KV_BLOCKS_IN_USE, capacity.usedBlocks());
                    metrics.record(MetricKey.KV_BLOCKS_TOTAL, capacity.totalBlocks());
                }
                int[] tokens = executor.decodeStep(compose(slots));
                for (int slot = 0; slot < slots.length; slot++) {
                    ScheduledRequest request = slots[slot];
                    if (request == null) {
                        continue; // inactive slot: ran, contributed nothing
                    }
                    int token = tokens[slot];
                    RequestHandle handle = handles.get(request);
                    boolean generating = request.isGenerating();
                    request.consume();
                    advanced++;

                    if (request.consumed() == request.promptTokens().length) {
                        // The prompt is fully ingested, so its KV exists and can be shared. Whole
                        // blocks only — a partial block is not something the table can point at.
                        int blockTokens = manager.capacity().blockSizeTokens();
                        int covered = request.promptTokens().length / blockTokens * blockTokens;
                        manager.rememberPrefix(request.lease(), request.promptTokens(), covered);
                    }

                    if (!generating) {
                        // Still feeding the prompt. The sample predicts a token we already have,
                        // so it is discarded rather than delivered as output.
                        continue;
                    }

                    if (metrics.isEnabled() && handle.tokenCount() == 0) {
                        Long submitted = submittedAtNanos.get(request);
                        if (submitted != null) {
                            // TTFT covers the queue wait and the prompt, which is what a caller
                            // experiences as latency — not just the decode that produced the token.
                            metrics.record(
                                    MetricKey.TIME_TO_FIRST_TOKEN, System.nanoTime() - submitted);
                        }
                    }
                    handle.appendToken(token); // before the callback, which runs outside the lock
                    int count = generated.merge(request, 1, Integer::sum);
                    // The stop token is the model's vocabulary; the budget is the scheduler's.
                    boolean finished =
                            executor.isStopToken(token) || count >= request.maxNewTokens();
                    if (finished) {
                        scheduler.complete(request);
                        terminal.add(handle);
                    }
                    if (handle.callback() != null) {
                        callbacks.add(() -> deliver(handle, token));
                    }
                }
            }
            // Outside every lock, on this thread.
            for (Runnable callback : callbacks) {
                callback.run();
            }
        } finally {
            stepping = null;
        }
        for (RequestHandle handle : terminal) {
            handle.signalTerminal();
        }
        return advanced;
    }

    /**
     * Runs one callback. A throw fails <b>that request only</b>, keeping the exception and the
     * already-appended token on its handle; the rest of the batch is unaffected.
     */
    private void deliver(RequestHandle handle, int token) {
        try {
            handle.callback().accept(token);
        } catch (RuntimeException | Error e) {
            synchronized (this) {
                if (!handle.request().isTerminal()) {
                    scheduler.fail(handle.request(), e);
                }
            }
            handle.signalTerminal();
        }
    }

    /**
     * Stops admission, terminalizes every outstanding handle, and releases the leases.
     *
     * <p>Does <b>not</b> close the model: the engine borrows it.
     */
    @Override
    public void close() {
        List<RequestHandle> toSignal = new ArrayList<>();
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            for (ScheduledRequest rejected : scheduler.shutdown()) {
                toSignal.add(handles.get(rejected));
            }
            for (ScheduledRequest running : scheduler.activeSlots()) {
                if (running != null) {
                    scheduler.cancel(running);
                    toSignal.add(handles.get(running));
                }
            }
        }
        for (RequestHandle handle : toSignal) {
            if (handle != null) {
                handle.signalTerminal();
            }
        }
    }

    /** The KV runtime this engine leases from. Exposed for capacity reporting, not for mutation. */
    public KvCacheManager kvRuntime() {
        return manager;
    }

    /**
     * Turns the scheduler's slots into the neutral batch the executor takes.
     *
     * <p>Inactive slots stay in the arrays, marked inactive rather than removed: the kernels run
     * the whole batch every step, and a slot that vanished would shift every slot after it.
     */
    private BatchSlots compose(ScheduledRequest[] slots) {
        boolean[] active = new boolean[slots.length];
        int[] tokens = new int[slots.length];
        int[] positions = new int[slots.length];
        int[] kvSlots = new int[slots.length];
        for (int slot = 0; slot < slots.length; slot++) {
            ScheduledRequest request = slots[slot];
            if (request == null) {
                continue;
            }
            active[slot] = true;
            RequestHandle handle = handles.get(request);
            int lastSampled =
                    handle.tokenCount() > 0 ? handle.tokens().get(handle.tokenCount() - 1) : 0;
            tokens[slot] = request.nextInput(lastSampled);
            // The absolute sequence position, which is how many tokens this sequence has been fed —
            // prompt and generated alike. The KV block table is indexed by it.
            positions[slot] = request.consumed();
            kvSlots[slot] = request.lease().slot();
        }
        return new BatchSlots(active, tokens, positions, kvSlots);
    }

    private static boolean isEmpty(ScheduledRequest[] slots) {
        for (ScheduledRequest request : slots) {
            if (request != null) {
                return false;
            }
        }
        return true;
    }
}
