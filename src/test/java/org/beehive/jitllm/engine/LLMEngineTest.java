package org.beehive.jllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.batch.BatchExecutor;
import org.beehive.jllm.runtime.kv.BlockPool;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.junit.Test;

/**
 * Class A by design: the {@link BatchExecutor} seam means every transition, the callback ordering,
 * the re-entrancy rule and the construction preconditions can be asserted without a device. What is
 * <b>not</b> here is the device-backed executor and #129's throughput number — those need a
 * production multi-slot batched plan, which does not exist outside the frozen bench.
 */
public class LLMEngineTest {

    private static final int BLOCK_TOKENS = 16;

    /**
     * A one-token prompt. The engine feeds the prompt before it generates, so a longer one would
     * shift every token count in these tests without testing anything more.
     */
    private static final int[] PROMPT = {7};

    private static KvCacheManager manager(int slots, int blocksPerSlot) {
        return new KvCacheManager(
                new BlockPool(slots * blocksPerSlot, blocksPerSlot, slots, BLOCK_TOKENS, 4096));
    }

    private static LLMEngine engine(
            KvCacheManager manager, BatchExecutor executor, int b, int queue) {
        return new LLMEngine(TestModels.sharedKvCapable(), manager, executor, b, queue);
    }

    // ── construction refuses what it must ───────────────────────────────────────────────────────

    @Test
    public void constructionRequiresAModelThatCanShareKvStorage() {
        try (KvCacheManager manager = manager(2, 4)) {
            IllegalArgumentException refused =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    new LLMEngine(
                                            TestModels.privateKvOnly(),
                                            manager,
                                            new FakeBatchExecutor(2, 2),
                                            2,
                                            8));

            assertTrue(refused.getMessage(), refused.getMessage().contains("shared KV storage"));
            assertTrue(
                    "names what is missing and why",
                    refused.getMessage().contains("named predecessor"));
            assertTrue(
                    "and refuses rather than degrading",
                    refused.getMessage().contains("no engine that"));
        }
    }

    @Test
    public void constructionRequiresAPositiveQueueBoundAndBatchSize() {
        try (KvCacheManager manager = manager(2, 4)) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> engine(manager, new FakeBatchExecutor(2, 2), 2, 0));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> engine(manager, new FakeBatchExecutor(0, 2), 0, 8));
        }
    }

    /** The engine borrows the model; closing the engine must not close it. */
    @Test
    public void theEngineBorrowsTheModel() {
        try (KvCacheManager manager = manager(2, 4)) {
            Model model = TestModels.sharedKvCapable();
            LLMEngine engine = new LLMEngine(model, manager, new FakeBatchExecutor(2, 1), 2, 8);
            assertSame(model, engine.model());
            engine.close();
            // Nothing to assert on the model beyond this: it has no close() of its own to observe,
            // which is itself the point — the engine has no way to close what it borrows.
            assertSame(model, engine.model());
        }
    }

    // ── the happy path ──────────────────────────────────────────────────────────────────────────

    @Test
    public void aRequestRunsToCompletionAndDeliversItsTokens() throws Exception {
        try (KvCacheManager manager = manager(2, 4)) {
            FakeBatchExecutor executor = new FakeBatchExecutor(2, 3);
            try (LLMEngine engine = engine(manager, executor, 2, 8)) {
                RequestHandle handle =
                        engine.addRequest(PROMPT, BLOCK_TOKENS * 4 - PROMPT.length, null);
                assertEquals(RequestState.QUEUED, handle.state());

                while (!handle.isTerminal()) {
                    engine.step();
                }

                assertEquals(RequestState.COMPLETED, handle.state());
                assertEquals("three tokens then the stop token", 4, handle.tokenCount());
                assertEquals("the lease went back", 0, manager.liveLeases());
            }
        }
    }

    /** 0…B active per step; inactive slots are passed to the executor, not skipped. */
    @Test
    public void inactiveSlotsAreStillPartOfTheBatch() {
        try (KvCacheManager manager = manager(4, 2)) {
            FakeBatchExecutor executor = new FakeBatchExecutor(4, 1);
            try (LLMEngine engine = engine(manager, executor, 4, 8)) {
                engine.addRequest(PROMPT, BLOCK_TOKENS * 2 - PROMPT.length, null);
                engine.step();

                assertEquals("the batch is always B wide", List.of(4), executor.slotCountsPerStep);
                assertEquals(
                        "but only one slot was active", List.of(1), executor.activeCountsPerStep);
            }
        }
    }

    @Test
    public void stepReturnsZeroWhenThereIsNothingToDo() {
        try (KvCacheManager manager = manager(2, 4)) {
            try (LLMEngine engine = engine(manager, new FakeBatchExecutor(2, 1), 2, 8)) {
                assertEquals(0, engine.step());
            }
        }
    }

    // ── the callback contract ────────────────────────────────────────────────────────────

    /** The token is appended before the callback runs, so a throwing callback still leaves it. */
    @Test
    public void aThrowingCallbackFailsOnlyItsOwnRequestAndKeepsBothTheTokenAndTheCause() {
        try (KvCacheManager manager = manager(2, 4)) {
            FakeBatchExecutor executor = new FakeBatchExecutor(2, 5);
            try (LLMEngine engine = engine(manager, executor, 2, 8)) {
                RuntimeException boom = new RuntimeException("callback threw");
                RequestHandle failing =
                        engine.addRequest(
                                PROMPT,
                                BLOCK_TOKENS * 4 - PROMPT.length,
                                token -> {
                                    throw boom;
                                });
                List<Integer> received = new ArrayList<>();
                RequestHandle healthy =
                        engine.addRequest(PROMPT, BLOCK_TOKENS * 4 - PROMPT.length, received::add);

                engine.step();

                assertEquals(RequestState.FAILED, failing.state());
                assertSame("the cause is retained", boom, failing.failure());
                assertEquals("and so is the token that caused it", 1, failing.tokenCount());

                assertEquals(
                        "the other request is untouched", RequestState.RUNNING, healthy.state());
                assertEquals(1, received.size());

                engine.step();
                assertEquals("and it keeps running", 2, healthy.tokenCount());
            }
        }
    }

    /** Callbacks may submit and cancel — which is what proves the locks were dropped first. */
    @Test
    public void aCallbackMaySubmitAndCancel() {
        try (KvCacheManager manager = manager(4, 2)) {
            try (LLMEngine engine = engine(manager, new FakeBatchExecutor(2, 3), 2, 8)) {
                AtomicReference<RequestHandle> submitted = new AtomicReference<>();
                RequestHandle first =
                        engine.addRequest(
                                PROMPT,
                                BLOCK_TOKENS * 2 - PROMPT.length,
                                token -> {
                                    if (submitted.get() == null) {
                                        submitted.set(
                                                engine.addRequest(
                                                        PROMPT,
                                                        BLOCK_TOKENS * 2 - PROMPT.length,
                                                        null));
                                    }
                                });

                engine.step();

                assertNotNull("a callback submitted without deadlocking", submitted.get());
                assertEquals(RequestState.QUEUED, submitted.get().state());

                engine.cancel(first);
                assertEquals(RequestState.CANCELLED, first.state());
            }
        }
    }

    /** A callback may not drive the engine: {@code step()} from inside one is a caller bug. */
    @Test
    public void aCallbackMayNotReenterStep() {
        try (KvCacheManager manager = manager(2, 4)) {
            try (LLMEngine engine = engine(manager, new FakeBatchExecutor(2, 3), 2, 8)) {
                AtomicReference<Throwable> caught = new AtomicReference<>();
                engine.addRequest(
                        PROMPT,
                        BLOCK_TOKENS * 4 - PROMPT.length,
                        token -> {
                            try {
                                engine.step();
                            } catch (Throwable t) {
                                caught.set(t);
                            }
                        });

                engine.step();

                assertNotNull("re-entering step() must throw", caught.get());
                assertTrue(caught.get() instanceof IllegalStateException);
                assertTrue(
                        caught.get().getMessage(),
                        caught.get().getMessage().contains("re-entered from a callback"));
            }
        }
    }

    /** Concurrent {@code step()} is a caller bug too. */
    @Test
    public void stepIsSingleCaller() throws Exception {
        try (KvCacheManager manager = manager(2, 4)) {
            FakeBatchExecutor slow = new FakeBatchExecutor(2, 50);
            try (LLMEngine engine = engine(manager, slow, 2, 8)) {
                AtomicReference<Throwable> fromOtherThread = new AtomicReference<>();
                AtomicInteger entered = new AtomicInteger();

                engine.addRequest(
                        PROMPT,
                        BLOCK_TOKENS * 4 - PROMPT.length,
                        token -> {
                            if (entered.getAndIncrement() > 0) {
                                return;
                            }
                            Thread other =
                                    new Thread(
                                            () -> {
                                                try {
                                                    engine.step();
                                                } catch (Throwable t) {
                                                    fromOtherThread.set(t);
                                                }
                                            },
                                            "second-stepper");
                            other.start();
                            try {
                                other.join();
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                        });

                engine.step();

                assertNotNull("a second thread inside step() must throw", fromOtherThread.get());
                assertTrue(
                        fromOtherThread.get().getMessage(),
                        fromOtherThread.get().getMessage().contains("single-caller"));
            }
        }
    }

    // ── shutdown ────────────────────────────────────────────────────────────────────────────────

    /** close() terminalizes everything and releases the leases, and never closes the model. */
    @Test
    public void closeLeavesNoHandleNonTerminalAndNoLeaseLive() {
        try (KvCacheManager manager = manager(2, 4)) {
            LLMEngine engine = engine(manager, new FakeBatchExecutor(1, 50), 1, 8);

            RequestHandle running =
                    engine.addRequest(PROMPT, BLOCK_TOKENS * 4 - PROMPT.length, null);
            RequestHandle queued =
                    engine.addRequest(PROMPT, BLOCK_TOKENS * 4 - PROMPT.length, null);
            engine.step();
            assertEquals(RequestState.RUNNING, running.state());
            assertEquals(RequestState.QUEUED, queued.state());

            engine.close();

            assertTrue("nothing is left non-terminal", running.isTerminal() && queued.isTerminal());
            assertEquals(RequestState.CANCELLED, running.state());
            assertEquals(RequestState.REJECTED, queued.state());
            assertEquals(RejectionReason.SHUTDOWN, queued.rejectionReason());
            assertEquals("every lease came back", 0, manager.liveLeases());

            engine.close(); // idempotent
        }
    }

    @Test
    public void awaitReturnsWhenTheRequestTerminates() throws Exception {
        try (KvCacheManager manager = manager(2, 4)) {
            try (LLMEngine engine = engine(manager, new FakeBatchExecutor(2, 2), 2, 8)) {
                RequestHandle handle =
                        engine.addRequest(PROMPT, BLOCK_TOKENS * 4 - PROMPT.length, null);
                Thread driver =
                        new Thread(
                                () -> {
                                    while (!handle.isTerminal()) {
                                        engine.step();
                                    }
                                },
                                "driver");
                driver.start();
                handle.await();
                driver.join();
                assertEquals(RequestState.COMPLETED, handle.state());
            }
        }
    }
}
