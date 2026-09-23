package org.beehive.jllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.auxiliary.metrics.InMemoryMetricsSink;
import org.beehive.jllm.runtime.kv.BlockPool;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.metrics.MetricKey;
import org.beehive.jllm.runtime.metrics.MetricsReport;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.junit.Test;

public class EngineMetricsTest {

    private static final int BLOCK_TOKENS = 16;
    private static final int[] PROMPT = {7};

    private static KvCacheManager manager(int slots, int blocksPerSlot) {
        return new KvCacheManager(
                new BlockPool(slots * blocksPerSlot, blocksPerSlot, slots, BLOCK_TOKENS, 4096));
    }

    private static LLMEngine engine(KvCacheManager manager, MetricsSink sink, int b, int queue) {
        return new LLMEngine(
                TestModels.sharedKvCapable(), manager, new FakeBatchExecutor(b, 4), b, queue, sink);
    }

    @Test
    public void admissionsRejectionsAndOccupancyAreRecorded() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        try (KvCacheManager manager = manager(2, 2)) {
            try (LLMEngine engine = engine(manager, sink, 2, 1)) {
                engine.addRequest(PROMPT, 4, null);
                engine.addRequest(PROMPT, 4, null); // queue bound is 1 → rejected
                engine.step();
                engine.step();

                MetricsReport report = sink.report();
                assertEquals(
                        "one admitted", 1L, (long) report.valueOr(MetricKey.REQUESTS_ADMITTED, 0));
                assertEquals(
                        "one refused", 1L, (long) report.valueOr(MetricKey.REQUESTS_REJECTED, 0));
                assertEquals("two steps ran", 2L, (long) report.valueOr(MetricKey.ENGINE_STEPS, 0));
                assertTrue(
                        "occupancy accumulated", report.valueOr(MetricKey.BATCH_OCCUPANCY, 0) >= 1);
                assertEquals(
                        "and block utilisation is reportable as a ratio",
                        4L,
                        (long) report.valueOr(MetricKey.KV_BLOCKS_TOTAL, 0));
            }
        }
    }

    /**
     * Queue wait is the cost strict FCFS trades starvation for. It is only a defensible trade while
     * somebody can see it, which is what this metric is for.
     */
    @Test
    public void queueWaitIsRecordedForARequestThatWaited() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        try (KvCacheManager manager = manager(2, 2)) {
            try (LLMEngine engine = engine(manager, sink, 1, 4)) { // B = 1: the second waits
                engine.addRequest(PROMPT, 4, null);
                engine.addRequest(PROMPT, 4, null);
                engine.step(); // admits the first only

                long afterFirst = sink.report().valueOr(MetricKey.QUEUE_WAIT_TIME, 0);
                while (engine.queueDepth() > 0) {
                    engine.step();
                }
                long afterSecond = sink.report().valueOr(MetricKey.QUEUE_WAIT_TIME, 0);

                assertTrue("the second request's wait was recorded", afterSecond > afterFirst);
                assertEquals(2L, (long) sink.report().valueOr(MetricKey.REQUESTS_ADMITTED, 0));
            }
        }
    }

    /** TTFT covers the wait and the prompt, because that is what a caller experiences. */
    @Test
    public void timeToFirstTokenIsRecordedOncePerRequest() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        try (KvCacheManager manager = manager(2, 2)) {
            try (LLMEngine engine = engine(manager, sink, 2, 4)) {
                RequestHandle handle = engine.addRequest(PROMPT, 4, null);
                while (!handle.isTerminal()) {
                    engine.step();
                }

                assertTrue(
                        "a request that produced tokens has a TTFT",
                        sink.report().valueOr(MetricKey.TIME_TO_FIRST_TOKEN, 0) > 0);
                assertTrue("and it produced more than one token", handle.tokenCount() > 1);
            }
        }
    }

    /**
     * Disabled is the default and must cost nothing — no timing taken, no map populated. A sink
     * that had to be asked to be quiet would be paid for by every deployment that never reads it.
     */
    @Test
    public void aDisabledSinkRecordsNothing() {
        InMemoryMetricsSink sink = new InMemoryMetricsSink();
        try (KvCacheManager manager = manager(2, 2)) {
            try (LLMEngine engine =
                    new LLMEngine(
                            TestModels.sharedKvCapable(),
                            manager,
                            new FakeBatchExecutor(2, 4),
                            2,
                            4)) { // no sink: disabled by default
                RequestHandle handle = engine.addRequest(PROMPT, 4, null);
                while (!handle.isTerminal()) {
                    engine.step();
                }
            }
        }
        assertTrue("nothing reached a sink that was never installed", sink.report().isEmpty());
    }
}
