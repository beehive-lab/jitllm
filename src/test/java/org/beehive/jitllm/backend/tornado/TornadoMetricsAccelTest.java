package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.metrics.MetricKey;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.junit.Test;

/**
 * Asserts the half of the seam that needs a real device: with a sink attached, the TornadoVM
 * profiler is switched on and its device kernel time, transfer bytes and device memory arrive as
 * {@link MetricKey} records; with the default disabled sink, nothing is recorded at all.
 *
 * <p>This is what makes the values real rather than plausible — every one of them was already being
 * produced by the runtime and discarded, so a test that only checked "the sink was called" would
 * pass on zeros.
 */
public class TornadoMetricsAccelTest {

    private static final int TOKENS = 8;

    /** Aggregates by the key's own rule, which is the whole reason {@code Aggregation} exists. */
    private static final class RecordingSink implements MetricsSink {
        private final Map<MetricKey, Long> values = new EnumMap<>(MetricKey.class);
        private final AtomicInteger records = new AtomicInteger();

        @Override
        public void record(MetricKey key, long value) {
            records.incrementAndGet();
            switch (key.aggregation()) {
                case SUM -> values.merge(key, value, Long::sum);
                case LATEST -> values.put(key, value);
            }
        }

        long get(MetricKey key) {
            return values.getOrDefault(key, 0L);
        }
    }

    @Test
    public void deviceMeasurementsReachTheSink() throws Exception {
        Path modelPath = fixtureOrSkip();
        RecordingSink sink = new RecordingSink();

        decode(modelPath, sink);

        assertTrue("no records at all — the profiler was never enabled", sink.records.get() > 0);
        assertTrue(
                "device kernel time: " + sink.get(MetricKey.DEVICE_KERNEL_TIME),
                sink.get(MetricKey.DEVICE_KERNEL_TIME) > 0);
        assertTrue(
                "bytes copied to device: " + sink.get(MetricKey.BYTES_COPIED_TO_DEVICE),
                sink.get(MetricKey.BYTES_COPIED_TO_DEVICE) > 0);
        assertTrue(
                "device memory used: " + sink.get(MetricKey.DEVICE_MEMORY_USED),
                sink.get(MetricKey.DEVICE_MEMORY_USED) > 0);
        assertTrue(
                "plan creation time: " + sink.get(MetricKey.PLAN_CREATION_TIME),
                sink.get(MetricKey.PLAN_CREATION_TIME) > 0);
        assertTrue(
                "JIT compile time: " + sink.get(MetricKey.JIT_COMPILE_TIME),
                sink.get(MetricKey.JIT_COMPILE_TIME) > 0);

        System.out.println(
                "[METRICS] kernel="
                        + sink.get(MetricKey.DEVICE_KERNEL_TIME)
                        + "ns"
                        + " in="
                        + sink.get(MetricKey.BYTES_COPIED_TO_DEVICE)
                        + "B"
                        + " out="
                        + sink.get(MetricKey.BYTES_COPIED_FROM_DEVICE)
                        + "B"
                        + " deviceMemory="
                        + sink.get(MetricKey.DEVICE_MEMORY_USED)
                        + "B"
                        + " over "
                        + sink.records.get()
                        + " records");
    }

    @Test
    public void aDisabledSinkIsNeverCalled() throws Exception {
        Path modelPath = fixtureOrSkip();

        final class DisabledSpySink implements MetricsSink {
            private final AtomicInteger calls = new AtomicInteger();

            @Override
            public void record(MetricKey key, long value) {
                calls.incrementAndGet();
            }

            @Override
            public boolean isEnabled() {
                return false;
            }
        }

        DisabledSpySink spy = new DisabledSpySink();
        decode(modelPath, spy);

        assertEquals(
                "a sink that says it is disabled must not be called at all", 0, spy.calls.get());
    }

    private static Path fixtureOrSkip() {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent: fixture " + Fixture.LLAMA_3_2_1B_Q8_0.fileName, false);
        }
        return modelPath;
    }

    private static void decode(Path modelPath, MetricsSink sink) throws Exception {
        Model model = ModelLoader.loadModel(modelPath, 512, true, true);
        State state = model.createNewState();
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model, sink);
        try {
            int token = model.shouldAddBeginOfText() ? model.chatFormat().getBeginOfText() : 0;
            for (int position = 0; position < TOKENS; position++) {
                org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                        model, state, token, position, plan);
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }
}
