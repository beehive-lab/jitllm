package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.junit.Test;

/**
 * The device-side path needs a real execution and is covered by {@code TornadoMetricsAccelTest}
 * (Class B).
 */
public class TornadoMetricsReporterTest {

    private static final class RecordingSink implements MetricsSink {
        private final Map<MetricKey, Long> values = new EnumMap<>(MetricKey.class);
        private final List<MetricKey> order = new ArrayList<>();

        @Override
        public void record(MetricKey key, long value) {
            values.put(key, value);
            order.add(key);
        }
    }

    @Test
    public void aDisabledSinkLeavesTheReporterInert() {
        TornadoMetricsReporter reporter = new TornadoMetricsReporter(MetricsSink.disabled());
        assertFalse(reporter.isEnabled());
        reporter.reportSetUp(1, 2, 3); // no observer, no work
        assertNull(reporter.report(null)); // and nothing to unwrap
    }

    /**
     * A null sink is treated as disabled rather than thrown on: the reporter sits on the decode
     * path, where an NPE per token would be a worse failure than no metrics.
     */
    @Test
    public void aNullSinkIsTreatedAsDisabled() {
        assertFalse(new TornadoMetricsReporter(null).isEnabled());
    }

    @Test
    public void setUpDurationsReachAnEnabledSink() {
        RecordingSink sink = new RecordingSink();
        TornadoMetricsReporter reporter = new TornadoMetricsReporter(sink);
        assertTrue(reporter.isEnabled());

        reporter.reportSetUp(11L, 22L, 33L);

        assertEquals(Long.valueOf(11L), sink.values.get(MetricKey.PLAN_CREATION_TIME));
        assertEquals(Long.valueOf(22L), sink.values.get(MetricKey.JIT_COMPILE_TIME));
        assertEquals(Long.valueOf(33L), sink.values.get(MetricKey.WEIGHT_UPLOAD_TIME));
        assertEquals(3, sink.order.size());
    }

    /**
     * The enabled state is read once, at construction: the profiler has to be switched on before
     * the plan compiles, so a sink that turns itself on later would be asked for results that were
     * never collected.
     */
    @Test
    public void theEnabledStateIsFixedAtConstruction() {
        class SwitchableSink implements MetricsSink {
            boolean on = false;

            @Override
            public void record(MetricKey key, long value) {}

            @Override
            public boolean isEnabled() {
                return on;
            }
        }
        SwitchableSink sink = new SwitchableSink();
        TornadoMetricsReporter reporter = new TornadoMetricsReporter(sink);
        sink.on = true;
        assertFalse(
                "the reporter must not start collecting for a plan that has no profiler",
                reporter.isEnabled());
    }
}
