package org.beehive.jitllm.auxiliary.metrics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Map;
import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.junit.Test;

/**
 * Note {@link RunMetrics} is a process-wide static holder, so these tests assert what a recorded
 * key does to the snapshot rather than trying to isolate one.
 */
public class RunMetricsSinkTest {

    @Test
    public void setUpMeasurementsLandInTheFieldsTheRendererAlreadyReads() {
        RunMetricsSink sink = new RunMetricsSink();

        sink.record(MetricKey.PLAN_CREATION_TIME, 111L);
        sink.record(MetricKey.JIT_COMPILE_TIME, 222L);
        sink.record(MetricKey.WEIGHT_UPLOAD_TIME, 333L);
        sink.record(MetricKey.MODEL_LOAD_TIME, 444L);

        RunMetricsSnapshot snapshot = RunMetrics.snapshot();
        assertEquals(111L, snapshot.tornadoPlanCreationDuration());
        assertEquals(222L, snapshot.tornadoJitDuration());
        assertEquals(333L, snapshot.tornadoReadOnlyWeightsCopyInDuration());
        assertEquals(444L, snapshot.loadDuration());
    }

    /**
     * The seam and the direct calls are the same collector: a value recorded through the sink is
     * indistinguishable from one set by a producer that never heard of {@link MetricsSink}.
     */
    @Test
    public void theSeamAndTheDirectSetterReachTheSameCollector() {
        RunMetrics.setTornadoMetrics(1L, 2L, 3L);
        assertEquals(1L, RunMetrics.snapshot().tornadoPlanCreationDuration());

        new RunMetricsSink().record(MetricKey.PLAN_CREATION_TIME, 9L);
        assertEquals(9L, RunMetrics.snapshot().tornadoPlanCreationDuration());
    }

    @Test
    public void deviceMeasurementsAreCollectedButNotPrinted() {
        RunMetricsSink sink = new RunMetricsSink();

        sink.record(MetricKey.DEVICE_KERNEL_TIME, 100L);
        sink.record(MetricKey.DEVICE_KERNEL_TIME, 40L);
        sink.record(MetricKey.DEVICE_MEMORY_USED, 5_000L);
        sink.record(MetricKey.DEVICE_MEMORY_USED, 7_000L);

        Map<MetricKey, Long> device = sink.deviceMetrics();
        assertEquals(
                "per-execution measurements total over the run",
                Long.valueOf(140L),
                device.get(MetricKey.DEVICE_KERNEL_TIME));
        assertEquals(
                "a level is the last reading, not a sum",
                Long.valueOf(7_000L),
                device.get(MetricKey.DEVICE_MEMORY_USED));

        // Nothing device-side has a field on the snapshot, so the printed report cannot change.
        String rendered = new HumanMetricsRenderer().render(RunMetrics.snapshot());
        assertFalse(rendered, rendered.contains("device_kernel"));
        assertFalse(rendered, rendered.contains("140"));
    }

    @Test
    public void whatItReturnsIsACopy() {
        RunMetricsSink sink = new RunMetricsSink();
        sink.record(MetricKey.DEVICE_KERNEL_TIME, 1L);
        Map<MetricKey, Long> first = sink.deviceMetrics();
        sink.record(MetricKey.DEVICE_KERNEL_TIME, 1L);
        assertEquals(Long.valueOf(1L), first.get(MetricKey.DEVICE_KERNEL_TIME));
        assertEquals(Long.valueOf(2L), sink.deviceMetrics().get(MetricKey.DEVICE_KERNEL_TIME));
    }

    /** Device collection costs profiler time per execution, so it is opt-in and off by default. */
    @Test
    public void collectionIsOffUnlessAskedFor() {
        String previous = System.getProperty(RunMetricsSink.ENABLE_PROPERTY);
        try {
            System.clearProperty(RunMetricsSink.ENABLE_PROPERTY);
            MetricsSink off = RunMetricsSink.installedOrDisabled();
            assertFalse(off.isEnabled());
            assertSame(MetricsSink.disabled(), off);

            System.setProperty(RunMetricsSink.ENABLE_PROPERTY, "true");
            MetricsSink on = RunMetricsSink.installedOrDisabled();
            assertTrue(on.isEnabled());
            assertTrue(on instanceof RunMetricsSink);
        } finally {
            if (previous == null) {
                System.clearProperty(RunMetricsSink.ENABLE_PROPERTY);
            } else {
                System.setProperty(RunMetricsSink.ENABLE_PROPERTY, previous);
            }
        }
    }
}
