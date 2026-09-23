package org.beehive.jitllm.auxiliary.metrics;

import java.util.EnumMap;
import java.util.Map;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsReport;
import org.beehive.jitllm.runtime.metrics.MetricsSink;

/**
 * The sink for a caller that wants the numbers rather than a printout: it accumulates what it is
 * given and hands back a {@link MetricsReport}.
 *
 * <p>Each key combines by its own {@link MetricKey#aggregation()} — summing per-execution
 * measurements, keeping the last reading of a level. Doing that here rather than in the producer is
 * the point of the seam: the producer reports what it measured and does not have to know how a
 * run's total is formed.
 *
 * <p>Safe to hand to a backend, which reports from its execution threads.
 */
public final class InMemoryMetricsSink implements MetricsSink {

    private final Map<MetricKey, Long> values = new EnumMap<>(MetricKey.class);

    @Override
    public synchronized void record(MetricKey key, long value) {
        switch (key.aggregation()) {
            case SUM -> values.merge(key, value, Long::sum);
            case LATEST -> values.put(key, value);
        }
    }

    /** Everything recorded so far. Taking one mid-run is fine; it is a copy. */
    public synchronized MetricsReport report() {
        return MetricsReport.of(values);
    }

    public synchronized void clear() {
        values.clear();
    }

    @Override
    public String toString() {
        return "InMemoryMetricsSink" + report();
    }
}
