package org.beehive.jitllm.auxiliary.metrics;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsReport;
import org.beehive.jitllm.runtime.metrics.MetricsSink;

/**
 * The sink behind the CLI's metrics report — {@link RunMetrics} seen through the Rule 17 seam.
 *
 * <p>{@code RunMetrics} predates the seam: it is a static holder that producers call directly and
 * the CLI prints at the end of a run. This sink does not replace it. It gives the same collector a
 * second, seam-shaped way in, so that a producer which knows only {@link MetricsSink} — a backend,
 * for instance — reaches the same report as a producer that calls {@code RunMetrics} directly.
 * <b>What the CLI prints is unchanged</b>: set-up and phase keys land in the very fields the
 * renderers already read.
 *
 * <p>Device-side keys have no field on {@link RunMetricsSnapshot} and are <b>not</b> printed. They
 * accumulate here, by the key's own {@link MetricKey#aggregation()} rule, and are readable through
 * {@link #deviceMetrics()} — programmatically available without changing a line of CLI output.
 *
 * <p>Installed only when asked for: see {@link #installedOrDisabled()}. Collecting device metrics
 * means enabling the backend's profiler, which is paid per execution, so the default stays {@link
 * MetricsSink#disabled()}.
 */
public final class RunMetricsSink implements MetricsSink {

    /** Opt-in switch. Absent or false ⇒ {@link MetricsSink#disabled()}. */
    public static final String ENABLE_PROPERTY = "jitllm.metrics.device";

    private final Map<MetricKey, Long> deviceMetrics = new EnumMap<>(MetricKey.class);

    /**
     * The sink the CLI installs: this one when {@code -Djitllm.metrics.device=true}, the disabled
     * sink otherwise. Returning the disabled sink rather than a flag on this class keeps the
     * "nobody is listening" answer in one place — the backend asks {@link #isEnabled()} and skips
     * the measurement entirely.
     */
    public static MetricsSink installedOrDisabled() {
        return Boolean.getBoolean(ENABLE_PROPERTY) ? new RunMetricsSink() : MetricsSink.disabled();
    }

    @Override
    public synchronized void record(MetricKey key, long value) {
        switch (key) {
                // Set-up and phase measurements have a home on the snapshot already; route them
                // there so the printed report is identical whichever way the producer reached it.
            case MODEL_LOAD_TIME -> RunMetrics.setLoadDuration(value);
            case PLAN_CREATION_TIME -> RunMetrics.setTornadoPlanCreation(value);
            case JIT_COMPILE_TIME -> RunMetrics.setTornadoJit(value);
            case WEIGHT_UPLOAD_TIME -> RunMetrics.setTornadoWeightUpload(value);
            default -> accumulate(key, value);
        }
    }

    private void accumulate(MetricKey key, long value) {
        switch (key.aggregation()) {
            case SUM -> deviceMetrics.merge(key, value, Long::sum);
            case LATEST -> deviceMetrics.put(key, value);
        }
    }

    /**
     * Everything this run measured: the CLI collector's own counters, plus the device-side keys it
     * has no field for. This is the programmatic form of what {@code printMetrics()} prints.
     */
    public synchronized MetricsReport report() {
        return RunMetrics.report().merge(MetricsReport.of(deviceMetrics));
    }

    /** What has been collected that the printed report has no field for. */
    public synchronized Map<MetricKey, Long> deviceMetrics() {
        return Collections.unmodifiableMap(new EnumMap<>(deviceMetrics));
    }

    @Override
    public String toString() {
        return "RunMetricsSink" + deviceMetrics();
    }
}
