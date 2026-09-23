package org.beehive.jllm.auxiliary.metrics;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.util.Objects;
import org.beehive.jllm.runtime.metrics.MetricKey;
import org.beehive.jllm.runtime.metrics.MetricsSink;

/**
 * A sink that writes each measurement to the platform logger — the alternative to library code
 * printing measurements itself (Rule 16).
 *
 * <p>Library code that calls {@code System.out} cannot be silenced or redirected by whoever embeds
 * it. Logging can: {@link System#getLogger} routes through whatever the application configured, and
 * with no configuration at all {@link Level#DEBUG} is not loggable, so this sink is a **no-op by
 * default** — which is also what {@link #isEnabled()} reports, so a producer skips the measurement
 * rather than taking one for nobody.
 *
 * <p>Level is DEBUG rather than INFO deliberately: a metric per execution is diagnostic volume, and
 * a sink that filled a production log by existing would simply not be used.
 */
public final class LoggingMetricsSink implements MetricsSink {

    private static final Level LEVEL = Level.DEBUG;

    private final Logger logger;

    public LoggingMetricsSink() {
        this(System.getLogger(LoggingMetricsSink.class.getName()));
    }

    /**
     * For an embedder that already has a logger — and for the tests, which need to read it back.
     */
    public LoggingMetricsSink(Logger logger) {
        this.logger = Objects.requireNonNull(logger, "logger");
    }

    @Override
    public boolean isEnabled() {
        return logger.isLoggable(LEVEL);
    }

    @Override
    public void record(MetricKey key, long value) {
        if (!logger.isLoggable(LEVEL)) {
            return;
        }
        logger.log(LEVEL, key.name() + "=" + value + " " + key.unit());
    }

    @Override
    public String toString() {
        return "LoggingMetricsSink[" + logger.getName() + ", enabled=" + isEnabled() + "]";
    }
}
