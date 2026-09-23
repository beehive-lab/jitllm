package org.beehive.jitllm.auxiliary.metrics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.junit.Test;

public class LoggingMetricsSinkTest {

    /** Captures what was logged, and at which level it would have been. */
    private static final class CapturingLogger implements Logger {
        private final List<String> messages = new ArrayList<>();
        private final Level enabledFrom;

        CapturingLogger(Level enabledFrom) {
            this.enabledFrom = enabledFrom;
        }

        @Override
        public String getName() {
            return "capturing";
        }

        @Override
        public boolean isLoggable(Level level) {
            return level.getSeverity() >= enabledFrom.getSeverity();
        }

        @Override
        public void log(Level level, ResourceBundle bundle, String msg, Throwable thrown) {
            if (isLoggable(level)) {
                messages.add(msg);
            }
        }

        @Override
        public void log(Level level, ResourceBundle bundle, String format, Object... params) {
            if (isLoggable(level)) {
                messages.add(format);
            }
        }
    }

    @Test
    public void withNoLoggingConfiguredTheSinkIsOffAndSaysSo() {
        // The default platform configuration does not log DEBUG, so the sink reports itself
        // disabled and a producer skips the measurement entirely.
        LoggingMetricsSink sink = new LoggingMetricsSink(new CapturingLogger(Level.INFO));
        assertFalse(sink.isEnabled());
    }

    @Test
    public void nothingIsWrittenWhileTheLevelIsOff() {
        CapturingLogger logger = new CapturingLogger(Level.INFO);
        LoggingMetricsSink sink = new LoggingMetricsSink(logger);

        sink.record(MetricKey.DEVICE_KERNEL_TIME, 1234L);

        assertTrue(logger.messages.toString(), logger.messages.isEmpty());
    }

    @Test
    public void whenDebugIsEnabledEachMeasurementIsLoggedWithItsUnit() {
        CapturingLogger logger = new CapturingLogger(Level.DEBUG);
        LoggingMetricsSink sink = new LoggingMetricsSink(logger);
        assertTrue(sink.isEnabled());

        sink.record(MetricKey.DEVICE_KERNEL_TIME, 1234L);
        sink.record(MetricKey.BYTES_COPIED_TO_DEVICE, 4096L);

        assertEquals(
                List.of("DEVICE_KERNEL_TIME=1234 NANOSECONDS", "BYTES_COPIED_TO_DEVICE=4096 BYTES"),
                logger.messages);
    }

    @Test
    public void theDefaultConstructorUsesThePlatformLogger() {
        // No configuration in a plain test JVM, so this must be inert rather than chatty.
        LoggingMetricsSink sink = new LoggingMetricsSink();
        sink.record(MetricKey.DECODE_TIME, 1L);
        assertFalse(sink.isEnabled());
    }
}
