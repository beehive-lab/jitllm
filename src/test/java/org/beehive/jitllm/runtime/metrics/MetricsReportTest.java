package org.beehive.jllm.runtime.metrics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;

public class MetricsReportTest {

    private static final long ONE_SECOND = 1_000_000_000L;

    private static MetricsReport report(Object... keyValues) {
        Map<MetricKey, Long> values = new EnumMap<>(MetricKey.class);
        for (int i = 0; i < keyValues.length; i += 2) {
            values.put((MetricKey) keyValues[i], ((Number) keyValues[i + 1]).longValue());
        }
        return MetricsReport.of(values);
    }

    @Test
    public void ratesAreDerivedFromTheCountsSoTheyCannotDisagree() {
        MetricsReport r =
                report(MetricKey.GENERATED_TOKENS, 64, MetricKey.DECODE_TIME, ONE_SECOND / 2);
        assertEquals(128.0, r.generatedTokensPerSecond().orElseThrow(), 1e-9);
    }

    @Test
    public void promptAndTotalRatesUseTheirOwnCountsAndDurations() {
        MetricsReport r =
                report(
                        MetricKey.PROMPT_TOKENS,
                        21,
                        MetricKey.PREFILL_TIME,
                        ONE_SECOND / 10,
                        MetricKey.GENERATED_TOKENS,
                        64,
                        MetricKey.DECODE_TIME,
                        ONE_SECOND / 2,
                        MetricKey.TOTAL_TIME,
                        ONE_SECOND);
        assertEquals(210.0, r.promptTokensPerSecond().orElseThrow(), 1e-9);
        assertEquals(85.0, r.totalTokensPerSecond().orElseThrow(), 1e-9);
    }

    /** Absent is not zero: "not measured" and "measured as nothing" are different claims. */
    @Test
    public void anUnrecordedKeyIsAbsentRatherThanZero() {
        MetricsReport r = report(MetricKey.DECODE_TIME, ONE_SECOND);
        assertEquals(Optional.empty(), r.value(MetricKey.PREFILL_TIME));
        assertEquals(0L, r.valueOr(MetricKey.PREFILL_TIME, 0L));
        assertTrue(r.value(MetricKey.DECODE_TIME).isPresent());
    }

    @Test
    public void aRateNeedsBothHalvesAndNeverDividesByZero() {
        assertEquals(
                Optional.empty(),
                report(MetricKey.GENERATED_TOKENS, 64).generatedTokensPerSecond());
        assertEquals(
                Optional.empty(),
                report(MetricKey.DECODE_TIME, ONE_SECOND).generatedTokensPerSecond());
        assertEquals(
                "a run too short to time is not a run of infinite speed",
                Optional.empty(),
                report(MetricKey.GENERATED_TOKENS, 64, MetricKey.DECODE_TIME, 0)
                        .generatedTokensPerSecond());
    }

    @Test
    public void mergeCombinesByTheKeysOwnAggregation() {
        MetricsReport first =
                report(MetricKey.DEVICE_KERNEL_TIME, 100, MetricKey.DEVICE_MEMORY_USED, 5_000);
        MetricsReport second =
                report(MetricKey.DEVICE_KERNEL_TIME, 40, MetricKey.DEVICE_MEMORY_USED, 7_000);

        MetricsReport merged = first.merge(second);
        assertEquals(Long.valueOf(140L), merged.value(MetricKey.DEVICE_KERNEL_TIME).orElseThrow());
        assertEquals(
                Long.valueOf(7_000L), merged.value(MetricKey.DEVICE_MEMORY_USED).orElseThrow());
    }

    @Test
    public void aReportIsImmutableAndCopiesWhatItIsGiven() {
        Map<MetricKey, Long> source = new EnumMap<>(MetricKey.class);
        source.put(MetricKey.DECODE_TIME, ONE_SECOND);
        MetricsReport r = MetricsReport.of(source);
        source.put(MetricKey.DECODE_TIME, 1L);

        assertEquals(Long.valueOf(ONE_SECOND), r.value(MetricKey.DECODE_TIME).orElseThrow());
        try {
            r.values().put(MetricKey.PREFILL_TIME, 1L);
            org.junit.Assert.fail("the report's map must not be modifiable");
        } catch (UnsupportedOperationException expected) {
            // as designed
        }
    }

    @Test
    public void anEmptyReportSaysSoRatherThanReportingZeroes() {
        MetricsReport empty = MetricsReport.empty();
        assertTrue(empty.isEmpty());
        assertFalse(empty.generatedTokensPerSecond().isPresent());
    }
}
