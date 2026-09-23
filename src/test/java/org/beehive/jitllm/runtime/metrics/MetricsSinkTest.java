package org.beehive.jitllm.runtime.metrics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public class MetricsSinkTest {

    /** A sink of the kind the later milestones add: records what it is given, nothing more. */
    private static final class RecordingSink implements MetricsSink {
        private final Map<MetricKey, Long> values = new EnumMap<>(MetricKey.class);
        private final List<MetricKey> order = new ArrayList<>();

        @Override
        public void record(MetricKey key, long value) {
            values.merge(key, value, Long::sum);
            order.add(key);
        }
    }

    @Test
    public void theDefaultSinkMeasuresNothingAndSaysSo() {
        MetricsSink sink = MetricsSink.disabled();
        assertFalse(
                "the default must report itself disabled, so producers skip the measurement",
                sink.isEnabled());
        sink.record(MetricKey.DECODE_TIME, 1234L); // harmless, not an error
    }

    @Test
    public void theDisabledSinkIsASingleton() {
        // Producers hold on to it; a new instance per call would allocate on the decode path.
        assertSame(MetricsSink.disabled(), MetricsSink.disabled());
    }

    @Test
    public void anImplementationIsEnabledUnlessItSaysOtherwise() {
        assertTrue(new RecordingSink().isEnabled());
    }

    @Test
    public void recordedValuesReachTheSinkInOrder() {
        RecordingSink sink = new RecordingSink();
        sink.record(MetricKey.PROMPT_TOKENS, 21);
        sink.record(MetricKey.GENERATED_TOKENS, 64);
        sink.record(MetricKey.GENERATED_TOKENS, 1);

        assertEquals(
                List.of(
                        MetricKey.PROMPT_TOKENS,
                        MetricKey.GENERATED_TOKENS,
                        MetricKey.GENERATED_TOKENS),
                sink.order);
        assertEquals(Long.valueOf(21), sink.values.get(MetricKey.PROMPT_TOKENS));
        assertEquals(Long.valueOf(65), sink.values.get(MetricKey.GENERATED_TOKENS));
    }

    @Test
    public void everyKeyDeclaresItsUnit() {
        for (MetricKey key : MetricKey.values()) {
            assertTrue(key.name(), key.unit() != null);
        }
    }

    @Test
    public void byteAndCountKeysAreNotMeasuredInNanoseconds() {
        // The unit is fixed by the key, so a producer cannot record bytes into a duration.
        assertEquals(MetricKey.Unit.BYTES, MetricKey.BYTES_COPIED_TO_DEVICE.unit());
        assertEquals(MetricKey.Unit.BYTES, MetricKey.DEVICE_MEMORY_USED.unit());
        assertEquals(MetricKey.Unit.COUNT, MetricKey.GENERATED_TOKENS.unit());
        assertEquals(MetricKey.Unit.NANOSECONDS, MetricKey.DEVICE_KERNEL_TIME.unit());
    }
}
