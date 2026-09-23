package org.beehive.jitllm.runtime.metrics;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;

/**
 * What a run measured, as a value — the read side of the seam.
 *
 * <p>Metrics used to exist only as printed output, which meant an embedder could see them and a
 * program could not. A report is the same numbers as data: immutable, keyed by {@link MetricKey},
 * with the rates derived rather than stored so that they cannot disagree with the counts they come
 * from.
 *
 * <p>A key that was never recorded is absent rather than zero. The distinction matters on this
 * seam: a prefill time of zero is a claim about the run, while an absent one says the phase was
 * never measured — for instance because the sink was disabled, or because the path has no separate
 * prefill phase.
 */
public final class MetricsReport {

    private static final double NANOS_PER_SECOND = 1e9;

    private final Map<MetricKey, Long> values;

    private MetricsReport(Map<MetricKey, Long> values) {
        this.values = Collections.unmodifiableMap(values);
    }

    public static MetricsReport of(Map<MetricKey, Long> values) {
        EnumMap<MetricKey, Long> copy = new EnumMap<>(MetricKey.class);
        values.forEach(
                (key, value) -> {
                    if (key != null && value != null) {
                        copy.put(key, value);
                    }
                });
        return new MetricsReport(copy);
    }

    public static MetricsReport empty() {
        return new MetricsReport(new EnumMap<>(MetricKey.class));
    }

    /** Empty when the key was never recorded — which is not the same as recorded zero. */
    public Optional<Long> value(MetricKey key) {
        return Optional.ofNullable(values.get(key));
    }

    public long valueOr(MetricKey key, long fallback) {
        return values.getOrDefault(key, fallback);
    }

    public Map<MetricKey, Long> values() {
        return values;
    }

    public boolean isEmpty() {
        return values.isEmpty();
    }

    /** Tokens per second for the prompt (prefill) phase; empty unless both parts were measured. */
    public Optional<Double> promptTokensPerSecond() {
        return rate(MetricKey.PROMPT_TOKENS, MetricKey.PREFILL_TIME);
    }

    /** Tokens per second for generation (decode) — the number the benchmark gate compares. */
    public Optional<Double> generatedTokensPerSecond() {
        return rate(MetricKey.GENERATED_TOKENS, MetricKey.DECODE_TIME);
    }

    /** Tokens per second over the whole call, prompt and generation together. */
    public Optional<Double> totalTokensPerSecond() {
        long tokens =
                valueOr(MetricKey.PROMPT_TOKENS, 0L) + valueOr(MetricKey.GENERATED_TOKENS, 0L);
        Optional<Long> duration = value(MetricKey.TOTAL_TIME);
        if (tokens <= 0 || duration.isEmpty() || duration.get() <= 0) {
            return Optional.empty();
        }
        return Optional.of(tokens / (duration.get() / NANOS_PER_SECOND));
    }

    /**
     * A rate needs both a count and a duration, and a zero duration is not a division to attempt —
     * a run too short to time is not a run of infinite speed.
     */
    private Optional<Double> rate(MetricKey countKey, MetricKey durationKey) {
        Optional<Long> count = value(countKey);
        Optional<Long> duration = value(durationKey);
        if (count.isEmpty() || duration.isEmpty() || count.get() <= 0 || duration.get() <= 0) {
            return Optional.empty();
        }
        return Optional.of(count.get() / (duration.get() / NANOS_PER_SECOND));
    }

    /** This report plus {@code other}; where both carry a key, the key's aggregation decides. */
    public MetricsReport merge(MetricsReport other) {
        EnumMap<MetricKey, Long> merged = new EnumMap<>(values);
        other.values.forEach(
                (key, value) ->
                        merged.merge(
                                key,
                                value,
                                (mine, theirs) ->
                                        key.aggregation() == MetricKey.Aggregation.SUM
                                                ? mine + theirs
                                                : theirs));
        return new MetricsReport(merged);
    }

    @Override
    public String toString() {
        return "MetricsReport" + values;
    }
}
