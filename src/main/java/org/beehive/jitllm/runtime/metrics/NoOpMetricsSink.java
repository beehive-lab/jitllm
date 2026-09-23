package org.beehive.jllm.runtime.metrics;

/**
 * The sink that discards everything, reached through {@link MetricsSink#disabled()}.
 *
 * <p>Package-private and stateless on purpose: it exists so that a producer never has to null-check
 * a sink, and so that "metrics off" is an ordinary object rather than a branch threaded through
 * every call site.
 */
final class NoOpMetricsSink implements MetricsSink {

    static final NoOpMetricsSink INSTANCE = new NoOpMetricsSink();

    private NoOpMetricsSink() {}

    @Override
    public void record(MetricKey key, long value) {
        // Discarded by design.
    }

    @Override
    public boolean isEnabled() {
        return false;
    }

    @Override
    public String toString() {
        return "MetricsSink.disabled()";
    }
}
