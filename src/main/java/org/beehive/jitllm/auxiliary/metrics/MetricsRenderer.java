package org.beehive.jitllm.auxiliary.metrics;

/** Renders a {@link RunMetricsSnapshot} to a string for a specific output format. */
@FunctionalInterface
public interface MetricsRenderer {
    String render(RunMetricsSnapshot snapshot);
}
