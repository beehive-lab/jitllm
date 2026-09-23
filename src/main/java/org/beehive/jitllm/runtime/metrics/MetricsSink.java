package org.beehive.jllm.runtime.metrics;

/**
 * Where measurements go — the one seam in this architecture that is written from below and read
 * from above.
 *
 * <p>Metrics run against the dependency direction of everything else: they are produced at the
 * bottom (backend, device) and consumed at the top (engine, API). A metrics facility placed in the
 * upper layers would be un-callable from where the data originates, so the interface lives here in
 * the runtime layer and the implementations live above it. Backends depend on this package and
 * never on an implementation of it. That edge is a designed permission, not a tolerated violation —
 * Rule 17 in {@code docs/architecture/architecture.md}.
 *
 * <h2>Off by default</h2>
 *
 * <p>Collection is not free: on the TornadoVM backend it means enabling the profiler, which is paid
 * per execution — that is, per token on the decode path. So the default sink is {@link
 * #disabled()}, and a producer that has to do work to obtain a value must ask {@link #isEnabled()}
 * first:
 *
 * <pre>{@code
 * if (sink.isEnabled()) {
 *     sink.record(MetricKey.DEVICE_KERNEL_TIME, result.getProfilerResult().getDeviceKernelTime());
 * }
 * }</pre>
 *
 * <p>Recording into a disabled sink is harmless — it is a no-op, not an error — but the guard is
 * what keeps the measurement itself from being taken.
 *
 * <h2>Threading</h2>
 *
 * <p>An implementation must tolerate calls from any thread. Producers span the loader, the
 * backend's execution threads and the generation loop, and nothing in this interface promises them
 * a common thread.
 */
public interface MetricsSink {

    /**
     * Records one measurement. The unit is fixed by the key ({@link MetricKey#unit()}); passing a
     * value in any other unit is a producer defect that no implementation can detect.
     */
    void record(MetricKey key, long value);

    /**
     * Whether anything consumes what is recorded. False on {@link #disabled()}; producers use it to
     * skip taking a measurement that would be discarded.
     */
    default boolean isEnabled() {
        return true;
    }

    /** The sink that measures nothing — the default everywhere, so telemetry is opt-in. */
    static MetricsSink disabled() {
        return NoOpMetricsSink.INSTANCE;
    }
}
