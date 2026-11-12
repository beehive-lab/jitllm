package org.beehive.jitllm.runtime.metrics;

/**
 * What a metric measures, and in what unit.
 *
 * <p>An enumeration rather than free-form strings: the producers are backends and the consumers are
 * the engine and the API, and a typo in a string key between the two is a metric that is silently
 * never read. Every key is named for the thing measured, not for where it is measured — {@link
 * #DEVICE_KERNEL_TIME} is device kernel time whichever backend reports it.
 *
 * <p>Keys are added as the milestone that produces them lands. A key nothing writes yet is worse
 * than no key: it reads as a supported measurement that is always zero.
 */
public enum MetricKey {

    // ── Set-up, measured once per run ─────────────────────────────────────────
    /** Reading and mapping the model file; excludes backend initialisation. */
    MODEL_LOAD_TIME(Unit.NANOSECONDS, Aggregation.LATEST),
    /** Building the backend's program description (task-graph construction). */
    PLAN_CREATION_TIME(Unit.NANOSECONDS, Aggregation.LATEST),
    /** Ahead-of-time compilation of that program. */
    JIT_COMPILE_TIME(Unit.NANOSECONDS, Aggregation.LATEST),
    /** First upload of the read-only weights to the device. */
    WEIGHT_UPLOAD_TIME(Unit.NANOSECONDS, Aggregation.LATEST),

    // ── Inference phases ──────────────────────────────────────────────────────
    PREFILL_TIME(Unit.NANOSECONDS, Aggregation.SUM),
    DECODE_TIME(Unit.NANOSECONDS, Aggregation.SUM),
    TOTAL_TIME(Unit.NANOSECONDS, Aggregation.LATEST),
    PROMPT_TOKENS(Unit.COUNT, Aggregation.SUM),
    GENERATED_TOKENS(Unit.COUNT, Aggregation.SUM),

    // ── Device-side, from the backend's profiler ──────────────────────────────
    /** Per execution, so the run total is the sum — see {@code TornadoMetricsReporter}. */
    DEVICE_KERNEL_TIME(Unit.NANOSECONDS, Aggregation.SUM),
    DEVICE_WRITE_TIME(Unit.NANOSECONDS, Aggregation.SUM),
    DEVICE_READ_TIME(Unit.NANOSECONDS, Aggregation.SUM),
    BYTES_COPIED_TO_DEVICE(Unit.BYTES, Aggregation.SUM),
    BYTES_COPIED_FROM_DEVICE(Unit.BYTES, Aggregation.SUM),
    /** A level, not a total: summing device memory across executions is meaningless. */
    DEVICE_MEMORY_USED(Unit.BYTES, Aggregation.LATEST),

    // ── Serving, from the engine ───────────────────────────────────────
    // These describe a server under load rather than a single generation, which is why they are
    // separated: a run that never used the engine reports none of them, and that absence is
    // information rather than a zero to be misread as "no queue wait".

    /**
     * Time from a request being submitted to its first token — time to first token.
     *
     * <p>The number a user experiences as latency. Summed across requests so that mean TTFT is
     * derivable against {@link #REQUESTS_ADMITTED}; a distribution needs a sink that keeps one,
     * which this vocabulary deliberately does not force on every implementation.
     */
    TIME_TO_FIRST_TOKEN(Unit.NANOSECONDS, Aggregation.SUM),

    /** Time a request spent queued before admission. */
    QUEUE_WAIT_TIME(Unit.NANOSECONDS, Aggregation.SUM),

    /** Requests that reached {@code RUNNING}. */
    REQUESTS_ADMITTED(Unit.COUNT, Aggregation.SUM),

    /** Requests refused before running — queue full, never fits, malformed, shutting down. */
    REQUESTS_REJECTED(Unit.COUNT, Aggregation.SUM),

    /**
     * Active slots summed over steps, so mean occupancy is this over {@link #ENGINE_STEPS}.
     *
     * <p>Occupancy is what says whether batching is earning anything: a batch that averages one
     * active slot is a slower single-token path with extra machinery.
     */
    BATCH_OCCUPANCY(Unit.COUNT, Aggregation.SUM),

    /** Batched iterations executed. The denominator for {@link #BATCH_OCCUPANCY}. */
    ENGINE_STEPS(Unit.COUNT, Aggregation.SUM),

    /** Leased KV blocks at the last step — a level, like device memory. */
    KV_BLOCKS_IN_USE(Unit.COUNT, Aggregation.LATEST),

    /** Blocks the pool holds in total, so utilisation is the ratio of the two. */
    KV_BLOCKS_TOTAL(Unit.COUNT, Aggregation.LATEST),

    /**
     * Blocks a request took from the prefix cache instead of prefilling.
     *
     * <p>The saving in the pool's own unit. Against {@link #REQUESTS_ADMITTED} it says how much of
     * the offered prompt work the cache is actually removing, which is the number that decides
     * whether a prefix cache is earning the pool capacity it holds.
     */
    PREFIX_BLOCKS_REUSED(Unit.COUNT, Aggregation.SUM);

    /** The unit of a recorded value. Every metric is a {@code long} in exactly one of these. */
    public enum Unit {
        NANOSECONDS,
        BYTES,
        COUNT
    }

    /**
     * How repeated records of the same key combine into the run's value.
     *
     * <p>The producer records a measurement; only the key knows whether measurements add up. Device
     * kernel time is reported per execution and totals over the run; device memory in use is a
     * level and the last reading is the answer. Without this a sink has to guess, and guessing
     * wrong turns a gauge into a number that grows with the token count.
     */
    public enum Aggregation {
        SUM,
        LATEST
    }

    private final Unit unit;
    private final Aggregation aggregation;

    MetricKey(Unit unit, Aggregation aggregation) {
        this.unit = unit;
        this.aggregation = aggregation;
    }

    public Unit unit() {
        return unit;
    }

    public Aggregation aggregation() {
        return aggregation;
    }
}
