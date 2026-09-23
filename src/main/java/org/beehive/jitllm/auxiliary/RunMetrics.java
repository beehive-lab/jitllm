package org.beehive.jitllm.auxiliary;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.auxiliary.metrics.GitHubMetricsRenderer;
import org.beehive.jitllm.auxiliary.metrics.HumanMetricsRenderer;
import org.beehive.jitllm.auxiliary.metrics.JsonMetricsRenderer;
import org.beehive.jitllm.auxiliary.metrics.MetricsRenderer;
import org.beehive.jitllm.auxiliary.metrics.RunMetricsSnapshot;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsReport;

/**
 * Singleton that accumulates fine-grained performance metrics across one inference run.
 *
 * <p>Metrics are set incrementally by different layers of the stack:
 *
 * <ul>
 *   <li>{@link #setLoadDuration} — called from {@code ModelLoader}
 *   <li>{@link #setTornadoMetrics} — called from TornadoVM plan constructors
 *   <li>{@link #setInferenceMetrics} — called from TokenGenerationLoop variants at end of
 *       generation
 *   <li>{@link #setHasPrefillPhase} — called from prefill-decode engine variants
 * </ul>
 *
 * <p>All durations are stored in nanoseconds. {@link #printMetrics()} builds an immutable {@link
 * RunMetricsSnapshot}, selects a {@link MetricsRenderer}, and writes to the configured sink.
 *
 * <p>Configurable via system properties:
 *
 * <ul>
 *   <li>{@code jitllm.metrics.format} — {@code human} (default) | {@code json} | {@code github}
 *   <li>{@code jitllm.metrics.output} — {@code stderr} (default) | {@code stdout} | {@code file}
 *   <li>{@code jitllm.metrics.file} — target path when {@code output=file}
 * </ul>
 */
public final class RunMetrics {

    // ── Core metrics (nanoseconds) ────────────────────────────────────────────
    private long totalDurationNs;
    private long loadDurationNs;
    private int promptEvalCount;
    private long promptEvalDurationNs;
    private int evalCount;
    private long evalDurationNs;
    private boolean hasPrefillPhase;

    // ── TornadoVM-specific metrics (nanoseconds) ──────────────────────────────
    /**
     * Which path executed, and the exact combination that selected it [D-7].
     *
     * <p>Null until a plan is built — a CPU run never sets it, and reporting "legacy" for a run
     * that had no accelerator plan at all would be a claim about something that did not happen.
     */
    private String executionPath;

    private String executionCombination;
    private Boolean executionQualified;
    private String executionOverride;

    private long tornadoPlanCreationNs;
    private long tornadoJitNs;
    private long readOnlyWeightsCopyInNs;

    // ── Singleton ─────────────────────────────────────────────────────────────
    private static final RunMetrics INSTANCE = new RunMetrics();

    private RunMetrics() {}

    // ── Setters ───────────────────────────────────────────────────────────────

    /** Records the time spent loading the model file (not including TornadoVM initialisation). */
    public static void setLoadDuration(long ns) {
        INSTANCE.loadDurationNs = ns;
    }

    /**
     * Records TornadoVM-specific initialisation durations.
     *
     * @param planCreationNs task-graph construction ({@code createExecutionPlan()})
     * @param jitNs JIT compilation ({@code withPreCompilation()})
     * @param weightCopyNs first-execution weight upload ({@code forceCopyInReadOnlyData()})
     */
    public static void setTornadoMetrics(long planCreationNs, long jitNs, long weightCopyNs) {
        INSTANCE.tornadoPlanCreationNs = planCreationNs;
        INSTANCE.tornadoJitNs = jitNs;
        INSTANCE.readOnlyWeightsCopyInNs = weightCopyNs;
    }

    /**
     * As {@link #setTornadoMetrics}, one measurement at a time. The metrics seam records each key
     * as it arrives ({@code RunMetricsSink}), so it has no moment at which all three are in hand.
     */
    public static void setTornadoPlanCreation(long ns) {
        INSTANCE.tornadoPlanCreationNs = ns;
    }

    /**
     * @see #setTornadoPlanCreation
     */
    public static void setTornadoJit(long ns) {
        INSTANCE.tornadoJitNs = ns;
    }

    /**
     * @see #setTornadoPlanCreation
     */
    public static void setTornadoWeightUpload(long ns) {
        INSTANCE.readOnlyWeightsCopyInNs = ns;
    }

    /**
     * Records inference-phase durations at the end of a generation run.
     *
     * @param promptCount number of prompt tokens processed (prefill)
     * @param prefillNs wall-clock time spent in the prefill phase
     * @param generatedCount number of tokens generated (decode)
     * @param decodeNs wall-clock time spent in the decode phase
     * @param totalNs total wall-clock time for the full inference call
     */
    public static void setInferenceMetrics(
            int promptCount, long prefillNs, int generatedCount, long decodeNs, long totalNs) {
        INSTANCE.promptEvalCount = promptCount;
        INSTANCE.promptEvalDurationNs = prefillNs;
        INSTANCE.evalCount = generatedCount;
        INSTANCE.evalDurationNs = decodeNs;
        INSTANCE.totalDurationNs = totalNs;
    }

    /**
     * Signals that prefill and decode are distinct timed phases. Called by {@code
     * TokenGenerationLoop}'s prefill paths before returning.
     */
    public static void setHasPrefillPhase(boolean value) {
        INSTANCE.hasPrefillPhase = value;
    }

    // ── Snapshot ──────────────────────────────────────────────────────────────

    /** Returns an immutable snapshot of all currently collected metrics. */
    /**
     * Records which path executed and the exact combination [D-7].
     *
     * <p>Called once per plan construction, from the one factory every caller reaches. A test that
     * wants to prove which path ran asserts on this rather than reading the {@code jitllm.lowering}
     * property back — the property says what was <b>asked for</b>.
     *
     * @param path {@code lowered} or {@code legacy}
     * @param combination {@code architecture/dtype/mode}, the exact triple qualification is keyed
     *     on
     */
    public static void setExecutionPath(
            String path, String combination, boolean qualified, String overrideSource) {
        INSTANCE.executionPath = path;
        INSTANCE.executionCombination = combination;
        INSTANCE.executionQualified = qualified;
        INSTANCE.executionOverride = overrideSource;
    }

    public static RunMetricsSnapshot snapshot() {
        RunMetrics m = INSTANCE;
        return RunMetricsSnapshot.of(
                m.totalDurationNs,
                m.loadDurationNs,
                m.promptEvalCount,
                m.promptEvalDurationNs,
                m.evalCount,
                m.evalDurationNs,
                m.hasPrefillPhase,
                m.tornadoPlanCreationNs,
                m.tornadoJitNs,
                m.readOnlyWeightsCopyInNs,
                m.executionPath,
                m.executionCombination,
                m.executionQualified,
                m.executionOverride);
    }

    /**
     * The same measurements as {@link #snapshot()}, keyed for the metrics seam — load, prefill,
     * decode, the token counts and, derived from them, the rates.
     *
     * <p>{@link #printMetrics()} was the only way out of this collector, which made the numbers
     * visible to a person and unavailable to a program. A caller that wants to act on throughput —
     * a benchmark harness, a server exposing them, an embedder — reads this instead of parsing the
     * printout.
     *
     * <p>A measurement that was never taken is absent rather than zero: a decode time of zero would
     * claim the run generated nothing in no time, which is a different statement from "generation
     * was not timed".
     */
    public static MetricsReport report() {
        RunMetricsSnapshot snap = snapshot();
        java.util.EnumMap<MetricKey, Long> values = new java.util.EnumMap<>(MetricKey.class);
        putIfMeasured(values, MetricKey.MODEL_LOAD_TIME, snap.loadDuration());
        putIfMeasured(values, MetricKey.PLAN_CREATION_TIME, snap.tornadoPlanCreationDuration());
        putIfMeasured(values, MetricKey.JIT_COMPILE_TIME, snap.tornadoJitDuration());
        putIfMeasured(
                values, MetricKey.WEIGHT_UPLOAD_TIME, snap.tornadoReadOnlyWeightsCopyInDuration());
        putIfMeasured(values, MetricKey.PREFILL_TIME, snap.promptEvalDuration());
        putIfMeasured(values, MetricKey.DECODE_TIME, snap.evalDuration());
        putIfMeasured(values, MetricKey.TOTAL_TIME, snap.totalDuration());
        putIfMeasured(values, MetricKey.PROMPT_TOKENS, snap.promptEvalCount());
        putIfMeasured(values, MetricKey.GENERATED_TOKENS, snap.evalCount());
        return MetricsReport.of(values);
    }

    private static void putIfMeasured(
            java.util.Map<MetricKey, Long> values, MetricKey key, long value) {
        if (value > 0) {
            values.put(key, value);
        }
    }

    // ── Output ────────────────────────────────────────────────────────────────

    /**
     * Builds a snapshot, selects a renderer based on {@code jitllm.metrics.format}, and writes the
     * result to the sink configured by {@code jitllm.metrics.output}.
     */
    public static void printMetrics() {
        RunMetricsSnapshot snap = snapshot();

        MetricsRenderer renderer =
                switch (System.getProperty("jitllm.metrics.format", "human").toLowerCase()) {
                    case "json" -> new JsonMetricsRenderer();
                    case "github" -> new GitHubMetricsRenderer();
                    default -> new HumanMetricsRenderer();
                };

        String rendered = renderer.render(snap);

        switch (System.getProperty("jitllm.metrics.output", "stderr").toLowerCase()) {
            case "stdout" -> System.out.print(rendered);
            case "file" -> writeToFile(rendered);
            default -> System.err.print(rendered);
        }
    }

    private static void writeToFile(String content) {
        String filePath = System.getProperty("jitllm.metrics.file");
        if (filePath == null || filePath.isBlank()) {
            throw new IllegalStateException(
                    "jitllm.metrics.output=file requires jitllm.metrics.file to be set");
        }
        Path path = Path.of(filePath);
        try {
            Path parent = path.getParent();
            if (parent != null) Files.createDirectories(parent);
            Files.writeString(path, content);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write metrics to " + filePath, e);
        }
    }
}
