package org.beehive.jllm.auxiliary.metrics;

import java.util.ArrayList;
import java.util.List;

/**
 * Renders metrics as an Ollama-compatible JSON object. All duration fields are in nanoseconds; rate
 * fields are in tokens per second. The {@code tornadovm} nested object is always included (fields
 * are zero on CPU runs).
 *
 * <p>Enable via system properties:
 *
 * <pre>
 *   -Djllm.metrics.format=json
 *   -Djllm.metrics.output=stdout   # pipe to jq or another tool
 * </pre>
 *
 * <p>Or write to a file:
 *
 * <pre>
 *   -Djllm.metrics.format=json
 *   -Djllm.metrics.output=file
 *   -Djllm.metrics.file=/path/to/metrics.json
 * </pre>
 */
public final class JsonMetricsRenderer implements MetricsRenderer {

    @Override
    public String render(RunMetricsSnapshot s) {
        List<String> fields = new ArrayList<>();
        fields.add(field("  ", "total_duration", s.totalDuration()));
        fields.add(field("  ", "load_duration", s.loadDuration()));
        fields.add(field("  ", "prompt_eval_count", s.promptEvalCount()));
        fields.add(field("  ", "prompt_eval_duration", s.promptEvalDuration()));
        fields.add(field("  ", "eval_count", s.evalCount()));
        fields.add(field("  ", "eval_duration", s.evalDuration()));
        fields.add(field("  ", "total_count", s.totalCount()));
        fields.add(field("  ", "prompt_eval_rate", s.promptEvalRate()));
        fields.add(field("  ", "eval_rate", s.evalRate()));
        fields.add(field("  ", "total_rate", s.totalRate()));
        fields.add(field("  ", "has_prefill_phase", s.hasPrefillPhase()));
        if (s.executionPath() != null) {
            fields.add(field("  ", "execution_path", s.executionPath()));
        }
        if (s.executionCombination() != null) {
            fields.add(field("  ", "execution_combination", s.executionCombination()));
        }
        if (s.executionQualified() != null) {
            fields.add(field("  ", "execution_qualified", s.executionQualified().booleanValue()));
        }
        if (s.executionOverride() != null) {
            fields.add(field("  ", "execution_override", s.executionOverride()));
        }
        fields.add(tornadoObject(s));
        return "{\n" + String.join(",\n", fields) + "\n}";
    }

    private static String tornadoObject(RunMetricsSnapshot s) {
        List<String> inner = new ArrayList<>();
        inner.add(field("    ", "plan_creation_duration", s.tornadoPlanCreationDuration()));
        inner.add(field("    ", "jit_duration", s.tornadoJitDuration()));
        inner.add(
                field(
                        "    ",
                        "read_only_weights_copy_in_duration",
                        s.tornadoReadOnlyWeightsCopyInDuration()));
        return "  \"tornadovm\": {\n" + String.join(",\n", inner) + "\n  }";
    }

    private static String field(String indent, String key, long value) {
        return indent + "\"" + key + "\": " + value;
    }

    private static String field(String indent, String key, int value) {
        return indent + "\"" + key + "\": " + value;
    }

    private static String field(String indent, String key, double value) {
        return indent + "\"" + key + "\": " + String.format("%.4f", value);
    }

    private static String field(String indent, String key, String value) {
        return indent + "\"" + key + "\": \"" + value + "\"";
    }

    private static String field(String indent, String key, boolean value) {
        return indent + "\"" + key + "\": " + value;
    }
}
