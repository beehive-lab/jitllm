package org.beehive.jllm.auxiliary.metrics;

/**
 * Renders metrics as a Markdown table suitable for appending to {@code $GITHUB_STEP_SUMMARY}.
 * TornadoVM rows (compile, JIT, weight copy-in) are included only when plan-creation duration is
 * non-zero, i.e. on GPU runs.
 *
 * <p>Enable via system properties and append the output file to the step summary:
 *
 * <pre>
 *   -Djllm.metrics.format=github
 *   -Djllm.metrics.output=file
 *   -Djllm.metrics.file=/tmp/metrics.md
 * </pre>
 *
 * <p>In a GitHub Actions workflow step:
 *
 * <pre>
 *   cat /tmp/metrics.md >> $GITHUB_STEP_SUMMARY
 * </pre>
 */
public final class GitHubMetricsRenderer implements MetricsRenderer {

    @Override
    public String render(RunMetricsSnapshot s) {
        StringBuilder sb = new StringBuilder();
        sb.append("| metric | value |\n");
        sb.append("|---|---:|\n");
        sb.append(String.format("| eval tok/s | %.2f |%n", s.evalRate()));
        sb.append(String.format("| prompt eval tok/s | %.2f |%n", s.promptEvalRate()));
        sb.append(String.format("| total tok/s | %.2f |%n", s.totalRate()));
        sb.append(String.format("| load ms | %.2f |%n", s.loadDuration() / 1_000_000.0));
        sb.append(String.format("| eval tokens | %d |%n", s.evalCount()));
        sb.append(String.format("| prompt tokens | %d |%n", s.promptEvalCount()));
        sb.append(String.format("| total tokens | %d |%n", s.totalCount()));
        if (s.tornadoPlanCreationDuration() > 0) {
            sb.append(
                    String.format(
                            "| compile ms | %.2f |%n",
                            s.tornadoPlanCreationDuration() / 1_000_000.0));
            sb.append(String.format("| jit ms | %.2f |%n", s.tornadoJitDuration() / 1_000_000.0));
            sb.append(
                    String.format(
                            "| weight copy-in ms | %.2f |%n",
                            s.tornadoReadOnlyWeightsCopyInDuration() / 1_000_000.0));
        }
        return sb.toString();
    }
}
