package org.beehive.jllm.auxiliary.metrics;

/**
 * Renders metrics in human-readable format to {@code stderr}.
 *
 * <p>This is the default renderer — no configuration needed. To enable explicitly:
 *
 * <pre>
 *   -Djllm.metrics.format=human   (default, can be omitted)
 *   -Djllm.metrics.output=stderr  (default, can be omitted)
 * </pre>
 *
 * <p>Startup timings are printed once by the CLI before generation.
 */
public final class HumanMetricsRenderer implements MetricsRenderer {

    @Override
    public String render(RunMetricsSnapshot s) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n==== Performance Metrics ====\n");

        if (s.hasPrefillPhase()) {
            sb.append(
                    String.format(
                            "Total achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n"
                                    + "¬Prefill achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n"
                                    + "¬Decode achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                            s.totalRate(),
                            s.totalCount(),
                            s.totalDuration() / 1e9,
                            s.promptEvalRate(),
                            s.promptEvalCount(),
                            s.promptEvalDuration() / 1e9,
                            s.evalRate(),
                            s.evalCount(),
                            s.evalDuration() / 1e9));
        } else {
            sb.append(
                    String.format(
                            "achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                            s.totalRate(), s.totalCount(), s.totalDuration() / 1e9));
        }

        return sb.toString();
    }
}
