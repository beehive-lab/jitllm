/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.auxiliary.metrics;

/**
 * Renders metrics as a Markdown table suitable for appending to {@code $GITHUB_STEP_SUMMARY}.
 * TornadoVM rows (compile, JIT, weight copy-in) are included only when plan-creation duration is
 * non-zero, i.e. on GPU runs.
 *
 * <p>Enable via system properties and append the output file to the step summary:
 *
 * <pre>
 *   -Djitllm.metrics.format=github
 *   -Djitllm.metrics.output=file
 *   -Djitllm.metrics.file=/tmp/metrics.md
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
