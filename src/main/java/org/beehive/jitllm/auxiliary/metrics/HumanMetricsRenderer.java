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
 * Renders metrics in human-readable format to {@code stderr}.
 *
 * <p>This is the default renderer — no configuration needed. To enable explicitly:
 *
 * <pre>
 *   -Djitllm.metrics.format=human   (default, can be omitted)
 *   -Djitllm.metrics.output=stderr  (default, can be omitted)
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
