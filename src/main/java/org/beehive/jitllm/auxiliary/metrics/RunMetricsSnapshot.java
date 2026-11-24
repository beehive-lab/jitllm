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
 * Immutable snapshot of all performance metrics for a single inference run.
 *
 * <p>Raw duration fields are in nanoseconds. Derived rate fields are in tokens per second.
 * Construct via {@link #of} — it computes the derived values from the raw inputs.
 */
public record RunMetricsSnapshot(
        long totalDuration,
        long loadDuration,
        int promptEvalCount,
        long promptEvalDuration,
        int evalCount,
        long evalDuration,
        boolean hasPrefillPhase,
        long tornadoPlanCreationDuration,
        long tornadoJitDuration,
        long tornadoReadOnlyWeightsCopyInDuration,
        String executionPath,
        String executionCombination,
        Boolean executionQualified,
        String executionOverride,
        // derived
        int totalCount,
        double promptEvalRate,
        double evalRate,
        double totalRate) {
    public static RunMetricsSnapshot of(
            long totalDuration,
            long loadDuration,
            int promptEvalCount,
            long promptEvalDuration,
            int evalCount,
            long evalDuration,
            boolean hasPrefillPhase,
            long tornadoPlanCreationDuration,
            long tornadoJitDuration,
            long tornadoReadOnlyWeightsCopyInDuration,
            String executionPath,
            String executionCombination,
            Boolean executionQualified,
            String executionOverride) {

        int totalCount = promptEvalCount + evalCount;
        double promptEvalRate = tokensPerSecond(promptEvalCount, promptEvalDuration);
        double evalRate = tokensPerSecond(evalCount, evalDuration);
        double totalRate = tokensPerSecond(totalCount, totalDuration);

        return new RunMetricsSnapshot(
                totalDuration,
                loadDuration,
                promptEvalCount,
                promptEvalDuration,
                evalCount,
                evalDuration,
                hasPrefillPhase,
                tornadoPlanCreationDuration,
                tornadoJitDuration,
                tornadoReadOnlyWeightsCopyInDuration,
                executionPath,
                executionCombination,
                executionQualified,
                executionOverride,
                totalCount,
                promptEvalRate,
                evalRate,
                totalRate);
    }

    private static double tokensPerSecond(int tokens, long durationNs) {
        return (durationNs > 0 && tokens > 0) ? tokens / (durationNs / 1e9) : 0.0;
    }
}
