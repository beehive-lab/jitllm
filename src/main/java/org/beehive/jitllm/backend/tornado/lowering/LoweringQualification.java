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
package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Which exact combinations have earned the lowered path as their default [D-1, D-2, D-3].
 *
 * <h2>A table, not conditionals</h2>
 */
public final class LoweringQualification {

    /**
     * One combination, at the granularity qualification is decided: architecture, <b>materialized
     * dtype</b> and execution mode. Never the family alone — Llama F16 and Llama Q8_0 compile
     * different graphs from different weight representations, which is what {@code
     * LoweredWeightRepresentationAccelTest} exists to prove.
     */
    public record Combination(ArchitectureId architecture, DataType dtype, ExecutionMode mode) {

        @Override
        public String toString() {
            return architecture.name() + "/" + dtype.name() + "/" + mode.name();
        }
    }

    /**
     * The qualified set.
     *
     * <p><b>Llama / F16 / STANDARD, and nothing else.</b> Measured 2026-09-02: lowered 144.37 tok/s
     * against legacy 144.62, a −0.2% delta inside the 3.0% tolerance, five interleaved pairs on one
     * build differentiated by the property alone, with all five candidate runs positively reporting
     * {@code execution_path=lowered}. Its other five gates were already green.
     *
     * <p>The other ten implemented combinations are missing at least one gate — most of them the
     * paired-performance one, which has never been run for any of them. They select legacy under
     * {@link LoweringMode#AUTO} until their evidence exists.
     */
    private static final Set<Combination> QUALIFIED =
            Set.of(
                    new Combination(
                            ArchitectureId.of("llama"), DataType.F16, ExecutionMode.STANDARD));

    private LoweringQualification() {}

    /** Whether this exact combination may lower by default. */
    public static boolean isQualified(
            ArchitectureId architecture, DataType dtype, ExecutionMode mode) {
        return QUALIFIED.contains(new Combination(architecture, dtype, mode));
    }

    /** The qualified set, for tests and diagnostics. */
    public static Set<Combination> qualified() {
        return QUALIFIED;
    }
}
