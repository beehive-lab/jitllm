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

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The capability sets providers reuse — <b>family-neutral</b>, so it names no architecture.
 *
 * <p>Shared constants are allowed to be central; a list of families is not. The difference is what
 * has to change when an architecture is added: nothing here.
 */
public final class TornadoSupportSets {

    /**
     * The two representations the lowered path builds for. A file kept in any other refuses it by
     * name ({@code -Djitllm.lowering=on}); the plan providers take every kept type.
     */
    public static final Set<DataType> BOTH_REPRESENTATIONS = Set.of(DataType.F16, DataType.Q8_0);

    /**
     * The quantized representations a load keeps as the file stores them, each read by its own
     * kernels.
     */
    public static final Set<DataType> NATIVE_QUANTIZED =
            Collections.unmodifiableSet(
                    EnumSet.of(
                            DataType.Q8_0,
                            DataType.Q4_0,
                            DataType.Q4_1,
                            DataType.Q5_0,
                            DataType.Q4_K,
                            DataType.Q5_K,
                            DataType.Q6_K,
                            DataType.IQ4_XS));

    /** F16 and {@link #NATIVE_QUANTIZED}: what a family migrated to native loading admits. */
    public static final Set<DataType> F16_AND_NATIVE_QUANTIZED =
            Collections.unmodifiableSet(
                    EnumSet.of(
                            DataType.F16,
                            DataType.Q8_0,
                            DataType.Q4_0,
                            DataType.Q4_1,
                            DataType.Q5_0,
                            DataType.Q4_K,
                            DataType.Q5_K,
                            DataType.Q6_K,
                            DataType.IQ4_XS));

    /** All three plan shapes. */
    public static final Set<ExecutionMode> EVERY_MODE = EnumSet.allOf(ExecutionMode.class);

    /** Single-token and batched prefill/decode, without the sequential prefill/decode plan. */
    public static final Set<ExecutionMode> STANDARD_AND_BATCH =
            EnumSet.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE);

    /** Single-token only, which is most families today. */
    public static final Set<ExecutionMode> STANDARD_ONLY = EnumSet.of(ExecutionMode.STANDARD);

    private TornadoSupportSets() {}
}
