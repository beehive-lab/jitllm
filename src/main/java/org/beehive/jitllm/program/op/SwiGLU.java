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
package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The SwiGLU feed-forward activation: {@code silu(gate) * up}, elementwise.
 *
 * <p>Named for what every supported family actually computes. A family using a different gate
 * activation would be a new kind rather than a parameter on this one — the set of activation
 * functions in use is small enough that a function selector here would be a dispatch key with one
 * value.
 *
 * @param gate the gate projection, to which SiLU is applied
 * @param up the up projection, multiplied in
 * @param output the activated hidden state; may be {@code gate}
 * @param dataType the representation the hidden state is held in
 */
public record SwiGLU(OperandRef gate, OperandRef up, OperandRef output, DataType dataType)
        implements Operation {

    public SwiGLU {
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SWIGLU;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(gate, up);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
