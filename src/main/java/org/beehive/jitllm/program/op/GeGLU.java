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
 * The GeGLU feed-forward activation: {@code gelu(gate) * up}, elementwise.
 *
 * <p>Gemma4's gate. Structurally {@link SwiGLU} with a different activation on the gate half, and a
 * separate operation rather than a parameter on it: exactly two gated activations exist across all
 * eight families, and an enum over two values would turn {@code SwiGLU} from a <i>name</i> into a
 * <i>parameter value</i>. If a third ever arrives, generalizing then is a mechanical change over
 * two call sites with equivalence tests already in place.
 *
 * @param gate the gate projection, to which GeLU is applied
 * @param up the up projection, multiplied in
 * @param output the activated hidden state; may be {@code gate}
 * @param dataType the representation the activation executes at
 */
public record GeGLU(OperandRef gate, OperandRef up, OperandRef output, DataType dataType)
        implements Operation {

    public GeGLU {
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.GEGLU;
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
