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
 * Softmax over the last dimension, one row at a time.
 *
 * <p>Listed as its own operation because it is one on the CPU path, where attention scores are
 * normalized as a separate step. A backend that fuses it into {@link Attention} — as the
 * flash-attention style kernels do — simply has no separate implementation to reach, which is a
 * property of that backend's mapping and not a reason to leave the operation unnamed.
 *
 * @param input the scores
 * @param output the normalized scores; may be {@code input}
 * @param rows how many independent rows are normalized
 * @param columns the length of one row
 * @param dataType the representation the scores are held in
 */
public record Softmax(OperandRef input, OperandRef output, int rows, int columns, DataType dataType)
        implements Operation {

    public Softmax {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException(
                    "rows and columns must be positive: " + rows + "x" + columns);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SOFTMAX;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
