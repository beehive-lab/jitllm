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
 * A weight matrix times a batch of activation rows.
 *
 * <p>What prefill and batched decode do, and what the tensor-core paths exist for. The batch size
 * is part of the description rather than an invocation value because the buffers are sized for it
 * and the plan is captured for it: a step may have fewer active rows than the batch it was built
 * for, but it may never have more.
 *
 * @param weight the weight matrix
 * @param input the activation rows, {@code batchSize} of them
 * @param output the resulting rows
 * @param rows rows of the weight matrix
 * @param columns columns of the weight matrix, and the width of one activation row
 * @param batchSize the number of rows the buffers are sized for
 * @param dataType the representation the weight matrix was materialized in
 */
public record MatMul(
        OperandRef.Weight weight,
        OperandRef input,
        OperandRef output,
        int rows,
        int columns,
        int batchSize,
        DataType dataType)
        implements Operation {

    public MatMul {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException(
                    "rows and columns must be positive: " + rows + "x" + columns);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.MAT_MUL;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(weight, input);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
