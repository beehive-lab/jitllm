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
 * Elementwise addition of a residual branch back into the main activation.
 *
 * <p>Trivial arithmetic, named anyway: it is where the residual stream is, and a program that does
 * not say so is a program whose data flow has to be inferred from buffer names.
 *
 * @param left the main activation
 * @param right the branch result being added back
 * @param output the sum; may be {@code left}
 * @param dataType the representation the residual stream is held in
 */
public record ResidualAdd(OperandRef left, OperandRef right, OperandRef output, DataType dataType)
        implements Operation {

    public ResidualAdd {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.RESIDUAL_ADD;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(left, right);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
