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
 * Multiplying every element by one scalar from the model's configuration.
 *
 * <p>Granite is why this exists. It applies maximal-update-parameterization (µP) factors at four
 * points a Llama-shaped model has none: after the embedding lookup, on each residual branch before
 * it is added back, and on the logits. Gemma scales its embeddings the same way.
 *
 * <p><b>The factor is configuration, not an invocation value.</b> It comes from the model file and
 * is the same for every token, so it belongs in the description — unlike a sampling temperature,
 * which changes per request and would compile a program per value.
 *
 * <p>Deliberately not folded into the neighbouring operation. A "{@code MatVec} with an optional
 * output scale" would give every family a parameter only one uses, and an implementation that has
 * to check it. Naming the scale keeps the families that do not scale free of it, and makes
 * Granite's four scaling points visible in the program rather than hidden inside four other
 * operations.
 *
 * @param input the values to scale
 * @param output the scaled values; may be {@code input}
 * @param factor the scalar from the model configuration
 * @param dataType the representation the multiplication executes at
 */
public record Scale(OperandRef input, OperandRef output, float factor, DataType dataType)
        implements Operation {

    public Scale {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SCALE;
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
