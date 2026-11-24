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
 * Stochastic selection of the next token from the logits.
 *
 * <p><b>Temperature, top-p and the seed are not here.</b> They change per request and are delivered
 * as invocation values — written into a persistent control array before the invocation, never by
 * rebinding one. Carrying them in the description would put them in the program signature, and
 * therefore in the compiled-program cache key, which would compile a separate program for every
 * temperature a caller ever asks for.
 *
 * <p>That is the whole difference from {@link ArgMax}: same operands, parameters supplied later.
 * Which of the two a program contains is an execution-policy choice resolved once at session
 * creation, not a per-token branch.
 *
 * @param logits the scores to sample from
 * @param output where the chosen identifier is written
 * @param dataType the representation the logits are held in
 */
public record Sample(OperandRef logits, OperandRef output, DataType dataType) implements Operation {

    public Sample {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SAMPLE;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(logits);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
