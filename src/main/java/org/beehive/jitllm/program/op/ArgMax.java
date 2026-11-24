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
 * Greedy selection: the identifier of the highest-scoring logit.
 *
 * <p>Sampling is an operation and may execute on the device (Rule 8b), which is what makes the
 * existing {@code jitllm.deviceSample} path expressible rather than a special case. Rule 14 is not
 * in tension with that: it forbids core abstractions from <i>requiring</i> a sampler, not from
 * naming one.
 *
 * <p>Greedy selection takes no parameters, which is exactly what separates it from {@link Sample}.
 *
 * @param logits the scores to choose from
 * @param output where the chosen identifier is written
 * @param dataType the representation the logits are held in
 */
public record ArgMax(OperandRef logits, OperandRef output, DataType dataType) implements Operation {

    public ArgMax {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.ARG_MAX;
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
