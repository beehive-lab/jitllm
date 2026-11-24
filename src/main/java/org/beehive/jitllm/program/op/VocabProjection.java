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
 * Projecting the final hidden state onto the vocabulary to produce logits.
 *
 * <p>Shaped like a {@link MatVec} and kept separate anyway, for two reasons that are not cosmetic.
 * It is the operation a phase <i>skips</i> — prefill runs every layer and stops before this one —
 * so a program that cannot name it cannot express phase selection. And its weight is frequently the
 * embedding table again, tied, which is a fact about this projection specifically.
 *
 * @param weight the output projection; often the tied embedding table
 * @param input the final hidden state
 * @param output the logits
 * @param vocabularySize the number of logits produced
 * @param dataType the representation the projection weight was materialized in
 */
public record VocabProjection(
        OperandRef.Weight weight,
        OperandRef input,
        OperandRef output,
        int vocabularySize,
        DataType dataType)
        implements Operation {

    public VocabProjection {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (vocabularySize <= 0) {
            throw new IllegalArgumentException(
                    "vocabularySize must be positive: " + vocabularySize);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.VOCAB_PROJECTION;
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
