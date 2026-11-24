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
 * Reading rows out of the token embedding table.
 *
 * <p>Where a forward pass begins. The token identifiers are an operand rather than a parameter
 * because they change every invocation and are delivered by writing into a control array, never by
 * rebinding one.
 *
 * <p>Some families scale the embedding immediately after the lookup. That scale belongs to the
 * architecture assembling the program — as a following operation, or as part of how the family
 * stores its table — not to the meaning of "look up a row".
 *
 * @param table the embedding table
 * @param tokenIds the identifiers to look up; one for decode, several for prefill
 * @param output the embeddings
 * @param embeddingDimension the width of one embedding row
 * @param dataType the representation the embedding table was materialized in
 */
public record EmbeddingLookup(
        OperandRef.Weight table,
        OperandRef tokenIds,
        OperandRef output,
        int embeddingDimension,
        DataType dataType)
        implements Operation {

    public EmbeddingLookup {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (embeddingDimension <= 0) {
            throw new IllegalArgumentException(
                    "embeddingDimension must be positive: " + embeddingDimension);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.EMBEDDING_LOOKUP;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(table, tokenIds);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
