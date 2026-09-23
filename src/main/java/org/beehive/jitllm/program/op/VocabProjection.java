package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

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
