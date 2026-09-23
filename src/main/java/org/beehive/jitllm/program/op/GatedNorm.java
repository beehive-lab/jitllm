package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Normalization gated by a second branch: {@code rms_norm(input, weight) * silu(gate)}, per group.
 *
 * <p>Neither half on its own. {@link SwiGLU} multiplies by {@code silu(gate)} but normalizes
 * nothing; composing {@link RmsNorm} with it needs an intermediate the fused form does not, and
 * would describe two operations where a backend runs one. Qwen3.5's recurrent branch ends this way,
 * normalizing per value head against a weight the heads share.
 *
 * @param input the branch output, normalized and gated
 * @param gate the parallel branch, read through a SiLU
 * @param weight the learned scale, one group's width, shared across groups
 * @param output where the result is written; may be {@code input}
 * @param groups how many groups are normalized independently
 * @param groupLength how many elements one group covers
 * @param epsilon the variance epsilon
 * @param dataType the representation this executes at
 */
public record GatedNorm(
        OperandRef input,
        OperandRef gate,
        OperandRef.Weight weight,
        OperandRef output,
        int groups,
        int groupLength,
        float epsilon,
        DataType dataType)
        implements Operation {

    public GatedNorm {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (groups < 1) {
            throw new IllegalArgumentException("groups must be at least 1: " + groups);
        }
        if (groupLength < 1) {
            throw new IllegalArgumentException("groupLength must be at least 1: " + groupLength);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.GATED_NORM;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input, gate, weight);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
