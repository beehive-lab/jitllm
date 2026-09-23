package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Rotary position embedding, applied in place to the query and key projections.
 *
 * <p>The rotation angle depends on the position, which is an invocation value rather than part of
 * the description: the same program runs at every position. What the description fixes is the
 * geometry — the head dimension and the base frequency.
 *
 * <p><b>The layout belongs to the model too.</b> Adjacent components are paired on the Llama family
 * and half-a-head apart on the Qwen and Gemma families; see {@link RopeLayout}. Both read valid
 * floats, so choosing wrong is silent.
 *
 * <p><b>The base belongs to the model.</b> It is carried here rather than assumed because assuming
 * it is a defect this project has already paid for five times: hard-coded bases of 50000 and
 * 1000000 sat in the Llama, Mistral, Qwen2, Qwen3 and batch-prefill kernels against model {@code
 * rope_theta} values of 500000 and 10000, and DeepSeek-R1-Distill-Qwen produced unrelated tokens as
 * a result. An operation that takes the base cannot repeat it.
 *
 * @param query the query projection, rotated in place
 * @param key the key projection, rotated in place
 * @param frequencies precomputed sine/cosine tables, when the backend uses them rather than
 *     computing the angles inline; empty when it does not
 * @param headDimension the size of one attention head
 * @param ropeTheta the model's rotary base frequency
 * @param layout which two components form a rotated pair
 * @param dataType the representation the rotation executes at
 */
public record RoPE(
        OperandRef query,
        OperandRef key,
        Optional<OperandRef.Weight> frequencies,
        int headDimension,
        float ropeTheta,
        RopeLayout layout,
        DataType dataType)
        implements Operation {

    public RoPE {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(frequencies, "frequencies");
        Objects.requireNonNull(layout, "layout");
        if (headDimension <= 0) {
            throw new IllegalArgumentException("headDimension must be positive: " + headDimension);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.ROPE;
    }

    @Override
    public List<OperandRef> inputs() {
        return frequencies
                .<List<OperandRef>>map(f -> List.of(query, key, f))
                .orElseGet(() -> List.of(query, key));
    }

    /** In place: the rotated projections are the same operands the rotation read. */
    @Override
    public List<OperandRef> outputs() {
        return List.of(query, key);
    }
}
