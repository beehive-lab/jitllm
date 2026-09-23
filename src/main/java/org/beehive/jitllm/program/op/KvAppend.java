package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Writing this step's key and value into the retained key/value store.
 *
 * <h2>Why this is not part of {@link Attention}</h2>
 *
 * <p>So the sequence is:
 *
 * <pre>
 *   key/value projection and normalization  →  KvAppend  →  Attention over a declared source
 * </pre>
 *
 * <p>An ordinary family appends to, and attends over, its own layer. A reuse layer performs neither
 * the projection nor this operation, and its attention reads the layer that did.
 *
 * <h2>Ordering is a real dependency</h2>
 *
 * @param key this step's key projection, after normalization and rotation
 * @param value this step's value projection
 * @param keyStore where keys are retained
 * @param valueStore where values are retained
 * @param width how many elements one position occupies — all key/value heads together
 * @param dataType the representation the key/value store holds
 */
public record KvAppend(
        OperandRef key,
        OperandRef value,
        OperandRef keyStore,
        OperandRef valueStore,
        int width,
        DataType dataType)
        implements Operation {

    public KvAppend {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(keyStore, "keyStore");
        Objects.requireNonNull(valueStore, "valueStore");
        Objects.requireNonNull(dataType, "dataType");
        if (width <= 0) {
            throw new IllegalArgumentException("width must be positive: " + width);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.KV_APPEND;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(key, value);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(keyStore, valueStore);
    }
}
