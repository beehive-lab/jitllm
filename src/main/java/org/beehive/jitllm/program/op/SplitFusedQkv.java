package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Separating a fused query/key/value projection into its three parts.
 *
 * <p>Phi3 projects Q, K and V with <b>one</b> weight matrix of width {@code queryWidth + 2 *
 * keyValueWidth}, then reads the three vectors out of the single result. The projection is
 * describable — the {@code ATTENTION_QKV} role names that matrix — but the split that follows it
 * was not, and a description that stopped at the projection would claim the model computes
 * something it does not.
 *
 * <h2>Why this is an operation and not materialization</h2>
 *
 * <p><b>A backend may fuse it into the projection and need not create an intermediate.</b> Fusion
 * is many-to-many and backend-owned, so a lowering that emits one task for the projection and the
 * split together is doing exactly what it should. What is not optional is that the program says the
 * split happens.
 *
 * <h2>Narrow on purpose</h2>
 *
 * @param fused the projection result holding query, key and value end to end
 * @param query where the query part is written
 * @param key where the key part is written
 * @param value where the value part is written
 * @param queryWidth elements belonging to the query
 * @param keyValueWidth elements belonging to each of the key and the value
 * @param dataType the representation the copy executes at
 */
public record SplitFusedQkv(
        OperandRef fused,
        OperandRef query,
        OperandRef key,
        OperandRef value,
        int queryWidth,
        int keyValueWidth,
        DataType dataType)
        implements Operation {

    public SplitFusedQkv {
        Objects.requireNonNull(fused, "fused");
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        if (queryWidth < 1) {
            throw new IllegalArgumentException("queryWidth must be at least 1: " + queryWidth);
        }
        if (keyValueWidth < 1) {
            throw new IllegalArgumentException(
                    "keyValueWidth must be at least 1: " + keyValueWidth);
        }
    }

    /** The width of the fused projection this reads, which is what the projection must produce. */
    public int fusedWidth() {
        return queryWidth + 2 * keyValueWidth;
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SPLIT_FUSED_QKV;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(fused);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(query, key, value);
    }
}
