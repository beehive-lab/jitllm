package org.beehive.jllm.format;

import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.runtime.tensor.Shape;
import org.beehive.jllm.runtime.tensor.TensorDescriptor;
import org.beehive.jllm.runtime.tensor.TensorLayout;
import org.beehive.jllm.runtime.tensor.TensorRole;

/**
 * Builds a runtime {@link TensorDescriptor} from a GGUF tensor entry.
 *
 * <p>The format side of {@code GGMLTensorEntry ──map──▶ TensorDescriptor ──materialize──▶ storage}
 * It names both vocabularies, as {@link DataTypeMapping} does, and for the same reason: it is the
 * boundary itself.
 */
public final class TensorDescriptors {

    private TensorDescriptors() {}

    /**
     * What this tensor will be once loaded for {@code target} — the data type its storage will
     * hold, with the layout that matches it.
     */
    public static TensorDescriptor describe(GGMLTensorEntry entry, ExecutionTarget target) {
        DataType dataType = DataTypeMapping.materializedType(entry.ggmlType(), target);
        return new TensorDescriptor(
                entry.name(),
                dataType,
                Shape.of(entry.shape()),
                roleOf(entry.name()),
                layoutOf(dataType));
    }

    /** What the file holds, before any conversion — the same tensor as the loader found it. */
    public static TensorDescriptor describeSource(GGMLTensorEntry entry) {
        DataType dataType = DataTypeMapping.sourceType(entry.ggmlType());
        return new TensorDescriptor(
                entry.name(),
                dataType,
                Shape.of(entry.shape()),
                roleOf(entry.name()),
                layoutOf(dataType));
    }

    /**
     * The layout a representation is stored in.
     *
     * <p>Block sizes come from the format's own type table rather than being restated here: the
     * file is where they are defined, and a second copy would be a second thing to get wrong.
     */
    public static TensorLayout layoutOf(DataType dataType) {
        GGMLType fileType = DataTypeMapping.asFileType(dataType);
        if (!dataType.isQuantized()) {
            return new TensorLayout.Dense(fileType.getTypeSize());
        }
        TensorLayout.ScaleFormat scale =
                switch (dataType) {
                    case Q8_0, Q4_0 -> TensorLayout.ScaleFormat.FP16;
                    default -> TensorLayout.ScaleFormat.HIERARCHICAL;
                };
        return new TensorLayout.BlockQuantized(
                fileType.getBlockSize(), fileType.getTypeSize(), scale);
    }

    /**
     * The role a GGUF tensor name denotes.
     *
     * <p>Derived centrally from the name rather than passed in by each family loader: the names are
     * a format convention, all seven loaders already spell them the same way, and threading a role
     * argument through every call site would be seven chances to disagree. A name this does not
     * recognize is {@link TensorRole#OTHER}, never a guess.
     */
    public static TensorRole roleOf(String tensorName) {
        if (tensorName == null) {
            return TensorRole.OTHER;
        }
        String suffix =
                tensorName.startsWith("blk.")
                        ? tensorName.substring(tensorName.indexOf('.', 4) + 1)
                        : tensorName;
        return switch (suffix) {
            case "token_embd.weight" -> TensorRole.TOKEN_EMBEDDING;
            case "attn_norm.weight" -> TensorRole.ATTENTION_NORM;
            case "attn_q.weight" -> TensorRole.ATTENTION_QUERY;
            case "attn_k.weight" -> TensorRole.ATTENTION_KEY;
            case "attn_v.weight" -> TensorRole.ATTENTION_VALUE;
            case "attn_qkv.weight" -> TensorRole.ATTENTION_QKV;
            case "attn_q_norm.weight" -> TensorRole.ATTENTION_QUERY_NORM;
            case "attn_k_norm.weight" -> TensorRole.ATTENTION_KEY_NORM;
            case "attn_output.weight" -> TensorRole.ATTENTION_OUTPUT;
            case "ffn_norm.weight" -> TensorRole.FFN_NORM;
            case "ffn_gate.weight" -> TensorRole.FFN_GATE;
            case "ffn_up.weight" -> TensorRole.FFN_UP;
            case "ffn_down.weight" -> TensorRole.FFN_DOWN;
            case "output_norm.weight" -> TensorRole.OUTPUT_NORM;
            case "output.weight" -> TensorRole.OUTPUT;
            default -> TensorRole.OTHER;
        };
    }
}
