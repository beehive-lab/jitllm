package org.beehive.jllm.format;

import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;

/**
 * The one place that says what a file's tensor becomes when it is loaded.
 *
 * <p>This mapping lives on the format side of the boundary, because it is the only thing allowed to
 * name both vocabularies: {@link GGMLType} describes what is in a GGUF file, {@link DataType}
 * describes what the engine executes, and nothing in the runtime, program, operation or backend
 * layers may see the former (Rule 4).
 *
 * <h2>Why the target is a parameter</h2>
 *
 * <p>Until now this logic existed as {@code AbstractModelLoader.effectiveGpuWeightType} — a switch
 * with no test and no CPU half. That method now delegates here.
 */
public final class DataTypeMapping {

    private DataTypeMapping() {}

    /**
     * What the file holds, in the runtime's vocabulary — before any conversion.
     *
     * <p>May be a block-encoded quantization: that is the honest answer for a K-quant file, and it
     * is what the CPU path goes on to execute.
     *
     * @throws UnsupportedOperationException naming the type, for a format nothing here executes
     */
    public static DataType sourceType(GGMLType fileType) {
        return switch (fileType) {
            case F32 -> DataType.F32;
            case F16 -> DataType.F16;
            case BF16 -> DataType.BF16;
            case Q8_0 -> DataType.Q8_0;
            case Q4_0 -> DataType.Q4_0;
            case Q4_1 -> DataType.Q4_1;
            case Q4_K -> DataType.Q4_K;
            case Q5_K -> DataType.Q5_K;
            case Q6_K -> DataType.Q6_K;
            default ->
                    throw new UnsupportedOperationException(
                            "No runtime representation for GGUF type "
                                    + fileType
                                    + "; it is neither executed nor materialized by this engine");
        };
    }

    /**
     * What the tensor is materialized as for {@code target} — the type its storage will actually
     * hold, and the type an operation on it is parameterized by.
     *
     * <p>On the CPU this is the source type: the host decodes blocks during compute, so nothing is
     * converted at load, and {@link DataType#narrowedFallback()} says which.
     *
     * <p><b>This is the legacy answer, and it is not what a family with native kernels uses.</b>
     * The backend has device storage and matrix-vector kernels for every quantization the engine
     * recognizes, so a tensor can be — and normally should be — kept in the layout the file gave
     * it, through {@code ModelLoader.loadTornadoTensorNative}. What remains here is the promotion
     * to {@link DataType#Q8_0} that families still on the older loading path rely on, which costs
     * roughly double the device memory for a 4-bit file.
     *
     * <p>It is named rather than silent: a caller asking this question is asking "what does the
     * <i>converting</i> path produce", and the conversion appears in the memory plan. Migrating the
     * remaining loaders off it is what removes the GPU branch entirely.
     */
    public static DataType materializedType(GGMLType fileType, ExecutionTarget target) {
        DataType source = sourceType(fileType);
        return switch (target) {
            case CPU -> source;
            case GPU -> legacyDevicePromotion(source);
        };
    }

    /**
     * What the older, converting device path turns a representation into.
     *
     * <p>Kept out of {@link DataType} deliberately. It is not a property of the representation —
     * nothing about Q4_K implies Q8_0 — but a property of a loading path that predates native
     * device storage.
     */
    private static DataType legacyDevicePromotion(DataType source) {
        return switch (source) {
            case Q4_0, Q4_1, Q4_K, Q5_K, Q6_K -> DataType.Q8_0;
            default -> source.narrowedFallback();
        };
    }

    /**
     * Whether {@code target} can run this file type at all, with or without conversion.
     *
     * <p>True for everything {@link #sourceType} recognizes today. It is a separate question from
     * {@link #materializedType} on purpose: a target that cannot execute a representation and
     * cannot materialize it either must be an error at load, not a wrong answer.
     */
    public static boolean isSupported(GGMLType fileType, ExecutionTarget target) {
        try {
            materializedType(fileType, target);
            return true;
        } catch (UnsupportedOperationException notSupported) {
            return false;
        }
    }

    /**
     * The representation activations are held in for a model whose weights are {@code fileType}.
     *
     * <p>Not the same question as the weights' type: an FP16 model keeps FP16 activations, and
     * everything quantized quantizes its activations to Q8_0 to match the kernels that consume
     * them. This mirrors the string switch in {@code AbstractModelLoader.getModelQuantization},
     * which drives the activation buffers a {@code State} allocates.
     */
    public static DataType activationType(GGMLType fileType) {
        DataType source = sourceType(fileType);
        return switch (source) {
            case F32, F16, BF16 -> DataType.F16;
            default -> DataType.Q8_0;
        };
    }

    /**
     * The GGUF type corresponding to a runtime type — the reverse direction, needed only while the
     * loaders and weight classes still speak {@link GGMLType}.
     */
    public static GGMLType asFileType(DataType dataType) {
        return switch (dataType) {
            case F32 -> GGMLType.F32;
            case F16 -> GGMLType.F16;
            case BF16 -> GGMLType.BF16;
            case Q8_0 -> GGMLType.Q8_0;
            case Q4_0 -> GGMLType.Q4_0;
            case Q4_1 -> GGMLType.Q4_1;
            case Q4_K -> GGMLType.Q4_K;
            case Q5_K -> GGMLType.Q5_K;
            case Q6_K -> GGMLType.Q6_K;
        };
    }
}
