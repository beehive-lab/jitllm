package org.beehive.jitllm.api;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * What a loaded model is: enough to identify it and to size a request against it.
 *
 * <p>Immutable and thread-safe.
 *
 * <p>The dtype accessors report <b>what executes</b>, not what the file holds. A Q6_K or Q4_0 file
 * loaded for the GPU says {@link DataType#Q8_0}, because that is what its weights were materialized
 * as and what the kernels read. The file's own type is a different question and is not answered
 * here.
 */
@Experimental
public final class ModelInfo {

    private final String name;
    private final String architecture;
    private final int contextLength;
    private final Path source;
    private final Set<DataType> weightTypes;
    private final DataType computeType;

    public ModelInfo(
            String name,
            String architecture,
            int contextLength,
            Path source,
            Set<DataType> weightTypes,
            DataType computeType) {
        this.name = Objects.requireNonNull(name, "name");
        this.architecture = Objects.requireNonNull(architecture, "architecture");
        this.contextLength = contextLength;
        this.source = source;
        this.weightTypes = Set.copyOf(Objects.requireNonNull(weightTypes, "weightTypes"));
        this.computeType = Objects.requireNonNull(computeType, "computeType");
        if (this.weightTypes.isEmpty()) {
            throw new IllegalArgumentException("a model has at least one weight representation");
        }
    }

    /** The model's own name, as recorded in the file it was loaded from. */
    public String name() {
        return name;
    }

    /** Architecture family — "llama", "qwen3", "phi3", … Not a dispatch key for users. */
    public String architecture() {
        return architecture;
    }

    /** Context length this model was loaded with, in tokens; the ceiling for a session. */
    public int contextLength() {
        return contextLength;
    }

    /** Where it was loaded from, or {@code null} if it did not come from a file. */
    public Path source() {
        return source;
    }

    /**
     * The representation the weights are stored in, when the whole set shares one.
     *
     * <p>Empty when it does not — a quantized GGUF commonly mixes types, keeping the embeddings and
     * the output projection at a wider one than the attention weights. An empty answer is the
     * honest one there: picking a single value would be choosing which tensors to describe and not
     * saying so. Use {@link #weightTypes()} for the full picture.
     */
    public Optional<DataType> weightType() {
        return weightTypes.size() == 1
                ? Optional.of(weightTypes.iterator().next())
                : Optional.empty();
    }

    /** Every representation present in the weights — one element for a uniform model. */
    public Set<DataType> weightTypes() {
        return weightTypes;
    }

    /**
     * The representation activations are held in, which is what the kernels compute with. An FP16
     * model computes in FP16; everything quantized quantizes its activations to match its kernels.
     */
    public DataType computeType() {
        return computeType;
    }

    @Override
    public String toString() {
        String weights = weightType().map(Enum::name).orElseGet(() -> "mixed " + weightTypes);
        return name
                + " ("
                + architecture
                + ", context "
                + contextLength
                + ", weights "
                + weights
                + ", compute "
                + computeType
                + ")";
    }
}
