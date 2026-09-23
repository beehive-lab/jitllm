package org.beehive.jllm.model.loader;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jllm.auxiliary.Pair;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.format.DataTypeMapping;
import org.beehive.jllm.format.GGMLTensorEntry;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.format.GGUF;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.beehive.jllm.tokenizer.Vocabulary;

/**
 * Abstract base class for model loaders using Template Method pattern. Provides common loading flow
 * with extension points for model-specific logic.
 *
 * @param <M> The specific Model type to load
 * @param <C> The specific Configuration type for the model
 */
public abstract class AbstractModelLoader<M extends Model, C extends Configuration> {

    /**
     * Rule 16: loading is library code, so its progress goes through the platform logger and an
     * embedder can silence or route it. Reached only under {@code
     * jllm.EnableTimingForTornadoVMInit}.
     */
    private static final System.Logger LOGGER =
            System.getLogger(AbstractModelLoader.class.getName());

    protected final FileChannel fileChannel;
    protected final GGUF gguf;
    protected final int contextLength;
    protected final boolean useTornadovm;

    protected Vocabulary vocabulary;

    protected AbstractModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.useTornadovm = useTornadovm;
    }

    /**
     * The activation representation this model runs with, from GGUF's {@code general.file_type}.
     *
     * <p>Not the same question as the weights' type: every quantization the GPU cannot execute
     * directly is materialized as Q8_0 (see {@link #effectiveGpuWeightType}), so they all report
     * Q8_0 activations. Static because it reads only its argument — which also makes it testable
     * without standing up a loader.
     */
    protected static String getModelQuantization(Map<String, Object> metadata) {
        int modelQuantizationAsInt = (int) metadata.get("general.file_type");
        return switch (modelQuantizationAsInt) {
            case 1 -> "FP16";
            case 2 ->
                    "Q8_0"; // Q4_0 — decoded directly on the CPU, materialized as Q8_0 for the GPU
            case 32 -> "FP16"; // MOSTLY_BF16 — BF16 weights use FP16 activation buffers
            case 7 -> "Q8_0";
            case 14, 15 -> "Q8_0"; // Q4_K_S, Q4_K_M (K-quants use Q8_0 activations)
            case 16, 17 -> "Q8_0"; // Q5_K_S, Q5_K_M
            case 18 -> "Q8_0"; // Q6_K
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: "
                                    + modelQuantizationAsInt
                                    + " (as int).");
        };
    }

    /**
     * Returns the effective GPU weight type for TornadoVM execution. K-quant types (Q4_K, Q5_K,
     * Q6_K) are dequantized to Q8_0 at load time.
     */
    /**
     * @deprecated Use {@code DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU)}. Kept
     *     while the diagnostic below still speaks the file's vocabulary; it answers in file types,
     *     which is what made the collapse invisible in the first place.
     */
    @Deprecated
    protected static GGMLType effectiveGpuWeightType(GGMLType ggmlType) {
        // Delegates to the mapping so the collapse is stated once and tested. A type this
        // engine does not execute at all is left alone here rather than rejected: the callers
        // already refuse anything that is not F16 or Q8_0, with a message naming the type.
        // BF16 collapses to F16 for the GPU, which is what #120's explicit switch did here.
        if (!DataTypeMapping.isSupported(ggmlType, ExecutionTarget.GPU)) {
            return ggmlType;
        }
        return DataTypeMapping.asFileType(
                DataTypeMapping.materializedType(ggmlType, ExecutionTarget.GPU));
    }

    private static String fileTypeName(int fileType) {
        return switch (fileType) {
            case 0 -> "F32";
            case 1 -> "F16";
            case 2 -> "Q4_0";
            case 32 -> "BF16";
            case 7 -> "Q8_0";
            case 14 -> "Q4_K_S";
            case 15 -> "Q4_K_M";
            case 16 -> "Q5_K_S";
            case 17 -> "Q5_K_M";
            case 18 -> "Q6_K";
            default -> "type_" + fileType;
        };
    }

    /**
     * Template method that defines the model loading workflow. Subclasses should not override this
     * method.
     *
     * @return The loaded model instance
     */
    public final M loadModel() {
        try {
            Map<String, Object> metadata = gguf.getMetadata();

            // Step 1: Load vocabulary
            this.vocabulary = loadVocabulary(metadata);

            // Step 2: Create tokenizer
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            // Step 3: Create configuration
            C config = createConfiguration(metadata);

            // Step 4: Load tensor entries
            Map<String, GGMLTensorEntry> tensorEntries;
            if (useTornadovm) {
                tensorEntries =
                        GGUF.loadTensorsTornado(
                                fileChannel,
                                gguf.getTensorDataOffset(),
                                gguf.getTensorInfos(),
                                org.beehive.jllm.backend.tornado.device.TornadoDevices.current());
            } else {
                tensorEntries =
                        GGUF.loadTensorsStandard(
                                fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
            }

            // Step 4: Load weights
            Weights weights = loadWeights(tensorEntries, config);

            // Step 5: Create and return model instance
            return createModel(config, tokenizer, weights);

        } catch (IOException e) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix() + "Failed to load model", e);
        }
    }

    /**
     * Load the vocabulary from GGUF metadata. Model-specific implementations should override this
     * method.
     *
     * @param metadata The GGUF metadata map
     * @return The loaded Vocabulary
     */
    protected abstract Vocabulary loadVocabulary(Map<String, Object> metadata);

    /**
     * Create a tokenizer instance for this model.
     *
     * @param metadata The GGUF metadata map
     * @param vocabulary The loaded vocabulary
     * @return The tokenizer instance
     */
    protected abstract Tokenizer createTokenizer(
            Map<String, Object> metadata, Vocabulary vocabulary);

    /**
     * Create a configuration instance from GGUF metadata.
     *
     * @param metadata The GGUF metadata map
     * @return The configuration instance
     */
    protected abstract C createConfiguration(Map<String, Object> metadata);

    /**
     * Load model weights from tensor entries. Default implementation handles common weight loading
     * logic.
     *
     * @param tensorEntries Map of tensor names to tensor entries
     * @param config The model configuration
     * @return The loaded weights
     */
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, C config) {
        // Precompute RoPE frequencies
        Pair<float[], float[]> ropeFreqs = precomputeRopeFrequencies(config);

        // Get token embeddings and output weights
        GGMLTensorEntry tokenEmbeddings = getTokenEmbeddings(tensorEntries);
        GGMLTensorEntry outputWeight = getOutputWeight(tensorEntries, tokenEmbeddings);

        // Delegate to specific implementation
        if (useTornadovm) {
            Weights weights =
                    createTornadoVMWeights(
                            tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            // Reported after loading, from the weights themselves. It used to be a prediction made
            // before them, from the output tensor's type — which said "Q4_0 -> Q8_0" for a family
            // that materializes nothing, and would have to guess again for every tensor of a mixed
            // model. The representation named here is the model's own; tensors whose role differs
            // keep theirs, and the memory plan reports those per tensor.
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                int fileType = (int) gguf.getMetadata().get("general.file_type");
                LOGGER.log(
                        System.Logger.Level.INFO,
                        "Loaded model weights in TornadoVM format ("
                                + fileTypeName(fileType)
                                + " -> "
                                + weights.dataType()
                                + ")");
            }
            return weights;
        } else {
            return createStandardWeights(
                    tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    /**
     * Create the final model instance.
     *
     * @param config The model configuration
     * @param tokenizer The tokenizer
     * @param weights The loaded weights
     * @return The model instance
     */
    protected abstract M createModel(C config, Tokenizer tokenizer, Weights weights);

    /**
     * Precompute RoPE frequencies for this model. Default implementation can be overridden for
     * custom RoPE configurations.
     */
    protected abstract Pair<float[], float[]> precomputeRopeFrequencies(C config);

    /**
     * Get token embeddings tensor entry. Default implementation can be overridden for different
     * tensor naming.
     */
    protected GGMLTensorEntry getTokenEmbeddings(Map<String, GGMLTensorEntry> tensorEntries) {
        return tensorEntries.get("token_embd.weight");
    }

    /**
     * Get output weight tensor entry. Default implementation falls back to token embeddings if
     * output.weight not found.
     */
    protected GGMLTensorEntry getOutputWeight(
            Map<String, GGMLTensorEntry> tensorEntries, GGMLTensorEntry tokenEmbeddings) {
        return tensorEntries.getOrDefault("output.weight", tokenEmbeddings);
    }

    /** Create standard (CPU) weights. */
    protected abstract Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            C config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight);

    /** Create TornadoVM (GPU) weights. */
    protected abstract Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            C config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight);
}
