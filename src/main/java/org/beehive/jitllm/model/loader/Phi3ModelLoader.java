package org.beehive.jllm.model.loader;

import static org.beehive.jllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jllm.auxiliary.Pair;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jllm.format.DataTypeMapping;
import org.beehive.jllm.format.GGMLTensorEntry;
import org.beehive.jllm.format.GGUF;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.standard.Phi3StandardWeights;
import org.beehive.jllm.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.format.Phi3ChatFormat;
import org.beehive.jllm.model.phi3.Phi3;
import org.beehive.jllm.model.phi3.Phi3Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tokenizer.Phi3Tokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.beehive.jllm.tokenizer.Vocabulary;

public class Phi3ModelLoader extends AbstractModelLoader<Phi3, Phi3Configuration> {

    /**
     * Rule 16: loading is library code, so its progress goes through the platform logger and an
     * embedder can silence or route it. Reached only under {@code
     * jllm.EnableTimingForTornadoVMInit}.
     */
    private static final System.Logger LOGGER = System.getLogger(Phi3ModelLoader.class.getName());

    private int modelContextLength;

    public Phi3ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            Tokenizer tokenizer = new Phi3Tokenizer(metadata, vocabulary);
            LOGGER.log(
                    System.Logger.Level.INFO, "Tokenizer: " + tokenizer.getClass().getSimpleName());
            return tokenizer;
        }
        return new Phi3Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Phi3Configuration createConfiguration(Map<String, Object> metadata) {
        final String modelPrefix = "phi3.";

        // Needed by precomputeRopeFrequencies(). Left unassigned it stayed 0, so the CPU path's
        // freq_cis tables were empty arrays and the first token threw
        // ArrayIndexOutOfBoundsException. The GPU path did not notice: its Phi3 RoPE kernel
        // computes the frequencies inline instead of reading these tables.
        this.modelContextLength = (int) metadata.get(modelPrefix + "context_length");

        var config =
                new Phi3Configuration(
                        getModelQuantization(metadata),
                        (int) metadata.get(modelPrefix + "embedding_length"), // dim
                        (int) metadata.get(modelPrefix + "feed_forward_length"), // hidden_dim
                        (int) metadata.get(modelPrefix + "block_count"), // n_layers
                        (int) metadata.get(modelPrefix + "attention.head_count"), // n_heads
                        metadata.containsKey(modelPrefix + "attention.head_count_kv")
                                ? (int) metadata.get(modelPrefix + "attention.head_count_kv")
                                : (int)
                                        metadata.get(
                                                modelPrefix + "attention.head_count"), // n_kv_heads
                        vocabulary.size(), // vocab_size
                        contextLength, // context_length (user-specified, not model)
                        (float)
                                metadata.getOrDefault(
                                        modelPrefix + "attention.layer_norm_rms_epsilon",
                                        1e-5f), // rms_norm_eps
                        (float)
                                metadata.getOrDefault(
                                        modelPrefix + "rope.freq_base", 10000f) // rope_theta
                        );
        return config;
    }

    // @formatter:off

    // @formatter:off
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Phi3Configuration config) {
        // Calculate head size from dim and numberOfHeads
        int headSize = config.dim() / config.numberOfHeads();

        return RopeFrequencies.precomputeFreqsCis(
                modelContextLength, // Use model context length for RoPE precomputation
                headSize, // Calculated head size
                config.ropeTheta(),
                false, // Phi3 uses standard RoPE, not neox-style based on reference
                8,
                1,
                3,
                8192 // Additional RoPE parameters from reference
                );
    }

    // @formatter:off

    @Override
    protected Phi3 createModel(Phi3Configuration config, Tokenizer tokenizer, Weights weights) {
        // Phi3 chat tokens
        ChatFormat.ChatTokens chatTokens =
                new ChatFormat.ChatTokens(
                        "<|system|>", "<|end|>", "<|user|>", "<|end|>", "<|assistant|>");

        return new Phi3(
                config,
                tokenizer,
                weights,
                new Phi3ChatFormat((Phi3Tokenizer) tokenizer, chatTokens));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Phi3Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        final int nl = config.numberOfLayers();

        return new Phi3StandardWeights(
                loadTensor(tokenEmbeddings), // token_embedding_table
                loadArrayOfTensors(
                        nl,
                        i ->
                                tensorEntries.get(
                                        "blk." + i + ".attn_norm.weight")), // rms_att_weight (as
                // FloatTensor[])
                loadArrayOfTensors(
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")), // wqkv (combined)
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")), // wo
                loadArrayOfTensors(
                        nl,
                        i ->
                                tensorEntries.get(
                                        "blk." + i + ".ffn_norm.weight")), // rms_ffn_weight (as
                // FloatTensor[])
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // wDown
                loadArrayOfTensors(
                        nl,
                        i ->
                                tensorEntries.get(
                                        "blk." + i + ".ffn_up.weight")), // wUp (separate, not
                // combined)
                loadTensor(
                        tensorEntries.get(
                                "output_norm.weight")), // rms_final_weight (as FloatTensor)
                new ArrayFloatTensor(ropeFreqsReal), // freq_cis_real
                new ArrayFloatTensor(ropeFreqsImag), // freq_cis_imag
                loadTensor(outputWeight), // wcls
                DataTypeMapping.sourceType(outputWeight.ggmlType()) // weightType
                );
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Phi3Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        DataType weightType =
                DataTypeMapping.materializedType(outputWeight.ggmlType(), ExecutionTarget.GPU);

        // Validate supported types
        if (weightType != DataType.F16 && weightType != DataType.Q8_0) {
            throw new UnsupportedOperationException(
                    "Type: " + weightType + " currently not supported for TornadoVM weights.");
        }

        final int nl = config.numberOfLayers();

        // Load all tensors uniformly as TornadoTensor hierarchy
        return new Phi3TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // fp32
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // fp32
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")), // fp32
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                loadTornadoTensor(outputWeight),
                weightType);
    }
    // @formatter:on
}
