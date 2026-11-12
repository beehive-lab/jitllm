package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.plan.DevstralPlanProvider;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.Devstral;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.model.format.DevstralChatFormat;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tokenizer.DevstralTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

public class DevstralModelLoader extends AbstractModelLoader<Devstral, DevstralConfiguration> {

    public DevstralModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokens(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new DevstralTokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected DevstralConfiguration createConfiguration(Map<String, Object> metadata) {
        String prefix = "mistral3";

        int modelContextLength = (int) metadata.get(prefix + ".context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;

        int vocabSize =
                metadata.containsKey(prefix + ".vocab_size")
                        ? (int) metadata.get(prefix + ".vocab_size")
                        : (int) metadata.get("tokenizer.ggml.tokens.length");

        // Devstral 2 has independent head dimension (head_dim != dim/num_heads)
        int headDim = (int) metadata.get(prefix + ".attention.key_length");

        return new DevstralConfiguration(
                getModelQuantization(metadata),
                (int) metadata.get(prefix + ".embedding_length"),
                (int) metadata.get(prefix + ".feed_forward_length"),
                (int) metadata.get(prefix + ".block_count"),
                (int) metadata.get(prefix + ".attention.head_count"),
                metadata.containsKey(prefix + ".attention.head_count_kv")
                        ? (int) metadata.get(prefix + ".attention.head_count_kv")
                        : (int) metadata.get(prefix + ".attention.head_count"),
                headDim,
                vocabSize,
                finalContextLength,
                (float) metadata.getOrDefault(prefix + ".attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault(prefix + ".rope.freq_base", 10000f));
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(DevstralConfiguration config) {
        Map<String, Object> metadata = gguf.getMetadata();
        String prefix = "mistral3";

        String ropeScalingType = (String) metadata.getOrDefault(prefix + ".rope.scaling.type", "");
        if ("yarn".equals(ropeScalingType)) {
            float factor = (float) metadata.get(prefix + ".rope.scaling.factor");
            float betaFast = (float) metadata.get(prefix + ".rope.scaling.yarn_beta_fast");
            float betaSlow = (float) metadata.get(prefix + ".rope.scaling.yarn_beta_slow");
            float logMultiplier =
                    (float)
                            metadata.getOrDefault(
                                    prefix + ".rope.scaling.yarn_log_multiplier", 0.0f);
            int originalContextLength =
                    (int) metadata.get(prefix + ".rope.scaling.original_context_length");

            return RopeFrequencies.precomputeFreqsCisYaRN(
                    config.contextLength(),
                    config.headDim(),
                    config.ropeTheta(),
                    factor,
                    betaFast,
                    betaSlow,
                    logMultiplier,
                    originalContextLength);
        }

        return RopeFrequencies.precomputeFreqsCis(
                config.contextLength(),
                config.headDim(),
                config.ropeTheta(),
                false,
                1.0f,
                1.0f,
                1.0f,
                config.contextLength());
    }

    // @formatter:on

    @Override
    protected Devstral createModel(
            DevstralConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Devstral(
                config, tokenizer, weights, new DevstralChatFormat((DevstralTokenizer) tokenizer));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DevstralConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new LlamaStandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DevstralConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        // A file whose projections are all Q4_K and Q6_K, as a Q4_K_M file's are, runs the K-quant
        // layers; DataType.Q4_K selects them. Any other quantized file runs the Q8_0 layers, which
        // read each tensor by its type. Every tensor stays as the file stores it.
        boolean kQuantLayers = DevstralPlanProvider.kQuantLayers(sourceTypes(tensorEntries), nl);
        TornadoTensor[] wq =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight"));
        TornadoTensor[] wk =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight"));
        TornadoTensor[] wv =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight"));
        TornadoTensor[] wo =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight"));
        TornadoTensor[] w1 =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight"));
        TornadoTensor[] w2 =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"));
        TornadoTensor[] w3 =
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight"));
        DataType weightType =
                kQuantLayers ? DataType.Q4_K : q8_0LayersType(wq, wk, wv, wo, w1, w2, w3);

        return new LlamaTornadoWeights(
                loadTornadoTensorNative(tokenEmbeddings),
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                wq,
                wk,
                wv,
                wo,
                loadArrayOfTornadoTensorsNative(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                w1,
                w2,
                w3,
                loadTornadoTensorNative(tensorEntries.get("output_norm.weight")),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                loadTornadoTensorNative(outputWeight),
                weightType);
    }

    /**
     * What a file that does not run the K-quant layers reports: F16 or Q8_0 when its projections
     * all are, and otherwise the first projection type that is neither Q8_0 nor Q4_K, so that the
     * Q8_0 layers, which read each tensor by its type, are the ones that run it.
     *
     * @throws ModelLoadException for a file mixing only Q4_K and Q8_0 projections, which no
     *     llama.cpp recipe writes and no Devstral plan reads
     */
    private static DataType q8_0LayersType(TornadoTensor[]... projections) {
        boolean q4_k = false;
        for (TornadoTensor[] layers : projections) {
            for (TornadoTensor w : layers) {
                DataType type = w.dataType();
                if (type == DataType.Q4_K) {
                    q4_k = true;
                } else if (type != DataType.Q8_0) {
                    return type;
                }
            }
        }
        if (q4_k) {
            throw new ModelLoadException(
                    "this Devstral file mixes Q4_K and Q8_0 projections without the layout the"
                            + " K-quant layers read; no Devstral plan reads that mix");
        }
        return DataType.Q8_0;
    }
    // @formatter:on
}
