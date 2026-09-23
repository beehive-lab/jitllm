package org.beehive.jllm.model.loader;

import static org.beehive.jllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jllm.auxiliary.Pair;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jllm.format.DataTypeMapping;
import org.beehive.jllm.format.GGMLTensorEntry;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.format.GGUF;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.devstral.Devstral;
import org.beehive.jllm.model.devstral.DevstralConfiguration;
import org.beehive.jllm.model.format.DevstralChatFormat;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tokenizer.DevstralTokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.beehive.jllm.tokenizer.Vocabulary;

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
        // Read from a per-layer weight rather than from output.weight, because a K-quant file is
        // mixed and the two disagree: Devstral's Q4_K_M stores its 240 blk.* tensors as Q4_K and
        // output.weight as Q6_K. The plan is built for the transformer layers, so the layers'
        // representation is the one it must be told about; the logits layer reads output.weight,
        // which is materialized as Q8_0 as before. Uniform F16 and Q8_0 files are unaffected —
        // there every tensor has the same type, so this reads exactly what it used to.
        GGMLTensorEntry perLayerWeight = tensorEntries.get("blk.0.attn_q.weight");
        // Only the per-layer weights are ever retained: the embedding and output tensors keep the
        // existing materialization, so the embedding and logits paths are untouched.
        final int nl = config.numberOfLayers();
        DataType weightType =
                DataTypeMapping.materializedType(
                        perLayerWeight != null
                                ? perLayerWeight.ggmlType()
                                : outputWeight.ggmlType(),
                        ExecutionTarget.GPU);
        // Q4_K is retained rather than materialized for this family:
        // it has Q4_K kernels, and materializing Q8_0 nearly doubles a 24B model's footprint.
        //
        // A "Q4_K_M" file mixes formats per tensor and per layer — Devstral's holds attn_v and
        // ffn_down as Q6_K in 20 of its 40 layers and Q4_K in the other 20 — so the layer graph
        // selects a kernel per tensor and both formats are retained. DataType.Q4_K is the marker
        // for "this model's per-layer weights are K-quants, kept as they are"; it selects the
        // K-quant plan, and each weight is then read by the kernel matching its own type.
        if (perLayerWeight != null && allKQuant(tensorEntries, nl)) {
            weightType = DataType.Q4_K;
        }

        if (weightType != DataType.F16
                && weightType != DataType.Q8_0
                && weightType != DataType.Q4_K) {
            throw new UnsupportedOperationException(
                    "Type: " + weightType + " currently not supported for TornadoVM weights.");
        }

        return new LlamaTornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensorsRetainingQ4_K(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                loadTornadoTensor(outputWeight),
                weightType);
    }

    /**
     * Whether every per-layer weight the K-quant layer graph reads is a format it has a kernel for.
     *
     * <p>Q4_K and Q6_K are retained and decoded in place; anything else is still materialized as
     * Q8_0, and a graph that mixed a materialized tensor into a retained plan would be reading one
     * block layout as another. The norms are F32 and are read by dtype-independent kernels, so they
     * are not consulted; these seven are the ones a K-quant kernel would decode.
     */
    private static boolean allKQuant(Map<String, GGMLTensorEntry> tensorEntries, int layers) {
        String[] kinds = {
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
        };
        for (int i = 0; i < layers; i++) {
            for (String kind : kinds) {
                GGMLTensorEntry entry = tensorEntries.get("blk." + i + "." + kind + ".weight");
                if (entry == null
                        || (entry.ggmlType() != GGMLType.Q4_K
                                && entry.ggmlType() != GGMLType.Q6_K)) {
                    return false;
                }
            }
        }
        return true;
    }
    // @formatter:on
}
