package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.function.IntFunction;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanProvider;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.format.MistralChatFormat;
import org.beehive.jitllm.model.mistral.Mistral;
import org.beehive.jitllm.model.mistral.MistralConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tokenizer.MistralTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

public class MistralModelLoader extends AbstractModelLoader<Mistral, MistralConfiguration> {

    public MistralModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new MistralTokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected MistralConfiguration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("llama.context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;

        // Get vocabulary size from metadata
        int vocabSize =
                metadata.containsKey("llama.vocab_size")
                        ? (int) metadata.get("llama.vocab_size")
                        : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new MistralConfiguration(
                getModelQuantization(metadata),
                (int) metadata.get("llama.embedding_length"),
                (int) metadata.get("llama.feed_forward_length"),
                (int) metadata.get("llama.block_count"),
                (int) metadata.get("llama.attention.head_count"),
                metadata.containsKey("llama.attention.head_count_kv")
                        ? (int) metadata.get("llama.attention.head_count_kv")
                        : (int) metadata.get("llama.attention.head_count"),
                vocabSize,
                finalContextLength,
                false,
                (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault("llama.rope.freq_base", 10000f));
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(MistralConfiguration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLength(),
                config.dim() / config.numberOfHeads(),
                config.ropeTheta(),
                false,
                1.0f,
                1.0f,
                1.0f,
                config.contextLength());
    }

    // @formatter:on

    @Override
    protected Mistral createModel(
            MistralConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Mistral(
                config, tokenizer, weights, new MistralChatFormat((MistralTokenizer) tokenizer));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            MistralConfiguration config,
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

    // @formatter:off

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            MistralConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        // On CUDA and Metal every tensor stays as the file stores it: Mistral's layers read each
        // one
        // by its own type. Elsewhere Q4_0, Q4_1 and the K-quants are converted to Q8_0 as before.
        boolean retain = TornadoPlanProvider.retainsNative(TornadoDevices.current().backend());
        TornadoTensor[] wq =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight"));
        TornadoTensor[] wk =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight"));
        TornadoTensor[] wv =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight"));
        TornadoTensor[] wo =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight"));
        TornadoTensor[] w1 =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight"));
        TornadoTensor[] w2 =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"));
        TornadoTensor[] w3 =
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight"));
        DataType weightType;
        if (retain) {
            weightType = keptType(wq, wk, wv, wo, w1, w2, w3);
        } else {
            weightType =
                    DataTypeMapping.materializedType(outputWeight.ggmlType(), ExecutionTarget.GPU);
            // Validate supported types
            if (weightType != DataType.F16 && weightType != DataType.Q8_0) {
                throw new UnsupportedOperationException(
                        "Type: " + weightType + " currently not supported for TornadoVM weights.");
            }
        }

        // Load all tensors uniformly as TornadoTensor hierarchy
        return new LlamaTornadoWeights(
                load(retain, tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // fp32
                wq,
                wk,
                wv,
                wo,
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // fp32
                w1,
                w2,
                w3,
                loadTornadoTensor(tensorEntries.get("output_norm.weight")), // fp32
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                load(retain, outputWeight),
                weightType);
    }

    /** A device tensor as the file stores it, or with Q4_0, Q4_1 and K-quants converted to Q8_0. */
    private static TornadoTensor load(boolean retain, GGMLTensorEntry entry) {
        return retain ? loadTornadoTensorNative(entry) : loadTornadoTensor(entry);
    }

    /** {@link #load} for each layer. */
    private static TornadoTensor[] perLayer(
            boolean retain, int layers, IntFunction<GGMLTensorEntry> entry) {
        return retain
                ? loadArrayOfTornadoTensorsNative(layers, entry)
                : loadArrayOfTornadoTensors(layers, entry);
    }
    // @formatter:on
}
