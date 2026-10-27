package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.LlamaPlanProvider;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.LlamaStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.format.LlamaChatFormat;
import org.beehive.jitllm.model.llama.Llama;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tokenizer.LlamaTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

public class LlamaModelLoader extends AbstractModelLoader<Llama, LlamaConfiguration> {

    public LlamaModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokens(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new LlamaTokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected LlamaConfiguration createConfiguration(Map<String, Object> metadata) {
        int vocabSize =
                metadata.containsKey("llama.vocab_size")
                        ? (int) metadata.get("llama.vocab_size")
                        : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new LlamaConfiguration(
                        getModelQuantization(metadata),
                        (int) metadata.get("llama.embedding_length"),
                        (int) metadata.get("llama.feed_forward_length"),
                        (int) metadata.get("llama.block_count"),
                        (int) metadata.get("llama.attention.head_count"),
                        metadata.containsKey("llama.attention.head_count_kv")
                                ? (int) metadata.get("llama.attention.head_count_kv")
                                : (int) metadata.get("llama.attention.head_count"),
                        vocabSize,
                        (int) metadata.get("llama.context_length"),
                        (float)
                                metadata.getOrDefault(
                                        "llama.attention.layer_norm_rms_epsilon", 1e-5f),
                        (float) metadata.getOrDefault("llama.rope.freq_base", 10000f))
                .withContextLength(contextLength);
    }

    // @formatter:on

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(LlamaConfiguration config) {
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

    @Override
    protected Llama createModel(LlamaConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Llama(
                config, tokenizer, weights, new LlamaChatFormat((LlamaTokenizer) tokenizer));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            LlamaConfiguration config,
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
            LlamaConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        DataType weightType =
                DataTypeMapping.materializedType(outputWeight.ggmlType(), ExecutionTarget.GPU);

        final int nl = config.numberOfLayers();

        // A Q4_0 file's per-layer weights are kept as they are rather than materialized as Q8_0,
        // which would take 4.5 bits per weight to 8.5 and roughly double what the model occupies on
        // the device. DataType.Q4_0 is the marker for "this model's per-layer weights are Q4_0,
        // kept as they are"; it selects the Q4_0 plan.
        //
        // The type is decided by the per-layer weights, not by the output projection, because those
        // are what the layer graph reads. A Q4_0 file leaves token_embd — the output projection
        // too, when they are tied — as Q6_K, so the output weight alone would say Q8_0 and select
        // the wrong plan.
        boolean retainQ4_0 =
                LlamaPlanProvider.retainsQ4_0(
                        sourceTypes(tensorEntries), nl, TornadoDevices.current().backend());
        if (retainQ4_0) {
            weightType = DataType.Q4_0;
        }

        // Validate supported types
        if (weightType != DataType.F16
                && weightType != DataType.Q8_0
                && weightType != DataType.Q4_0) {
            throw new UnsupportedOperationException(
                    "Type: " + weightType + " currently not supported for TornadoVM weights.");
        }

        // Load all tensors uniformly as TornadoTensor hierarchy
        LlamaTornadoWeights weights =
                new LlamaTornadoWeights(
                        loadTornadoTensor(tokenEmbeddings),
                        loadArrayOfTornadoTensors(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // fp32
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                        loadArrayOfTornadoTensors(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // fp32
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                        // ffn_down keeps its own type: llama.cpp's Q4_0 recipe writes some layers'
                        // down
                        // projection as Q4_1, and the layer graph picks the kernel per layer.
                        retainQ4_0
                                ? loadArrayOfTornadoTensorsNative(
                                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"))
                                : loadArrayOfTornadoTensors(
                                        nl,
                                        i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                        perLayerQuantized(
                                retainQ4_0,
                                nl,
                                i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                        loadTornadoTensor(tensorEntries.get("output_norm.weight")), // fp32
                        TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                        TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                        // Q4_0 files keep the output projection as Q6_K. Kept as it is, the logits
                        // graph
                        // reads it with the packed-integer Q6_K kernel instead of a Q8_0 copy that
                        // is a
                        // quarter larger and read in floating point.
                        retainQ4_0 && outputWeight.ggmlType() == GGMLType.Q6_K
                                ? loadTornadoTensorNative(outputWeight)
                                : loadTornadoTensor(outputWeight),
                        weightType);
        return weights;
    }

    // @formatter:on

    /**
     * A per-layer weight array, retaining Q4_0 when the whole layer stack is Q4_0.
     *
     * <p>All or nothing on purpose. A graph that mixed a retained Q4_0 tensor into a plan whose
     * kernels read Q8_0 blocks would be reading one block layout as another — 18-byte blocks
     * addressed as 34-byte ones — which produces weights of plausible magnitude and fluent, wrong
     * text. There is no per-tensor dispatch in these layers, so the decision is made once for the
     * model.
     */
    private static TornadoTensor[] perLayerQuantized(
            boolean retainQ4_0, int layers, java.util.function.IntFunction<GGMLTensorEntry> entry) {
        return retainQ4_0
                ? loadArrayOfTornadoTensorsRetainingQ4_0(layers, entry)
                : loadArrayOfTornadoTensors(layers, entry);
    }
}
