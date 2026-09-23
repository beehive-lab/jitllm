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
import org.beehive.jllm.model.format.LlamaChatFormat;
import org.beehive.jllm.model.llama.Llama;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tokenizer.LlamaTokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.beehive.jllm.tokenizer.Vocabulary;

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
        boolean retainQ4_0 = RETAIN_Q4_0 && allQ4_0(tensorEntries, nl);
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
        return new LlamaTornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // fp32
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // fp32
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                perLayerQuantized(
                        retainQ4_0, nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")), // fp32
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                loadTornadoTensor(outputWeight),
                weightType);
    }

    // @formatter:on

    /**
     * Whether a Q4_0 file's weights are kept as they are, rather than materialized as Q8_0.
     *
     * <p>On by default — retaining is the point. It is switchable so the two can be measured
     * against each other on <b>one file</b>, which is the only comparison that isolates the
     * representation: comparing a Q4_0 model against a separately quantized Q8_0 one measures the
     * quantization as well. It is also the fallback if a device turns out to miscompile the Q4_0
     * kernels, the way the warp path did on OpenCL.
     */
    private static final boolean RETAIN_Q4_0 =
            !"false".equalsIgnoreCase(System.getProperty("jllm.q4_0.retain", "true"));

    /**
     * A per-layer weight array, retaining Q4_0 when the whole layer stack is Q4_0.
     *
     * <p>All or nothing on purpose. A graph that mixed a retained Q4_0 tensor into a plan whose
     * kernels read Q8_0 blocks would be reading one block layout as another — 18-byte blocks
     * addressed as 34-byte ones — which produces weights of plausible magnitude and fluent, wrong
     * text. There is no per-tensor dispatch in these layers, so the decision is made once for the
     * model.
     */
    private static org.beehive.jllm.backend.tornado.tensor.TornadoTensor[] perLayerQuantized(
            boolean retainQ4_0, int layers, java.util.function.IntFunction<GGMLTensorEntry> entry) {
        return retainQ4_0
                ? loadArrayOfTornadoTensorsRetainingQ4_0(layers, entry)
                : loadArrayOfTornadoTensors(layers, entry);
    }

    /**
     * Whether every per-layer weight the Q4_0 layer graph reads is Q4_0.
     *
     * <p>The norms are F32 and are read by dtype-independent kernels, so they are not consulted;
     * these seven are the ones a Q4_0 kernel would decode.
     */
    private static boolean allQ4_0(Map<String, GGMLTensorEntry> tensorEntries, int layers) {
        String[] kinds = {
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
        };
        for (int layer = 0; layer < layers; layer++) {
            for (String kind : kinds) {
                GGMLTensorEntry entry = tensorEntries.get("blk." + layer + "." + kind + ".weight");
                if (entry == null || entry.ggmlType() != GGMLType.Q4_0) {
                    return false;
                }
            }
        }
        return layers > 0;
    }
}
