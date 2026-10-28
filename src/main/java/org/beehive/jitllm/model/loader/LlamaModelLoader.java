package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.LlamaPlanProvider;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanProvider;
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
import org.beehive.jitllm.runtime.backend.BackendId;
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
        final int nl = config.numberOfLayers();
        BackendId backend = TornadoDevices.current().backend();

        // The Q4_0 layers run a file whose per-layer weights are all Q4_0 (ffn_down may be Q4_1),
        // decided by those weights, not by the output projection: a Q4_0 file leaves token_embd -
        // the output projection too, when they are tied - as Q6_K. DataType.Q4_0 selects them.
        boolean q4_0Layers = LlamaPlanProvider.retainsQ4_0(sourceTypes(tensorEntries), nl, backend);
        // On CUDA and Metal every tensor stays as the file stores it; elsewhere Q4_0 is kept for
        // the Q4_0 layers only, and everything else is converted to Q8_0 as before.
        boolean retain = TornadoPlanProvider.retainsNative(backend);

        TornadoTensor[] wq =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_q.weight"));
        TornadoTensor[] wk =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_k.weight"));
        TornadoTensor[] wv =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_v.weight"));
        TornadoTensor[] wo =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_output.weight"));
        TornadoTensor[] w1 =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".ffn_gate.weight"));
        // ffn_down keeps its own type: llama.cpp's Q4_0 recipe writes some layers' down projection
        // as Q4_1, and the layer graph picks the kernel per layer.
        TornadoTensor[] w2 =
                retain || q4_0Layers
                        ? loadArrayOfTornadoTensorsNative(
                                nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"))
                        : loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"));
        TornadoTensor[] w3 =
                perLayer(
                        retain,
                        q4_0Layers,
                        nl,
                        i -> tensorEntries.get("blk." + i + ".ffn_up.weight"));

        DataType weightType;
        if (q4_0Layers) {
            weightType = DataType.Q4_0;
        } else if (retain) {
            weightType = q8_0LayersType(wq, wk, wv, wo, w1, w2, w3);
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
        LlamaTornadoWeights weights =
                new LlamaTornadoWeights(
                        retain
                                ? loadTornadoTensorNative(tokenEmbeddings)
                                : loadTornadoTensor(tokenEmbeddings),
                        loadArrayOfTornadoTensors(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // fp32
                        wq,
                        wk,
                        wv,
                        wo,
                        loadArrayOfTornadoTensors(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // fp32
                        w1,
                        w2,
                        w3,
                        loadTornadoTensor(tensorEntries.get("output_norm.weight")), // fp32
                        TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                        TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                        // Q4_0 files keep the output projection as Q6_K. Kept as it is, the logits
                        // graph reads it with the packed-integer Q6_K kernel instead of a Q8_0 copy
                        // that is a quarter larger and read in floating point.
                        retain || (q4_0Layers && outputWeight.ggmlType() == GGMLType.Q6_K)
                                ? loadTornadoTensorNative(outputWeight)
                                : loadTornadoTensor(outputWeight),
                        weightType);
        return weights;
    }

    // @formatter:on

    /**
     * A per-layer weight array: as the file stores it where the load keeps every tensor ({@code
     * retain}); otherwise Q4_0 kept for the Q4_0 layers and everything else converted to Q8_0.
     *
     * <p>All or nothing on purpose. A graph that mixed a retained Q4_0 tensor into a plan whose
     * kernels read Q8_0 blocks would be reading one block layout as another; a kept file instead
     * runs the layers that read each tensor by its type.
     */
    private static TornadoTensor[] perLayer(
            boolean retain,
            boolean q4_0Layers,
            int layers,
            java.util.function.IntFunction<GGMLTensorEntry> entry) {
        if (retain) {
            return loadArrayOfTornadoTensorsNative(layers, entry);
        }
        return q4_0Layers
                ? loadArrayOfTornadoTensorsRetainingQ4_0(layers, entry)
                : loadArrayOfTornadoTensors(layers, entry);
    }

    /**
     * What a kept file that does not run the Q4_0 layers reports: F16 or Q8_0 when its projections
     * all are, and otherwise the first projection type that is neither Q8_0 nor Q4_0, so that the
     * Q8_0 layers, which read each tensor by its type, are the ones that run it.
     *
     * @throws ModelLoadException for a file mixing only Q4_0 and Q8_0 projections, which no
     *     llama.cpp recipe writes and no Llama plan reads
     */
    private static DataType q8_0LayersType(TornadoTensor[]... projections) {
        boolean q4_0 = false;
        for (TornadoTensor[] layers : projections) {
            for (TornadoTensor w : layers) {
                DataType type = w.dataType();
                if (type == DataType.Q4_0) {
                    q4_0 = true;
                } else if (type != DataType.Q8_0) {
                    return type;
                }
            }
        }
        if (q4_0) {
            throw new ModelLoadException(
                    "this Llama file mixes Q4_0 and Q8_0 projections without the layout the Q4_0"
                            + " layers read; no Llama plan reads that mix");
        }
        return DataType.Q8_0;
    }
}
