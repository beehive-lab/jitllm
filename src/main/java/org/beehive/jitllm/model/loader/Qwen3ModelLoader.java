package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;
import static org.beehive.jitllm.tokenizer.Vocabulary.fromTokensAndScores;

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
import org.beehive.jitllm.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.format.ChatFormat.ChatTokens;
import org.beehive.jitllm.model.format.Qwen3ChatFormat;
import org.beehive.jitllm.model.qwen3.Qwen3;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

public class Qwen3ModelLoader extends AbstractModelLoader<Qwen3, Qwen3Configuration> {

    public Qwen3ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen =
                "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    // @formatter:off
    @Override
    protected Qwen3Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen3.context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;

        int vocabSize = vocabulary.size();

        return new Qwen3Configuration(
                getModelQuantization(metadata),
                (int) metadata.get("qwen3.embedding_length"),
                (int) metadata.get("qwen3.feed_forward_length"),
                (int) metadata.get("qwen3.block_count"),
                (int) metadata.get("qwen3.attention.head_count"),
                metadata.containsKey("qwen3.attention.head_count_kv")
                        ? (int) metadata.get("qwen3.attention.head_count_kv")
                        : (int) metadata.get("qwen3.attention.head_count"),
                (int) metadata.get("qwen3.attention.key_length"),
                (int) metadata.get("qwen3.attention.value_length"),
                vocabSize,
                modelContextLength,
                finalContextLength,
                false,
                (float) metadata.get("qwen3.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen3.rope.freq_base"));
    }

    // @formatter:on

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen3Configuration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLengthModel(),
                config.numberOfHeadsKey(),
                config.ropeTheta(),
                false,
                0,
                0,
                0,
                0);
    }

    // @formatter:off
    @Override
    protected Qwen3 createModel(Qwen3Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen =
                "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens =
                isDeepSeekR1DistillQwen
                        ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                        : new ChatTokens(
                                "<|im_start|>",
                                "<|im_end|>",
                                "",
                                "<|end_of_text|>",
                                "<|endoftext|>");
        return new Qwen3(
                config,
                tokenizer,
                weights,
                new Qwen3ChatFormat((Qwen3Tokenizer) tokenizer, chatTokens));
    }

    // @formatter:off

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen3Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        final int nl = config.numberOfLayers();

        return new Qwen3StandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_norm.weight")), // rms_att_weight
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")), // wq
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")), // wk
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")), // wv
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")), // wo
                loadArrayOfTensors(
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")), // attnKNorm
                loadArrayOfTensors(
                        nl,
                        i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")), // attnQNorm
                loadArrayOfTensors(
                        nl,
                        i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")), // rms_ffn_weight
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                loadTensor(tensorEntries.get("output_norm.weight")), // rms_final_weight
                new ArrayFloatTensor(ropeFreqsReal),
                new ArrayFloatTensor(ropeFreqsImag),
                tensorEntries.containsKey("output.weight")
                        ? ModelLoader.loadTensor(tensorEntries.get("output.weight"))
                        : loadTensor(tokenEmbeddings), // weights are shared
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen3Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        // On CUDA and Metal every tensor stays as the file stores it: Qwen3's layers read each one
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
        // A converted file reports what its output projection becomes; a kept one, its
        // projections' type: F16 or Q8_0 as before, Q4_K for a Q4_K_M file. A kept file that mixes
        // Q8_0 with other types reports the first other type, so nothing that serves only uniform
        // Q8_0 files (MLX) takes it.
        DataType weightType =
                retain
                        ? keptType(wq, wk, wv, wo, w1, w2, w3)
                        : DataTypeMapping.materializedType(
                                outputWeight.ggmlType(), ExecutionTarget.GPU);

        return new Qwen3TornadoWeights(
                load(retain, tokenEmbeddings),
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                wq,
                wk,
                wv,
                wo,
                // Qwen3-specific: attnKNorm and attnQNorm (always F32)
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                perLayer(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                w1,
                w2,
                w3,
                load(retain, tensorEntries.get("output_norm.weight")), // fp32
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
