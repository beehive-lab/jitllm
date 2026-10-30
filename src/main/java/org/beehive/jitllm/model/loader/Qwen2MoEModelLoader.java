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
import org.beehive.jitllm.inference.weights.standard.Qwen2MoEStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Qwen3ChatFormat;
import org.beehive.jitllm.model.qwen2.Qwen2MoE;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

public class Qwen2MoEModelLoader extends AbstractModelLoader<Qwen2MoE, Qwen2MoEConfiguration> {
    public Qwen2MoEModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen =
                "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen2MoEConfiguration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLengthModel(),
                config.headSize(),
                config.ropeTheta(),
                false,
                8,
                1,
                3,
                8192);
    }

    @Override
    protected Qwen2MoE createModel(
            Qwen2MoEConfiguration config, Tokenizer tokenizer, Weights weights) {
        ChatFormat.ChatTokens chatTokens =
                new ChatFormat.ChatTokens(
                        "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen2MoE(
                config,
                tokenizer,
                weights,
                new Qwen3ChatFormat((Qwen3Tokenizer) tokenizer, chatTokens));
    }

    @Override
    protected Qwen2MoEConfiguration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen2moe.context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;
        int numberOfKeyValueHeads = (int) metadata.get("qwen2moe.attention.head_count_kv");
        int vocabSize = vocabulary.size();

        int moeHiddenDim = gguf.getTensorInfos().get("blk.0.ffn_down_exps.weight").dimensions()[0];

        return new Qwen2MoEConfiguration(
                getModelQuantization(metadata), // quantization
                (int) metadata.get("qwen2moe.embedding_length"), // dim
                0, // hiddenDim
                (int) metadata.get("qwen2moe.block_count"), // numberOfLayers
                (int) metadata.get("qwen2moe.attention.head_count"), // numberOfHeads
                numberOfKeyValueHeads, // numberOfKeyValueHeads
                numberOfKeyValueHeads, // numberOfHeadsKey
                numberOfKeyValueHeads, // numberOfHeadsValue
                vocabSize, // vocabularySize
                modelContextLength, // contextLengthModel
                finalContextLength, // contextLength
                (int) metadata.get("qwen2moe.expert_count"), // numberOfExperts
                (int) metadata.get("qwen2moe.expert_used_count"), // numberOfExpertsUsed
                moeHiddenDim, // moeHiddenDim
                (int) metadata.get("qwen2moe.feed_forward_length"), // sharedExpertHiddenDim
                false, // sharedWeights
                (float) metadata.get("qwen2moe.attention.layer_norm_rms_epsilon"), // rmsNormEps
                (float) metadata.get("qwen2moe.rope.freq_base") // ropeTheta
                );
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen2MoEConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new Qwen2MoEStandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                null, // w1: qwen2moe has no dense ffn_gate, only routed/shared experts below
                null, // w2: qwen2moe has no dense ffn_down
                null, // w3: qwen2moe has no dense ffn_up
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_exps.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up_exps.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_shexp.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up_shexp.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_down_shexp.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp_shexp.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen2MoEConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        // On CUDA and Metal every tensor stays as the file stores it: Qwen2-MoE's layers read each
        // one by its own type. Elsewhere Q4_0, Q4_1 and the K-quants are converted to Q8_0 as
        // before.
        boolean retain = TornadoPlanProvider.retainsNative(TornadoDevices.current().backend());
        TornadoTensor[] wq = perLayer(retain, nl, i -> layer(tensorEntries, i, "attn_q"));
        TornadoTensor[] wk = perLayer(retain, nl, i -> layer(tensorEntries, i, "attn_k"));
        TornadoTensor[] wv = perLayer(retain, nl, i -> layer(tensorEntries, i, "attn_v"));
        TornadoTensor[] wo = perLayer(retain, nl, i -> layer(tensorEntries, i, "attn_output"));
        TornadoTensor[] gateExperts =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_gate_exps"));
        TornadoTensor[] upExperts =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_up_exps"));
        TornadoTensor[] downExperts =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_down_exps"));
        TornadoTensor[] sharedGate =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_gate_shexp"));
        TornadoTensor[] sharedUp =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_up_shexp"));
        TornadoTensor[] sharedDown =
                perLayer(retain, nl, i -> layer(tensorEntries, i, "ffn_down_shexp"));
        DataType weightType;
        if (retain) {
            weightType =
                    keptType(
                            wq,
                            wk,
                            wv,
                            wo,
                            gateExperts,
                            upExperts,
                            downExperts,
                            sharedGate,
                            sharedUp,
                            sharedDown);
        } else {
            weightType =
                    DataTypeMapping.materializedType(outputWeight.ggmlType(), ExecutionTarget.GPU);
            if (weightType != DataType.F16 && weightType != DataType.Q8_0) {
                throw new UnsupportedOperationException(
                        "Type: " + weightType + " currently not supported for TornadoVM weights.");
            }
        }

        return new Qwen2MoETornadoWeights(
                load(retain, tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                wq,
                wk,
                wv,
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                wo,
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight")),
                gateExperts,
                upExperts,
                downExperts,
                sharedGate,
                sharedUp,
                sharedDown,
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp_shexp.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                load(retain, outputWeight),
                weightType);
    }

    private static GGMLTensorEntry layer(
            Map<String, GGMLTensorEntry> tensorEntries, int layer, String role) {
        return tensorEntries.get("blk." + layer + "." + role + ".weight");
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
}
