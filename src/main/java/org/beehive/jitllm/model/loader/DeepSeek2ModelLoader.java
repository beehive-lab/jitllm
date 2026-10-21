package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.tokenizer.Vocabulary.fromTokens;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.DeepSeek2StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek2TornadoWeights;
import org.beehive.jitllm.model.deepseek2.DeepSeek2;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.model.format.Glm4ChatFormat;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q8_0FloatTensor;
import org.beehive.jitllm.tokenizer.Glm4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loads a {@code deepseek2} GGUF with latent attention and a mixture of experts (GLM-4.7-Flash).
 *
 * <p>Every tensor keeps the representation the file gave it, except Q5_0 — which some quantizers
 * use for the shared experts — rewritten as Q8_0 by {@link Q5_0Widening}, exactly.
 */
public class DeepSeek2ModelLoader extends AbstractModelLoader<DeepSeek2, DeepSeek2Configuration> {

    /** As for qwen35: the declared 202752 positions would not fit; this is llama.cpp's order. */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    /** What llama.cpp recognizes as GLM-4.7-Flash when the file names no gating function. */
    private static final int GLM_VOCABULARY = 154880;

    public DeepSeek2ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokens(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Glm4Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected DeepSeek2Configuration createConfiguration(Map<String, Object> metadata) {
        String p = "deepseek2.";
        int modelContextLength = (int) metadata.get(p + "context_length");
        int finalContextLength =
                contextLength > 0
                        ? Math.min(contextLength, modelContextLength)
                        : Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
        int layers = (int) metadata.get(p + "block_count");
        int nextn =
                metadata.containsKey(p + "nextn_predict_layers")
                        ? (int) metadata.get(p + "nextn_predict_layers")
                        : 0;
        int heads = (int) metadata.get(p + "attention.head_count");
        int ropeDim = (int) metadata.get(p + "rope.dimension_count");
        int keyMla = (int) metadata.get(p + "attention.key_length_mla");
        int valueMla = (int) metadata.get(p + "attention.value_length_mla");
        int vocabulary = this.vocabulary.size();

        // llama.cpp's rule for a file that names no gating function: GLM-4.7-Flash uses sigmoid
        // scores, every older DeepSeek V2 file a softmax.
        boolean sigmoid;
        if (metadata.containsKey(p + "expert_gating_func")) {
            sigmoid = (int) metadata.get(p + "expert_gating_func") == 2;
        } else {
            int trunk = layers - nextn;
            sigmoid = (trunk == 47 || trunk == 48) && vocabulary == GLM_VOCABULARY;
        }
        if (metadata.containsKey(p + "rope.scaling.type")
                || metadata.containsKey(p + "rope.scaling.factor")) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "deepseek2 with RoPE scaling (YaRN) is not supported");
        }

        return new DeepSeek2Configuration(
                getModelQuantization(metadata),
                (int) metadata.get(p + "embedding_length"),
                (int) metadata.get(p + "feed_forward_length"),
                layers - nextn,
                metadata.containsKey(p + "leading_dense_block_count")
                        ? (int) metadata.get(p + "leading_dense_block_count")
                        : 0,
                heads,
                metadata.containsKey(p + "attention.q_lora_rank")
                        ? (int) metadata.get(p + "attention.q_lora_rank")
                        : 0,
                (int) metadata.get(p + "attention.kv_lora_rank"),
                keyMla - ropeDim,
                ropeDim,
                valueMla,
                (int) metadata.get(p + "expert_count"),
                (int) metadata.get(p + "expert_used_count"),
                (int) metadata.get(p + "expert_feed_forward_length"),
                (int) metadata.get(p + "expert_shared_count"),
                metadata.containsKey(p + "expert_weights_scale")
                        ? (float) metadata.get(p + "expert_weights_scale")
                        : 1f,
                metadata.containsKey(p + "expert_weights_norm")
                        && (boolean) metadata.get(p + "expert_weights_norm"),
                sigmoid,
                vocabulary,
                modelContextLength,
                finalContextLength,
                (float) metadata.get(p + "attention.layer_norm_rms_epsilon"),
                (float) metadata.get(p + "rope.freq_base"));
    }

    // @formatter:on

    /** RoPE tables over the rotated width only. */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(DeepSeek2Configuration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLength(), config.ropeDim(), config.ropeTheta(), false, 0, 0, 0, 0);
    }

    @Override
    protected DeepSeek2 createModel(
            DeepSeek2Configuration config, Tokenizer tokenizer, Weights weights) {
        if (config.qLoraRank() == 0) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "deepseek2 without a compressed query (the Lite form) is not supported");
        }
        return new DeepSeek2(
                config,
                tokenizer,
                weights,
                new Glm4ChatFormat((Glm4Tokenizer) tokenizer, vocabulary));
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DeepSeek2Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        DeepSeek2LayerWeights<FloatTensor> layers =
                layerWeights(
                        config,
                        tensorEntries,
                        new FloatTensor[0],
                        DeepSeek2ModelLoader::hostTensor);
        return new DeepSeek2StandardWeights(
                hostTensor(tokenEmbeddings),
                layers,
                hostTensor(required(tensorEntries, "output_norm.weight")),
                hostTensor(outputWeight),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                DataTypeMapping.sourceType(
                        required(tensorEntries, "blk.0.attn_q_a.weight").ggmlType()));
    }

    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DeepSeek2Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        DeepSeek2LayerWeights<TornadoTensor> layers =
                layerWeights(
                        config,
                        tensorEntries,
                        new TornadoTensor[0],
                        DeepSeek2ModelLoader::deviceTensor);
        return new DeepSeek2TornadoWeights(
                deviceTensor(tokenEmbeddings),
                layers,
                deviceTensor(required(tensorEntries, "output_norm.weight")),
                deviceTensor(outputWeight),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                DataTypeMapping.sourceType(
                        required(tensorEntries, "blk.0.attn_q_a.weight").ggmlType()));
    }

    /** A device tensor in the file's representation. Q5_0 has no device decoder here. */
    static TornadoTensor deviceTensor(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.Q5_0) {
            throw new UnsupportedOperationException(
                    "deepseek2 on the accelerator does not read Q5_0 tensors yet");
        }
        return ModelLoader.loadTornadoTensorNative(entry);
    }

    /** A host tensor, Q5_0 rewritten as Q8_0. */
    static FloatTensor hostTensor(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.Q5_0) {
            long elements = elementCount(entry);
            return new Q8_0FloatTensor(
                    Math.toIntExact(elements),
                    Q5_0Widening.toQ8_0(entry.memorySegment(), 0, elements, 0));
        }
        return ModelLoader.loadTensor(entry);
    }

    static long elementCount(GGMLTensorEntry entry) {
        long n = 1;
        for (int d : entry.shape()) {
            n *= d;
        }
        return n;
    }

    // @formatter:off
    /** Every block's tensors; slots a block does not carry stay null. */
    static <T> DeepSeek2LayerWeights<T> layerWeights(
            DeepSeek2Configuration config,
            Map<String, GGMLTensorEntry> entries,
            T[] type,
            java.util.function.Function<GGMLTensorEntry, T> load) {
        int n = config.numberOfLayers();
        T[] attnNorm = java.util.Arrays.copyOf(type, n),
                qA = java.util.Arrays.copyOf(type, n),
                qANorm = java.util.Arrays.copyOf(type, n),
                qB = java.util.Arrays.copyOf(type, n),
                kvAMqa = java.util.Arrays.copyOf(type, n),
                kvANorm = java.util.Arrays.copyOf(type, n),
                kB = java.util.Arrays.copyOf(type, n),
                vB = java.util.Arrays.copyOf(type, n),
                wo = java.util.Arrays.copyOf(type, n),
                ffnNorm = java.util.Arrays.copyOf(type, n),
                ffnGate = java.util.Arrays.copyOf(type, n),
                ffnUp = java.util.Arrays.copyOf(type, n),
                ffnDown = java.util.Arrays.copyOf(type, n),
                router = java.util.Arrays.copyOf(type, n),
                routerBias = java.util.Arrays.copyOf(type, n),
                gateExps = java.util.Arrays.copyOf(type, n),
                upExps = java.util.Arrays.copyOf(type, n),
                downExps = java.util.Arrays.copyOf(type, n),
                shGate = java.util.Arrays.copyOf(type, n),
                shUp = java.util.Arrays.copyOf(type, n),
                shDown = java.util.Arrays.copyOf(type, n);
        for (int l = 0; l < n; l++) {
            String b = "blk." + l + ".";
            attnNorm[l] = load.apply(required(entries, b + "attn_norm.weight"));
            qA[l] = load.apply(required(entries, b + "attn_q_a.weight"));
            qANorm[l] = load.apply(required(entries, b + "attn_q_a_norm.weight"));
            qB[l] = load.apply(required(entries, b + "attn_q_b.weight"));
            kvAMqa[l] = load.apply(required(entries, b + "attn_kv_a_mqa.weight"));
            kvANorm[l] = load.apply(required(entries, b + "attn_kv_a_norm.weight"));
            kB[l] = load.apply(required(entries, b + "attn_k_b.weight"));
            vB[l] = load.apply(required(entries, b + "attn_v_b.weight"));
            wo[l] = load.apply(required(entries, b + "attn_output.weight"));
            ffnNorm[l] = load.apply(required(entries, b + "ffn_norm.weight"));
            if (config.isDenseLayer(l)) {
                ffnGate[l] = load.apply(required(entries, b + "ffn_gate.weight"));
                ffnUp[l] = load.apply(required(entries, b + "ffn_up.weight"));
                ffnDown[l] = load.apply(required(entries, b + "ffn_down.weight"));
            } else {
                router[l] = load.apply(required(entries, b + "ffn_gate_inp.weight"));
                GGMLTensorEntry bias = entries.get(b + "exp_probs_b.bias");
                routerBias[l] = bias == null ? null : load.apply(bias);
                gateExps[l] = load.apply(required(entries, b + "ffn_gate_exps.weight"));
                upExps[l] = load.apply(required(entries, b + "ffn_up_exps.weight"));
                downExps[l] = load.apply(required(entries, b + "ffn_down_exps.weight"));
                if (config.sharedExperts() > 0) {
                    shGate[l] = load.apply(required(entries, b + "ffn_gate_shexp.weight"));
                    shUp[l] = load.apply(required(entries, b + "ffn_up_shexp.weight"));
                    shDown[l] = load.apply(required(entries, b + "ffn_down_shexp.weight"));
                }
            }
        }
        return new DeepSeek2LayerWeights<>(
                attnNorm,
                qA,
                qANorm,
                qB,
                kvAMqa,
                kvANorm,
                kB,
                vB,
                wo,
                ffnNorm,
                ffnGate,
                ffnUp,
                ffnDown,
                router,
                routerBias,
                gateExps,
                upExps,
                downExps,
                shGate,
                shUp,
                shDown);
    }

    // @formatter:on

    static GGMLTensorEntry required(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "deepseek2 expects "
                            + name
                            + ", which this file does not carry");
        }
        return entry;
    }
}
