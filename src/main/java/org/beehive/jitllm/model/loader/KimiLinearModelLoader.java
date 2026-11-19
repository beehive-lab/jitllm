package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.tokenizer.Vocabulary.fromTokens;

import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.KdaWeights;
import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.KimiLinearStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.KimiLinearTornadoWeights;
import org.beehive.jitllm.model.format.KimiChatFormat;
import org.beehive.jitllm.model.kimilinear.KimiLinear;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.KimiTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loads a llama.cpp {@code kimi-linear} GGUF (Kimi-Linear-48B-A3B).
 *
 * <p>A block is KDA when its {@code attention.head_count_kv} is zero and MLA otherwise; the first
 * {@code leading_dense_block_count} blocks have a dense feed-forward, the rest a mixture of experts
 * with shared experts. The MLA key tail is not rotated: its RoPE tables are the identity, so the
 * rotation the latent-attention kernels apply leaves it as it is. Every tensor keeps the
 * representation the file gave it on the device; on the host Q5_0 is widened to Q8_0, as for {@code
 * deepseek2}.
 */
public class KimiLinearModelLoader
        extends AbstractModelLoader<KimiLinear, KimiLinearConfiguration> {

    /** The {@code general.architecture} llama.cpp gives Kimi-Linear, and its key prefix. */
    public static final String ARCHITECTURE = "kimi-linear";

    /** The declared 1M positions would not fit a host session; the same default as deepseek2. */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    /** llama.cpp's {@code LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID}. */
    private static final int SIGMOID_GATING = 2;

    public KimiLinearModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokens(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new KimiTokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected KimiLinearConfiguration createConfiguration(Map<String, Object> metadata) {
        String p = ARCHITECTURE + ".";
        int modelContextLength = (int) metadata.get(p + "context_length");
        int finalContextLength =
                contextLength > 0
                        ? Math.min(contextLength, modelContextLength)
                        : Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
        int layers = (int) metadata.get(p + "block_count");
        int[] keyValueHeads = perLayer(metadata.get(p + "attention.head_count_kv"), layers);
        int[] attentionTypes = new int[layers];
        for (int l = 0; l < layers; l++) {
            attentionTypes[l] =
                    keyValueHeads[l] == 0
                            ? KimiLinearConfiguration.KDA
                            : KimiLinearConfiguration.MLA;
        }
        int gating = (int) metadata.getOrDefault(p + "expert_gating_func", SIGMOID_GATING);
        if (gating != SIGMOID_GATING) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "kimi-linear with expert gating function "
                            + gating
                            + " is not supported; the model routes with sigmoid scores");
        }
        int ropeDim = (int) metadata.get(p + "rope.dimension_count");
        int keyMla = (int) metadata.get(p + "attention.key_length_mla");
        int heads = (int) metadata.get(p + "attention.head_count");
        return new KimiLinearConfiguration(
                getModelQuantization(metadata),
                (int) metadata.get(p + "embedding_length"),
                (int) metadata.get(p + "feed_forward_length"),
                layers,
                attentionTypes,
                (int) metadata.getOrDefault(p + "leading_dense_block_count", 0),
                heads,
                (int) metadata.get(p + "kda.head_dim"),
                (int) metadata.get(p + "ssm.conv_kernel"),
                heads,
                (int) metadata.get(p + "attention.kv_lora_rank"),
                keyMla - ropeDim,
                ropeDim,
                (int) metadata.get(p + "attention.value_length_mla"),
                (int) metadata.get(p + "expert_count"),
                (int) metadata.get(p + "expert_used_count"),
                perLayer(metadata.get(p + "expert_feed_forward_length"), 1)[0],
                (int) metadata.get(p + "expert_shared_count"),
                (float) metadata.getOrDefault(p + "expert_weights_scale", 1.0f),
                // The model renormalizes the chosen weights (moe_renormalize); llama.cpp's
                // build_moe_ffn is called with normalization on for this architecture.
                (boolean) metadata.getOrDefault(p + "expert_weights_norm", true),
                this.vocabulary.size(),
                modelContextLength,
                finalContextLength,
                (float) metadata.get(p + "attention.layer_norm_rms_epsilon"));
    }

    // @formatter:on

    /** A per-layer integer, written either once for every layer or as an array. */
    private static int[] perLayer(Object value, int layers) {
        if (value instanceof int[] values) {
            return values;
        }
        int[] values = new int[layers];
        Arrays.fill(values, (int) value);
        return values;
    }

    /**
     * The identity: every pair's cosine is 1 and sine 0, so the latent-attention rotation leaves
     * the key tail and the query tail exactly as they are, as the model (with no RoPE) requires.
     */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(KimiLinearConfiguration config) {
        int n = config.contextLength() * (config.ropeDim() / 2);
        float[] cos = new float[n];
        Arrays.fill(cos, 1f);
        return new Pair<>(cos, new float[n]);
    }

    @Override
    protected KimiLinear createModel(
            KimiLinearConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new KimiLinear(
                config, tokenizer, weights, new KimiChatFormat(tokenizer, vocabulary));
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            KimiLinearConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new KimiLinearStandardWeights(
                DeepSeek2ModelLoader.hostTensor(tokenEmbeddings),
                layerWeights(
                        config,
                        tensorEntries,
                        new FloatTensor[0],
                        DeepSeek2ModelLoader::hostTensor),
                DeepSeek2ModelLoader.hostTensor(required(tensorEntries, "output_norm.weight")),
                DeepSeek2ModelLoader.hostTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            KimiLinearConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new KimiLinearTornadoWeights(
                ModelLoader.loadTornadoTensorNative(tokenEmbeddings),
                layerWeights(
                        config,
                        tensorEntries,
                        new TornadoTensor[0],
                        ModelLoader::loadTornadoTensorNative),
                ModelLoader.loadTornadoTensorNative(required(tensorEntries, "output_norm.weight")),
                ModelLoader.loadTornadoTensorNative(outputWeight),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    // @formatter:off
    /** Every layer's tensors; slots a layer does not carry stay null. */
    static <T> KimiLinearLayerWeights<T> layerWeights(
            KimiLinearConfiguration config,
            Map<String, GGMLTensorEntry> entries,
            T[] type,
            Function<GGMLTensorEntry, T> load) {
        int n = config.numberOfLayers();
        T[] attnNorm = Arrays.copyOf(type, n), ffnNorm = Arrays.copyOf(type, n);
        T[] q = Arrays.copyOf(type, n),
                k = Arrays.copyOf(type, n),
                v = Arrays.copyOf(type, n),
                convQ = Arrays.copyOf(type, n),
                convK = Arrays.copyOf(type, n),
                convV = Arrays.copyOf(type, n),
                decayA = Arrays.copyOf(type, n),
                decayB = Arrays.copyOf(type, n),
                dtBias = Arrays.copyOf(type, n),
                a = Arrays.copyOf(type, n),
                beta = Arrays.copyOf(type, n),
                gateA = Arrays.copyOf(type, n),
                gateB = Arrays.copyOf(type, n),
                outNorm = Arrays.copyOf(type, n),
                kdaOut = Arrays.copyOf(type, n);
        T[] mlaQ = Arrays.copyOf(type, n),
                kvAMqa = Arrays.copyOf(type, n),
                kvANorm = Arrays.copyOf(type, n),
                kB = Arrays.copyOf(type, n),
                vB = Arrays.copyOf(type, n),
                mlaOut = Arrays.copyOf(type, n);
        T[] gate = Arrays.copyOf(type, n),
                up = Arrays.copyOf(type, n),
                down = Arrays.copyOf(type, n),
                router = Arrays.copyOf(type, n),
                routerBias = Arrays.copyOf(type, n),
                gateExps = Arrays.copyOf(type, n),
                upExps = Arrays.copyOf(type, n),
                downExps = Arrays.copyOf(type, n),
                shGate = Arrays.copyOf(type, n),
                shUp = Arrays.copyOf(type, n),
                shDown = Arrays.copyOf(type, n);

        for (int l = 0; l < n; l++) {
            String b = "blk." + l + ".";
            Function<String, T> get = name -> load.apply(required(entries, b + name));
            attnNorm[l] = get.apply("attn_norm.weight");
            ffnNorm[l] = get.apply("ffn_norm.weight");
            if (config.isKda(l)) {
                q[l] = get.apply("attn_q.weight");
                k[l] = get.apply("attn_k.weight");
                v[l] = get.apply("attn_v.weight");
                convQ[l] = get.apply("ssm_conv1d_q.weight");
                convK[l] = get.apply("ssm_conv1d_k.weight");
                convV[l] = get.apply("ssm_conv1d_v.weight");
                decayA[l] = get.apply("ssm_f_a.weight");
                decayB[l] = get.apply("ssm_f_b.weight");
                dtBias[l] = get.apply("ssm_dt.bias");
                a[l] = get.apply("ssm_a");
                beta[l] = get.apply("ssm_beta.weight");
                gateA[l] = get.apply("ssm_g_a.weight");
                gateB[l] = get.apply("ssm_g_b.weight");
                outNorm[l] = get.apply("ssm_norm.weight");
                kdaOut[l] = get.apply("attn_output.weight");
            } else {
                mlaQ[l] = get.apply("attn_q.weight");
                kvAMqa[l] = get.apply("attn_kv_a_mqa.weight");
                kvANorm[l] = get.apply("attn_kv_a_norm.weight");
                kB[l] = get.apply("attn_k_b.weight");
                vB[l] = get.apply("attn_v_b.weight");
                mlaOut[l] = get.apply("attn_output.weight");
            }
            if (config.isDenseLayer(l)) {
                gate[l] = get.apply("ffn_gate.weight");
                up[l] = get.apply("ffn_up.weight");
                down[l] = get.apply("ffn_down.weight");
            } else {
                router[l] = get.apply("ffn_gate_inp.weight");
                routerBias[l] = get.apply("exp_probs_b.bias");
                gateExps[l] = get.apply("ffn_gate_exps.weight");
                upExps[l] = get.apply("ffn_up_exps.weight");
                downExps[l] = get.apply("ffn_down_exps.weight");
                if (config.sharedExperts() > 0) {
                    shGate[l] = get.apply("ffn_gate_shexp.weight");
                    shUp[l] = get.apply("ffn_up_shexp.weight");
                    shDown[l] = get.apply("ffn_down_shexp.weight");
                }
            }
        }
        return new KimiLinearLayerWeights<>(
                attnNorm,
                ffnNorm,
                new KdaWeights<>(
                        q, k, v, convQ, convK, convV, decayA, decayB, dtBias, a, beta, gateA, gateB,
                        outNorm, kdaOut),
                new KimiLinearLayerWeights.Mla<>(mlaQ, kvAMqa, kvANorm, kB, vB, mlaOut),
                new KimiLinearLayerWeights.FeedForward<>(
                        gate,
                        up,
                        down,
                        router,
                        routerBias,
                        gateExps,
                        upExps,
                        downExps,
                        shGate,
                        shUp,
                        shDown));
    }

    // @formatter:on

    static GGMLTensorEntry required(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "kimi-linear expects "
                            + name
                            + ", which this file does not carry");
        }
        return entry;
    }
}
