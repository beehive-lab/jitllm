package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.MXFP4TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.GptOssStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.GptOssTornadoWeights;
import org.beehive.jitllm.model.format.HarmonyChatFormat;
import org.beehive.jitllm.model.gptoss.GptOss;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.GptOssTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/** Loads {@code gpt-oss} GGUF files (llama.cpp's {@code ggml-org/gpt-oss-*-GGUF}). */
public class GptOssModelLoader extends AbstractModelLoader<GptOss, GptOssConfiguration> {

    private static final String ARCH = "gpt-oss.";

    public GptOssModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new GptOssTokenizer(metadata, vocabulary);
    }

    @Override
    protected GptOssConfiguration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get(ARCH + "context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;
        int heads = (int) metadata.get(ARCH + "attention.head_count");
        int keyLength = (int) metadata.get(ARCH + "attention.key_length");
        int valueLength = (int) metadata.getOrDefault(ARCH + "attention.value_length", keyLength);
        if (keyLength != valueLength) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "gpt-oss with key length "
                            + keyLength
                            + " and value length "
                            + valueLength
                            + " is not supported: the attention reads one head width");
        }
        String scaling = (String) metadata.getOrDefault(ARCH + "rope.scaling.type", "none");
        float factor =
                "yarn".equals(scaling)
                        ? (float) metadata.getOrDefault(ARCH + "rope.scaling.factor", 1f)
                        : 1f;
        int pattern =
                metadata.get(ARCH + "attention.sliding_window_pattern") instanceof Integer p
                        ? p
                        : 2;
        GptOssConfiguration config =
                new GptOssConfiguration(
                        getModelQuantization(metadata),
                        (int) metadata.get(ARCH + "embedding_length"),
                        (int) metadata.get(ARCH + "block_count"),
                        heads,
                        (int) metadata.get(ARCH + "attention.head_count_kv"),
                        keyLength,
                        vocabulary.size(),
                        modelContextLength,
                        finalContextLength,
                        (int) metadata.get(ARCH + "expert_count"),
                        (int) metadata.get(ARCH + "expert_used_count"),
                        (int) metadata.get(ARCH + "expert_feed_forward_length"),
                        (float) metadata.get(ARCH + "attention.layer_norm_rms_epsilon"),
                        (float) metadata.get(ARCH + "rope.freq_base"),
                        factor,
                        (int)
                                metadata.getOrDefault(
                                        ARCH + "rope.scaling.original_context_length",
                                        modelContextLength),
                        (float) metadata.getOrDefault(ARCH + "rope.scaling.yarn_beta_fast", 32f),
                        (float) metadata.getOrDefault(ARCH + "rope.scaling.yarn_beta_slow", 1f),
                        (int) metadata.get(ARCH + "attention.sliding_window"),
                        pattern);
        validate(config);
        return config;
    }

    /** Rejects a block this implementation cannot run, before any weight is read. */
    static void validate(GptOssConfiguration c) {
        String problem = null;
        if (c.headSize() % 2 != 0) {
            problem = "an odd head size " + c.headSize();
        } else if (c.numberOfHeads() % c.numberOfKeyValueHeads() != 0) {
            problem = c.numberOfHeads() + " query heads over " + c.numberOfKeyValueHeads();
        } else if (c.dim() % 32 != 0 || c.expertHiddenDim() % 32 != 0 || c.queryDim() % 32 != 0) {
            problem = "a width that is not a whole number of 32-value blocks";
        } else if (c.numberOfExpertsUsed() > c.numberOfExperts()) {
            problem = "more experts used than exist";
        }
        if (problem != null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix() + "gpt-oss with " + problem);
        }
    }

    /**
     * YaRN, as ggml computes it for {@code ggml_rope_ext} with {@code ext_factor = 1}: each pair's
     * angle blends the interpolated and extrapolated frequency by a ramp over the correction
     * dimensions, and both sine and cosine carry the magnitude {@code 1 + 0.1 ln(factor)}.
     * llama.cpp sets {@code attn_factor} so that this is the only magnitude applied.
     */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(GptOssConfiguration config) {
        return yarn(
                config.contextLength(),
                config.headSize(),
                config.ropeTheta(),
                config.ropeScalingFactor(),
                config.ropeOriginalContextLength(),
                config.yarnBetaFast(),
                config.yarnBetaSlow());
    }

    static Pair<float[], float[]> yarn(
            int contextLength,
            int headSize,
            float base,
            float factor,
            int originalContext,
            float betaFast,
            float betaSlow) {
        int half = headSize / 2;
        float[] cr = new float[contextLength * half];
        float[] ci = new float[contextLength * half];
        float freqScale = 1f / factor;
        boolean extend = factor > 1f;
        float low =
                Math.max(
                        0f, (float) Math.floor(corrDim(headSize, originalContext, betaFast, base)));
        float high =
                Math.min(
                        headSize - 1f,
                        (float) Math.ceil(corrDim(headSize, originalContext, betaSlow, base)));
        float mscale = extend ? 1f + 0.1f * (float) Math.log(1f / freqScale) : 1f;
        float[] invFreq = new float[half];
        float[] mix = new float[half];
        float thetaScale = (float) Math.pow(base, -2.0 / headSize);
        float theta = 1f;
        for (int i = 0; i < half; i++) {
            invFreq[i] = theta;
            float y = (i - low) / Math.max(0.001f, high - low);
            mix[i] = extend ? 1f - Math.min(1f, Math.max(0f, y)) : 1f;
            theta *= thetaScale;
        }
        for (int pos = 0; pos < contextLength; pos++) {
            for (int i = 0; i < half; i++) {
                float extrap = pos * invFreq[i];
                float interp = freqScale * extrap;
                float angle = interp * (1f - mix[i]) + extrap * mix[i];
                cr[pos * half + i] = (float) Math.cos(angle) * mscale;
                ci[pos * half + i] = (float) Math.sin(angle) * mscale;
            }
        }
        return new Pair<>(cr, ci);
    }

    /** ggml's {@code ggml_rope_yarn_corr_dim}. */
    private static double corrDim(int nDims, int nCtxOrig, float nRot, float base) {
        return nDims * Math.log(nCtxOrig / (nRot * 2 * Math.PI)) / (2 * Math.log(base));
    }

    @Override
    protected GptOss createModel(GptOssConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new GptOss(
                config, tokenizer, weights, new HarmonyChatFormat((GptOssTokenizer) tokenizer));
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            GptOssConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        int nl = config.numberOfLayers();
        GptOssLayerWeights<FloatTensor> layers =
                new GptOssLayerWeights<>(
                        cpu(nl, tensorEntries, "attn_q.bias"),
                        cpu(nl, tensorEntries, "attn_k.bias"),
                        cpu(nl, tensorEntries, "attn_v.bias"),
                        cpu(nl, tensorEntries, "attn_output.bias"),
                        cpu(nl, tensorEntries, "attn_sinks.weight"),
                        cpu(nl, tensorEntries, "ffn_gate_inp.weight"),
                        cpu(nl, tensorEntries, "ffn_gate_inp.bias"),
                        cpu(nl, tensorEntries, "ffn_gate_exps.weight"),
                        cpu(nl, tensorEntries, "ffn_gate_exps.bias"),
                        cpu(nl, tensorEntries, "ffn_up_exps.weight"),
                        cpu(nl, tensorEntries, "ffn_up_exps.bias"),
                        cpu(nl, tensorEntries, "ffn_down_exps.weight"),
                        cpu(nl, tensorEntries, "ffn_down_exps.bias"));
        return new GptOssStandardWeights(
                loadTensor(tokenEmbeddings),
                cpu(nl, tensorEntries, "attn_norm.weight"),
                cpu(nl, tensorEntries, "attn_q.weight"),
                cpu(nl, tensorEntries, "attn_k.weight"),
                cpu(nl, tensorEntries, "attn_v.weight"),
                cpu(nl, tensorEntries, "attn_output.weight"),
                cpu(nl, tensorEntries, "post_attention_norm.weight"),
                layers,
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            GptOssConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        int nl = config.numberOfLayers();
        TornadoTensor[] wq = gpu(nl, tensorEntries, "attn_q.weight");
        TornadoTensor[] wk = gpu(nl, tensorEntries, "attn_k.weight");
        TornadoTensor[] wv = gpu(nl, tensorEntries, "attn_v.weight");
        TornadoTensor[] wo = gpu(nl, tensorEntries, "attn_output.weight");
        GptOssLayerWeights<TornadoTensor> layers =
                new GptOssLayerWeights<>(
                        gpu(nl, tensorEntries, "attn_q.bias"),
                        gpu(nl, tensorEntries, "attn_k.bias"),
                        gpu(nl, tensorEntries, "attn_v.bias"),
                        gpu(nl, tensorEntries, "attn_output.bias"),
                        gpu(nl, tensorEntries, "attn_sinks.weight"),
                        gpu(nl, tensorEntries, "ffn_gate_inp.weight"),
                        gpu(nl, tensorEntries, "ffn_gate_inp.bias"),
                        mxfp4(gpu(nl, tensorEntries, "ffn_gate_exps.weight")),
                        gpu(nl, tensorEntries, "ffn_gate_exps.bias"),
                        mxfp4(gpu(nl, tensorEntries, "ffn_up_exps.weight")),
                        gpu(nl, tensorEntries, "ffn_up_exps.bias"),
                        mxfp4(gpu(nl, tensorEntries, "ffn_down_exps.weight")),
                        gpu(nl, tensorEntries, "ffn_down_exps.bias"));
        return new GptOssTornadoWeights(
                loadTornadoTensorNative(tokenEmbeddings),
                gpu(nl, tensorEntries, "attn_norm.weight"),
                wq,
                wk,
                wv,
                wo,
                gpu(nl, tensorEntries, "post_attention_norm.weight"),
                layers,
                loadTornadoTensorNative(tensorEntries.get("output_norm.weight")),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                loadTornadoTensorNative(outputWeight),
                keptType(wq, wk, wv, wo));
    }

    private static FloatTensor[] cpu(
            int layers, Map<String, GGMLTensorEntry> tensorEntries, String role) {
        return loadArrayOfTensors(layers, i -> required(tensorEntries, i, role));
    }

    private static TornadoTensor[] gpu(
            int layers, Map<String, GGMLTensorEntry> tensorEntries, String role) {
        return loadArrayOfTornadoTensorsNative(layers, i -> required(tensorEntries, i, role));
    }

    /** The experts, checked to be MXFP4: the device layers read no other representation. */
    private static TornadoTensor[] mxfp4(TornadoTensor[] experts) {
        for (TornadoTensor expert : experts) {
            if (!(expert instanceof MXFP4TornadoTensor)) {
                throw new ModelLoadException(
                        DiagnosticCode.MODEL_MALFORMED.prefix()
                                + "gpt-oss experts are "
                                + expert.dataType()
                                + "; the device layers read MXFP4 experts only");
            }
        }
        return experts;
    }

    private static GGMLTensorEntry required(
            Map<String, GGMLTensorEntry> tensorEntries, int layer, String role) {
        String name = "blk." + layer + "." + role;
        GGMLTensorEntry entry = tensorEntries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix() + "gpt-oss file has no " + name);
        }
        return entry;
    }
}
