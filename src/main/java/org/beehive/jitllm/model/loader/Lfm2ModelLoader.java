package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.tokenizer.Vocabulary.fromTokensAndScores;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.Lfm2StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.format.ChatFormat.ChatTokens;
import org.beehive.jitllm.model.format.Lfm2ChatFormat;
import org.beehive.jitllm.model.lfm2.Lfm2;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.Lfm2Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loads an {@code lfm2moe} GGUF.
 *
 * <p>Each layer carries one of two mixer weight sets — attention or short convolution — and one of
 * two feed-forward weight sets — dense or routed experts. Which, is decided by {@link
 * Lfm2Configuration#isAttentionLayer(int)} and {@link Lfm2Configuration#isMoeLayer(int)} and
 * nothing else, at load and at execution alike.
 *
 * <p>The metadata is validated before any weight is read: a router this engine does not implement,
 * or a geometry that does not divide, would otherwise load and compute plausible nonsense.
 */
public class Lfm2ModelLoader extends AbstractModelLoader<Lfm2, Lfm2Configuration> {

    private static final String PREFIX = "lfm2moe.";

    /** {@code LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID}, the only router implemented here. */
    private static final int GATING_SIGMOID = 2;

    /**
     * What a caller who asked for no particular context length gets. The 24B declares 131072, whose
     * key/value arrays are over five gigabytes on the host; this is the same default Qwen 3.5's
     * loader takes for the same reason.
     */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    public Lfm2ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Lfm2Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Lfm2Configuration createConfiguration(Map<String, Object> metadata) {
        int blocks = (int) metadata.get(PREFIX + "block_count");
        int modelContextLength = (int) metadata.get(PREFIX + "context_length");
        Lfm2Configuration config =
                new Lfm2Configuration(
                        getModelQuantization(metadata),
                        (int) metadata.get(PREFIX + "embedding_length"),
                        (int) metadata.get(PREFIX + "feed_forward_length"),
                        blocks,
                        (int) metadata.get(PREFIX + "attention.head_count"),
                        perLayer(metadata.get(PREFIX + "attention.head_count_kv"), blocks),
                        (int) metadata.get(PREFIX + "shortconv.l_cache"),
                        (int) metadata.get(PREFIX + "expert_count"),
                        (int) metadata.get(PREFIX + "expert_used_count"),
                        (int) metadata.get(PREFIX + "expert_feed_forward_length"),
                        (int) metadata.getOrDefault(PREFIX + "leading_dense_block_count", 0),
                        vocabulary.size(),
                        modelContextLength,
                        resolveContextLength(modelContextLength),
                        (float) metadata.get(PREFIX + "attention.layer_norm_rms_epsilon"),
                        (float) metadata.get(PREFIX + "rope.freq_base"));
        validate(config, metadata);
        return config;
    }

    // @formatter:on

    /** {@code head_count_kv} is an array in this family; a scalar means every layer attends. */
    private static int[] perLayer(Object value, int blocks) {
        if (value instanceof int[] array) {
            return array.clone();
        }
        int[] filled = new int[blocks];
        java.util.Arrays.fill(filled, (int) value);
        return filled;
    }

    private int resolveContextLength(int modelContextLength) {
        if (contextLength > 0) {
            return Math.min(contextLength, modelContextLength);
        }
        return Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
    }

    /** Refuses a metadata block this engine cannot run, naming the relationship that fails. */
    private static void validate(Lfm2Configuration config, Map<String, Object> metadata) {
        String prefix = DiagnosticCode.MODEL_MALFORMED.prefix();
        Object gating = metadata.get(PREFIX + "expert_gating_func");
        if (gating != null && (int) gating != GATING_SIGMOID) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.expert_gating_func is "
                            + gating
                            + "; only the sigmoid router ("
                            + GATING_SIGMOID
                            + ") is implemented");
        }
        Object scale = metadata.get(PREFIX + "expert_weights_scale");
        if (scale != null && (float) scale != 0f && (float) scale != 1f) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.expert_weights_scale is "
                            + scale
                            + "; scaled expert weights are not implemented");
        }
        if (config.headCountKv().length != config.numberOfLayers()) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.attention.head_count_kv has "
                            + config.headCountKv().length
                            + " entries for "
                            + config.numberOfLayers()
                            + " layers");
        }
        int kvHeads = config.numberOfKeyValueHeads();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int heads = config.headCountKv()[l];
            if (heads != 0 && heads != kvHeads) {
                throw new ModelLoadException(
                        prefix
                                + "lfm2moe attention layers disagree about their key/value heads: "
                                + kvHeads
                                + " and "
                                + heads
                                + " at layer "
                                + l);
            }
        }
        if (kvHeads == 0 || config.numberOfHeads() % kvHeads != 0) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.attention.head_count ("
                            + config.numberOfHeads()
                            + ") must be a positive multiple of head_count_kv ("
                            + kvHeads
                            + ")");
        }
        if (config.dim() % config.numberOfHeads() != 0) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.embedding_length ("
                            + config.dim()
                            + ") must divide by head_count ("
                            + config.numberOfHeads()
                            + "): the head width is their quotient");
        }
        if (config.shortConvCache() < 2) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.shortconv.l_cache must be at least 2, got "
                            + config.shortConvCache());
        }
        if (config.expertUsedCount() < 1 || config.expertUsedCount() > config.expertCount()) {
            throw new ModelLoadException(
                    prefix
                            + "lfm2moe.expert_used_count ("
                            + config.expertUsedCount()
                            + ") must be between 1 and expert_count ("
                            + config.expertCount()
                            + ")");
        }
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Lfm2Configuration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLength(), config.headSize(), config.ropeTheta(), false, 0, 0, 0, 0);
    }

    @Override
    protected Lfm2 createModel(Lfm2Configuration config, Tokenizer tokenizer, Weights weights) {
        ChatTokens chatTokens =
                new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|endoftext|>", "");
        return new Lfm2(
                config,
                tokenizer,
                weights,
                new Lfm2ChatFormat((Lfm2Tokenizer) tokenizer, chatTokens));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Lfm2Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int layers = config.numberOfLayers();
        FloatTensor[] attnNorm = new FloatTensor[layers];
        FloatTensor[] ffnNorm = new FloatTensor[layers];
        FloatTensor[] ffnGate = new FloatTensor[layers];
        FloatTensor[] ffnUp = new FloatTensor[layers];
        FloatTensor[] ffnDown = new FloatTensor[layers];
        FloatTensor[] ffnGateInp = new FloatTensor[layers];
        FloatTensor[] expProbsBias = new FloatTensor[layers];
        FloatTensor[] ffnGateExps = new FloatTensor[layers];
        FloatTensor[] ffnUpExps = new FloatTensor[layers];
        FloatTensor[] ffnDownExps = new FloatTensor[layers];
        FloatTensor[] wq = new FloatTensor[layers];
        FloatTensor[] wk = new FloatTensor[layers];
        FloatTensor[] wv = new FloatTensor[layers];
        FloatTensor[] wo = new FloatTensor[layers];
        FloatTensor[] attnQNorm = new FloatTensor[layers];
        FloatTensor[] attnKNorm = new FloatTensor[layers];
        FloatTensor[] convInProj = new FloatTensor[layers];
        FloatTensor[] convKernel = new FloatTensor[layers];
        FloatTensor[] convOutProj = new FloatTensor[layers];

        for (int l = 0; l < layers; l++) {
            String blk = "blk." + l + ".";
            attnNorm[l] = required(tensorEntries, blk + "attn_norm.weight");
            ffnNorm[l] = required(tensorEntries, blk + "ffn_norm.weight");
            if (config.isMoeLayer(l)) {
                ffnGateInp[l] = required(tensorEntries, blk + "ffn_gate_inp.weight");
                expProbsBias[l] = required(tensorEntries, blk + "exp_probs_b.bias");
                ffnGateExps[l] = required(tensorEntries, blk + "ffn_gate_exps.weight");
                ffnUpExps[l] = required(tensorEntries, blk + "ffn_up_exps.weight");
                ffnDownExps[l] = required(tensorEntries, blk + "ffn_down_exps.weight");
            } else {
                ffnGate[l] = required(tensorEntries, blk + "ffn_gate.weight");
                ffnUp[l] = required(tensorEntries, blk + "ffn_up.weight");
                ffnDown[l] = required(tensorEntries, blk + "ffn_down.weight");
            }
            if (config.isAttentionLayer(l)) {
                wq[l] = required(tensorEntries, blk + "attn_q.weight");
                wk[l] = required(tensorEntries, blk + "attn_k.weight");
                wv[l] = required(tensorEntries, blk + "attn_v.weight");
                wo[l] = required(tensorEntries, blk + "attn_output.weight");
                attnQNorm[l] = required(tensorEntries, blk + "attn_q_norm.weight");
                attnKNorm[l] = required(tensorEntries, blk + "attn_k_norm.weight");
            } else {
                convInProj[l] = required(tensorEntries, blk + "shortconv.in_proj.weight");
                convKernel[l] = required(tensorEntries, blk + "shortconv.conv.weight");
                convOutProj[l] = required(tensorEntries, blk + "shortconv.out_proj.weight");
            }
        }

        return new Lfm2StandardWeights(
                ModelLoader.loadTensor(tokenEmbeddings),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnUp,
                ffnDown,
                ffnGateInp,
                expProbsBias,
                ffnGateExps,
                ffnUpExps,
                ffnDownExps,
                wq,
                wk,
                wv,
                wo,
                attnQNorm,
                attnKNorm,
                convInProj,
                convKernel,
                convOutProj,
                required(tensorEntries, "token_embd_norm.weight"),
                ModelLoader.loadTensor(outputWeight),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                representativeType(tensorEntries, config, tokenEmbeddings));
    }

    // @formatter:on

    /**
     * The representation the model reports: that of the first short-convolution input projection.
     *
     * <p>Only a label on the host, where every tensor is read in its own representation. The file
     * is mixed by construction — Q4_K/Q5_K/Q6_K experts, Q8_0 attention and dense layers — and no
     * single type describes it.
     */
    private static org.beehive.jitllm.runtime.tensor.DataType representativeType(
            Map<String, GGMLTensorEntry> entries,
            Lfm2Configuration config,
            GGMLTensorEntry tokenEmbeddings) {
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (!config.isAttentionLayer(l)) {
                GGMLTensorEntry entry = entries.get("blk." + l + ".shortconv.in_proj.weight");
                return DataTypeMapping.sourceType(entry.ggmlType());
            }
        }
        return DataTypeMapping.sourceType(tokenEmbeddings.ggmlType());
    }

    private static FloatTensor required(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "lfm2moe expects "
                            + name
                            + ", which this file does not carry");
        }
        return ModelLoader.loadTensor(entry);
    }

    // @formatter:off
    /**
     * Device weights, every tensor in the representation the file gave it.
     *
     * <p>Nothing is materialized: the 24B's experts are Q4_K, Q5_K and Q6_K, its attention and
     * dense layers Q8_0, and every layer graph task is chosen by the representation of the tensor
     * it reads. Materializing the K-quants as Q8_0 would nearly double the 14 GB the file holds.
     */
    // @formatter:on
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Lfm2Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int layers = config.numberOfLayers();
        TornadoTensor[] attnNorm = new TornadoTensor[layers];
        TornadoTensor[] ffnNorm = new TornadoTensor[layers];
        TornadoTensor[] ffnGate = new TornadoTensor[layers];
        TornadoTensor[] ffnUp = new TornadoTensor[layers];
        TornadoTensor[] ffnDown = new TornadoTensor[layers];
        TornadoTensor[] ffnGateInp = new TornadoTensor[layers];
        TornadoTensor[] expProbsBias = new TornadoTensor[layers];
        TornadoTensor[] ffnGateExps = new TornadoTensor[layers];
        TornadoTensor[] ffnUpExps = new TornadoTensor[layers];
        TornadoTensor[] ffnDownExps = new TornadoTensor[layers];
        TornadoTensor[] wq = new TornadoTensor[layers];
        TornadoTensor[] wk = new TornadoTensor[layers];
        TornadoTensor[] wv = new TornadoTensor[layers];
        TornadoTensor[] wo = new TornadoTensor[layers];
        TornadoTensor[] attnQNorm = new TornadoTensor[layers];
        TornadoTensor[] attnKNorm = new TornadoTensor[layers];
        TornadoTensor[] convInProj = new TornadoTensor[layers];
        TornadoTensor[] convKernel = new TornadoTensor[layers];
        TornadoTensor[] convOutProj = new TornadoTensor[layers];

        for (int l = 0; l < layers; l++) {
            String blk = "blk." + l + ".";
            attnNorm[l] = device(tensorEntries, blk + "attn_norm.weight");
            ffnNorm[l] = device(tensorEntries, blk + "ffn_norm.weight");
            if (config.isMoeLayer(l)) {
                ffnGateInp[l] = device(tensorEntries, blk + "ffn_gate_inp.weight");
                expProbsBias[l] = device(tensorEntries, blk + "exp_probs_b.bias");
                ffnGateExps[l] = device(tensorEntries, blk + "ffn_gate_exps.weight");
                ffnUpExps[l] = device(tensorEntries, blk + "ffn_up_exps.weight");
                ffnDownExps[l] = device(tensorEntries, blk + "ffn_down_exps.weight");
            } else {
                ffnGate[l] = device(tensorEntries, blk + "ffn_gate.weight");
                ffnUp[l] = device(tensorEntries, blk + "ffn_up.weight");
                ffnDown[l] = device(tensorEntries, blk + "ffn_down.weight");
            }
            if (config.isAttentionLayer(l)) {
                wq[l] = device(tensorEntries, blk + "attn_q.weight");
                wk[l] = device(tensorEntries, blk + "attn_k.weight");
                wv[l] = device(tensorEntries, blk + "attn_v.weight");
                wo[l] = device(tensorEntries, blk + "attn_output.weight");
                attnQNorm[l] = device(tensorEntries, blk + "attn_q_norm.weight");
                attnKNorm[l] = device(tensorEntries, blk + "attn_k_norm.weight");
            } else {
                convInProj[l] = device(tensorEntries, blk + "shortconv.in_proj.weight");
                convKernel[l] = device(tensorEntries, blk + "shortconv.conv.weight");
                convOutProj[l] = device(tensorEntries, blk + "shortconv.out_proj.weight");
            }
        }

        return new Lfm2TornadoWeights(
                ModelLoader.loadTornadoTensorNative(tokenEmbeddings),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                wq,
                wk,
                wv,
                wo,
                attnQNorm,
                attnKNorm,
                ffnGateInp,
                expProbsBias,
                ffnGateExps,
                ffnUpExps,
                ffnDownExps,
                convInProj,
                convKernel,
                convOutProj,
                device(tensorEntries, "token_embd_norm.weight"),
                ModelLoader.loadTornadoTensorNative(outputWeight),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
                representativeType(tensorEntries, config, tokenEmbeddings));
    }

    private static TornadoTensor device(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "lfm2moe expects "
                            + name
                            + ", which this file does not carry");
        }
        return ModelLoader.loadTornadoTensorNative(entry);
    }
}
