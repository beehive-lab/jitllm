package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.tokenizer.Vocabulary.fromTokens;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Glm5NextLayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.Glm5NextStandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Glm5NextTornadoWeights;
import org.beehive.jitllm.model.format.Glm4ChatFormat;
import org.beehive.jitllm.model.glm5next.Glm5Next;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.Glm4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loads a llama.cpp {@code glm5-next} GGUF: the text backbone of GLM-5.3-Flash, as llama.cpp's
 * converter writes it, or {@code scripts/glm5next/convert.py} for a slice of the layers.
 *
 * <p>A block is KDA when its {@code attention.head_count_kv} is zero and sparse attention
 * otherwise; the first {@code leading_dense_block_count} blocks have a dense feed-forward. The MTP
 * layers, {@code nextn_predict_layers} of them, are the last blocks. Each layer's routed experts
 * are one tensor per projection in the file; they are loaded as stacks of {@link
 * Glm5NextConfiguration#expertChunkSize} experts, so that no stack reaches 2³¹ bytes. The tokenizer
 * is GLM-4's. The accelerator path runs the trunk only; the MTP layer is loaded but not executed
 * there.
 */
public class Glm5NextModelLoader extends AbstractModelLoader<Glm5Next, Glm5NextConfiguration> {

    /** The {@code general.architecture} llama.cpp gives GLM-5.3-Flash, and its key prefix. */
    public static final String ARCHITECTURE = "glm5-next";

    /** The declared 1M positions would not fit a host session; the same default as deepseek2. */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    /** The mHC expansion's sigmoid scale: fixed by the model, so not in the file. */
    private static final float HC_POST_SCALE = 2.0f;

    /** The KDA decay's lower bound when the file leaves it out, as llama.cpp assumes. */
    private static final float DEFAULT_GATE_LOWER_BOUND = -5.0f;

    /** The most bytes one loaded expert stack may take: a device array is int-indexed. */
    private static final long MAX_STACK_BYTES = Integer.MAX_VALUE - 4096L;

    /** Lowers {@link #MAX_STACK_BYTES}, so small models can be loaded as several stacks too. */
    public static final String STACK_BYTES_PROPERTY = "jitllm.glm5next.expertStackBytes";

    public Glm5NextModelLoader(
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
    protected Glm5NextConfiguration createConfiguration(Map<String, Object> metadata) {
        String p = ARCHITECTURE + ".";
        int modelContextLength = (int) metadata.get(p + "context_length");
        int finalContextLength =
                contextLength > 0
                        ? Math.min(contextLength, modelContextLength)
                        : Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
        int nextn = (int) metadata.getOrDefault(p + "nextn_predict_layers", 0);
        int layers = (int) metadata.get(p + "block_count") - nextn;
        int[] keyValueHeads = perLayer(metadata.get(p + "attention.head_count_kv"), layers + nextn);
        int denseLead = (int) metadata.getOrDefault(p + "leading_dense_block_count", 0);
        int[] attentionTypes = new int[layers];
        int[] ffnTypes = new int[layers];
        for (int l = 0; l < layers; l++) {
            attentionTypes[l] =
                    keyValueHeads[l] == 0
                            ? Glm5NextConfiguration.KDA
                            : Glm5NextConfiguration.SPARSE_ATTENTION;
            ffnTypes[l] =
                    l < denseLead ? Glm5NextConfiguration.DENSE_FFN : Glm5NextConfiguration.MOE_FFN;
        }
        if (metadata.get(p + "attention.indexer.types") instanceof boolean[] full) {
            for (boolean f : full) {
                if (!f) {
                    throw new ModelLoadException(
                            DiagnosticCode.MODEL_MALFORMED.prefix()
                                    + "glm5-next layers that share another layer's indexer are not"
                                    + " supported");
                }
            }
        }
        int heads = (int) metadata.get(p + "attention.head_count");
        int expertCount = (int) metadata.get(p + "expert_count");
        return new Glm5NextConfiguration(
                quantization(metadata),
                (int) metadata.get(p + "embedding_length"),
                (int) metadata.get(p + "feed_forward_length"),
                layers,
                nextn,
                attentionTypes,
                ffnTypes,
                this.vocabulary.size(),
                modelContextLength,
                finalContextLength,
                (float) metadata.get(p + "attention.layer_norm_rms_epsilon"),
                (int) metadata.get(p + "hyper_connection.count"),
                (int) metadata.get(p + "hyper_connection.sinkhorn_iterations"),
                (float) metadata.get(p + "hyper_connection.epsilon"),
                HC_POST_SCALE,
                heads,
                (int) metadata.get(p + "kda.head_dim"),
                (int) metadata.get(p + "ssm.conv_kernel"),
                (float) metadata.getOrDefault(p + "kda.gate_lower_bound", DEFAULT_GATE_LOWER_BOUND),
                heads,
                (int) metadata.get(p + "attention.q_lora_rank"),
                (int) metadata.get(p + "attention.kv_lora_rank"),
                (int) metadata.get(p + "attention.key_length_mla"),
                (int) metadata.get(p + "attention.value_length_mla"),
                (int) metadata.get(p + "attention.indexer.head_count"),
                (int) metadata.get(p + "attention.indexer.key_length"),
                (int) metadata.get(p + "attention.indexer.top_k"),
                (int) metadata.get(p + "attention.indexer.kpool"),
                (boolean) metadata.getOrDefault(p + "attention.indexer.kpool_select_tail", true),
                expertCount,
                (int) metadata.get(p + "expert_used_count"),
                perLayer(metadata.get(p + "expert_feed_forward_length"), 1)[0],
                (int) metadata.get(p + "expert_shared_count"),
                expertChunkSize(expertCount),
                (float) metadata.getOrDefault(p + "expert_weights_scale", 1.0f),
                (boolean) metadata.getOrDefault(p + "expert_weights_norm", false),
                swigluLimit(metadata.get(p + "swiglu_clamp_exp")));
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

    /** The SwiGLU clamp, one value for every layer; none means no clamp. */
    private static float swigluLimit(Object value) {
        if (value == null) {
            return Float.POSITIVE_INFINITY;
        }
        if (value instanceof float[] values) {
            float limit = 0f;
            for (float v : values) {
                limit = Math.max(limit, v);
            }
            return limit > 0f ? limit : Float.POSITIVE_INFINITY;
        }
        return (float) value;
    }

    /**
     * How many experts a loaded stack holds: the fewest stacks that keep each under {@link
     * #MAX_STACK_BYTES} (or {@link #STACK_BYTES_PROPERTY}), the experts shared out evenly between
     * them.
     */
    private int expertChunkSize(int expertCount) {
        long limit = Math.min(MAX_STACK_BYTES, Long.getLong(STACK_BYTES_PROPERTY, MAX_STACK_BYTES));
        long bytes = 0;
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            if (info.name().endsWith("_exps.weight")) {
                bytes = Math.max(bytes, byteSize(info.ggmlType(), info.dimensions()));
            }
        }
        long stacks = Math.max(1, (bytes + limit - 1) / limit);
        return (int) ((expertCount + stacks - 1) / stacks);
    }

    private static long byteSize(GGMLType type, int[] dimensions) {
        long elements = 1;
        for (int d : dimensions) {
            elements *= d;
        }
        return elements * type.getTypeSize() / type.getBlockSize();
    }

    /** F32 files (the reference model's) report F32; the rest as {@link #getModelQuantization}. */
    private static String quantization(Map<String, Object> metadata) {
        return (int) metadata.get("general.file_type") == 0
                ? "F32"
                : getModelQuantization(metadata);
    }

    /** No RoPE anywhere: no tables. */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Glm5NextConfiguration config) {
        return new Pair<>(new float[0], new float[0]);
    }

    @Override
    protected Glm5Next createModel(
            Glm5NextConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Glm5Next(
                config,
                tokenizer,
                weights,
                new Glm4ChatFormat((Glm4Tokenizer) tokenizer, vocabulary));
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Glm5NextConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Glm5NextStandardWeights(
                ModelLoader.loadTensor(tokenEmbeddings),
                layerWeights(
                        config,
                        withExpertStacks(tensorEntries, config, false),
                        new FloatTensor[0],
                        ModelLoader::loadTensor),
                ModelLoader.loadTensor(required(tensorEntries, "output_norm.weight")),
                ModelLoader.loadTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    /**
     * Device weights, every tensor as the file holds it. The decode plan reads Q8_0 projections;
     * its provider refuses any other weight type before a plan is built.
     */
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Glm5NextConfiguration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Glm5NextTornadoWeights(
                ModelLoader.loadTornadoTensorNative(tokenEmbeddings),
                layerWeights(
                        config,
                        withExpertStacks(tensorEntries, config, true),
                        new TornadoTensor[0],
                        ModelLoader::loadTornadoTensorNative),
                ModelLoader.loadTornadoTensorNative(required(tensorEntries, "output_norm.weight")),
                ModelLoader.loadTornadoTensorNative(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    /**
     * The entries with each layer's expert tensors also split into stacks of {@link
     * Glm5NextConfiguration#expertChunkSize} experts, named {@code ffn_*_exps.<stack>.weight}. A
     * stack is a slice of the mapped file; for the device it is mapped again with room for the
     * native array header in front, as {@link GGUF#loadTensorsTornado} maps a whole tensor.
     */
    private Map<String, GGMLTensorEntry> withExpertStacks(
            Map<String, GGMLTensorEntry> entries, Glm5NextConfiguration config, boolean device) {
        Map<String, GGMLTensorEntry> out = new HashMap<>(entries);
        Arena arena = Arena.ofAuto();
        for (GGMLTensorEntry entry : entries.values()) {
            String name = entry.name();
            if (!name.endsWith("_exps.weight")) {
                continue;
            }
            int[] shape = entry.shape();
            int experts = shape[2];
            long perExpert = byteSize(entry.ggmlType(), new int[] {shape[0], shape[1]});
            long header = device ? entry.memorySegment().byteSize() - perExpert * experts : 0;
            long fileOffset = gguf.getTensorDataOffset() + gguf.getTensorInfos().get(name).offset();
            String stem = name.substring(0, name.length() - ".weight".length());
            for (int c = 0, start = 0; start < experts; c++, start += config.expertChunkSize()) {
                int count = Math.min(config.expertChunkSize(), experts - start);
                long bytes = perExpert * count;
                MemorySegment segment;
                if (device) {
                    try {
                        segment =
                                fileChannel.map(
                                        FileChannel.MapMode.PRIVATE,
                                        fileOffset + perExpert * start - header,
                                        bytes + header,
                                        arena);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                    for (long i = 0; i < header; i++) {
                        segment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
                    }
                } else {
                    segment = entry.memorySegment().asSlice(perExpert * start, bytes);
                }
                String stack = stem + "." + c + ".weight";
                out.put(
                        stack,
                        new GGMLTensorEntry(
                                device ? segment : entry.mappedFile(),
                                stack,
                                entry.ggmlType(),
                                new int[] {shape[0], shape[1], count},
                                segment));
            }
        }
        return out;
    }

    // @formatter:off
    /** Every layer's tensors, the trunk then the MTP layers; slots a layer lacks stay null. */
    static <T> Glm5NextLayerWeights<T> layerWeights(
            Glm5NextConfiguration config,
            Map<String, GGMLTensorEntry> entries,
            T[] type,
            Function<GGMLTensorEntry, T> load) {
        int n = config.allLayers();
        Slots<T> s = new Slots<>(type, n);
        T[] hcAttnFn = s.next(),
                hcAttnBase = s.next(),
                hcAttnScale = s.next(),
                hcFfnFn = s.next(),
                hcFfnBase = s.next(),
                hcFfnScale = s.next(),
                attnNorm = s.next(),
                ffnNorm = s.next(),
                q = s.next(),
                k = s.next(),
                v = s.next(),
                convQ = s.next(),
                convK = s.next(),
                convV = s.next(),
                decayA = s.next(),
                decayB = s.next(),
                dtBias = s.next(),
                a = s.next(),
                beta = s.next(),
                gateA = s.next(),
                gateB = s.next(),
                outNorm = s.next(),
                kdaOut = s.next(),
                qA = s.next(),
                qANorm = s.next(),
                qB = s.next(),
                kvA = s.next(),
                kvANorm = s.next(),
                kB = s.next(),
                vB = s.next(),
                attnOut = s.next(),
                indexQ = s.next(),
                indexK = s.next(),
                indexKNorm = s.next(),
                indexKNormBias = s.next(),
                indexWeights = s.next(),
                poolApe = s.next(),
                poolGate = s.next(),
                gate = s.next(),
                up = s.next(),
                down = s.next(),
                router = s.next(),
                routerBias = s.next(),
                shGate = s.next(),
                shUp = s.next(),
                shDown = s.next(),
                embedNorm = s.next(),
                hiddenNorm = s.next(),
                inputProjection = s.next(),
                headNorm = s.next();
        List<List<T>> gateExps = new ArrayList<>(),
                upExps = new ArrayList<>(),
                downExps = new ArrayList<>();
        int chunks =
                (config.expertCount() + config.expertChunkSize() - 1) / config.expertChunkSize();

        for (int l = 0; l < n; l++) {
            String b = "blk." + l + ".";
            Function<String, T> get = name -> load.apply(required(entries, b + name));
            attnNorm[l] = get.apply("attn_norm.weight");
            ffnNorm[l] = get.apply("ffn_norm.weight");
            if (l < config.numberOfLayers()) {
                hcAttnFn[l] = get.apply("hc_attn_fn.weight");
                hcAttnBase[l] = get.apply("hc_attn_base.weight");
                hcAttnScale[l] = get.apply("hc_attn_scale.weight");
                hcFfnFn[l] = get.apply("hc_ffn_fn.weight");
                hcFfnBase[l] = get.apply("hc_ffn_base.weight");
                hcFfnScale[l] = get.apply("hc_ffn_scale.weight");
            } else {
                embedNorm[l] = get.apply("nextn.enorm.weight");
                hiddenNorm[l] = get.apply("nextn.hnorm.weight");
                inputProjection[l] = get.apply("nextn.eh_proj.weight");
                headNorm[l] = get.apply("nextn.shared_head_norm.weight");
            }
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
                qA[l] = get.apply("attn_q_a.weight");
                qANorm[l] = get.apply("attn_q_a_norm.weight");
                qB[l] = get.apply("attn_q_b.weight");
                kvA[l] = get.apply("attn_kv_a_mqa.weight");
                kvANorm[l] = get.apply("attn_kv_a_norm.weight");
                kB[l] = get.apply("attn_k_b.weight");
                vB[l] = get.apply("attn_v_b.weight");
                attnOut[l] = get.apply("attn_output.weight");
                indexQ[l] = get.apply("indexer.attn_q_b.weight");
                indexK[l] = get.apply("indexer.attn_k.weight");
                indexKNorm[l] = get.apply("indexer.k_norm.weight");
                indexKNormBias[l] = get.apply("indexer.k_norm.bias");
                indexWeights[l] = get.apply("indexer.proj.weight");
                poolApe[l] = get.apply("indexer_compressor_ape.weight");
                poolGate[l] = get.apply("indexer_compressor_gate.weight");
            }
            List<T> g = new ArrayList<>(), u = new ArrayList<>(), d = new ArrayList<>();
            if (config.isMoe(l)) {
                router[l] = get.apply("ffn_gate_inp.weight");
                routerBias[l] = get.apply("exp_probs_b.bias");
                for (int c = 0; c < chunks; c++) {
                    g.add(get.apply("ffn_gate_exps." + c + ".weight"));
                    u.add(get.apply("ffn_up_exps." + c + ".weight"));
                    d.add(get.apply("ffn_down_exps." + c + ".weight"));
                }
                if (config.sharedExperts() > 0) {
                    shGate[l] = get.apply("ffn_gate_shexp.weight");
                    shUp[l] = get.apply("ffn_up_shexp.weight");
                    shDown[l] = get.apply("ffn_down_shexp.weight");
                }
            } else {
                gate[l] = get.apply("ffn_gate.weight");
                up[l] = get.apply("ffn_up.weight");
                down[l] = get.apply("ffn_down.weight");
            }
            gateExps.add(g);
            upExps.add(u);
            downExps.add(d);
        }
        return new Glm5NextLayerWeights<>(
                new Glm5NextLayerWeights.Mhc<>(
                        hcAttnFn, hcAttnBase, hcAttnScale, hcFfnFn, hcFfnBase, hcFfnScale),
                attnNorm,
                ffnNorm,
                new Glm5NextLayerWeights.Kda<>(
                        q, k, v, convQ, convK, convV, decayA, decayB, dtBias, a, beta, gateA, gateB,
                        outNorm, kdaOut),
                new Glm5NextLayerWeights.SparseAttention<>(
                        qA,
                        qANorm,
                        qB,
                        kvA,
                        kvANorm,
                        kB,
                        vB,
                        attnOut,
                        indexQ,
                        indexK,
                        indexKNorm,
                        indexKNormBias,
                        indexWeights,
                        poolApe,
                        poolGate),
                new Glm5NextLayerWeights.FeedForward<>(
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
                        shDown),
                new Glm5NextLayerWeights.Nextn<>(embedNorm, hiddenNorm, inputProjection, headNorm));
    }

    // @formatter:on

    /** Hands out per-layer arrays of the weights' element type. */
    private record Slots<T>(T[] type, int layers) {
        T[] next() {
            return Arrays.copyOf(type, layers);
        }
    }

    static GGMLTensorEntry required(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "glm5next expects "
                            + name
                            + ", which this file does not carry");
        }
        return entry;
    }
}
