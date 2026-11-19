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
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.DeepSeek4StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek4TornadoWeights;
import org.beehive.jitllm.model.deepseek4.DeepSeek4;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.model.format.DeepSeekV4ChatFormat;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.DeepSeekV3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loads a llama.cpp {@code deepseek4} GGUF (DeepSeek-V4), as llama.cpp's converter writes it, or
 * {@code scripts/deepseek4/convert.py} for a slice of the layers.
 *
 * <p>{@code attention.compress_ratios} gives each block's ratio (0, 4 or 128); the first {@code
 * hash_layer_count} blocks route by token through {@code ffn_gate_tid2eid}. The MTP layers, {@code
 * nextn_predict_layers} of them, are the last blocks. Each layer's routed experts are one tensor
 * per projection in the file; they are loaded as stacks of {@link
 * DeepSeek4Configuration#expertChunkSize} experts, so that no stack reaches 2³¹ bytes.
 */
public class DeepSeek4ModelLoader extends AbstractModelLoader<DeepSeek4, DeepSeek4Configuration> {

    /** The {@code general.architecture} llama.cpp gives DeepSeek-V4, and its key prefix. */
    public static final String ARCHITECTURE = "deepseek4";

    /** The declared 1M positions would not fit a host session; the same default as glm5next. */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    /** llama.cpp's {@code expert_gating_func} for {@code sqrtsoftplus}, the only one V4 uses. */
    private static final int SQRT_SOFTPLUS = 4;

    /** The most bytes one loaded expert stack may take: a device array is int-indexed. */
    private static final long MAX_STACK_BYTES = Integer.MAX_VALUE - 4096L;

    /** Lowers {@link #MAX_STACK_BYTES}, so small models can be loaded as several stacks too. */
    public static final String STACK_BYTES_PROPERTY = "jitllm.deepseek4.expertStackBytes";

    public DeepSeek4ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokens(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new DeepSeekV3Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected DeepSeek4Configuration createConfiguration(Map<String, Object> metadata) {
        String p = ARCHITECTURE + ".";
        int gating = (int) metadata.getOrDefault(p + "expert_gating_func", SQRT_SOFTPLUS);
        if (gating != SQRT_SOFTPLUS) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "deepseek4 expects sqrtsoftplus expert scoring, not gating function "
                            + gating);
        }
        int modelContextLength = (int) metadata.get(p + "context_length");
        int finalContextLength =
                contextLength > 0
                        ? Math.min(contextLength, modelContextLength)
                        : Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
        int nextn = (int) metadata.getOrDefault(p + "nextn_predict_layers", 0);
        int blocks = (int) metadata.get(p + "block_count");
        int[] ratios =
                Arrays.copyOf(
                        perLayer(metadata.get(p + "attention.compress_ratios"), blocks), blocks);
        for (int r : ratios) {
            if (r != 0
                    && r != DeepSeek4Configuration.INDEXED_RATIO
                    && r != DeepSeek4Configuration.DENSE_RATIO) {
                throw new ModelLoadException(
                        DiagnosticCode.MODEL_MALFORMED.prefix()
                                + "deepseek4 supports compression ratios 0, 4 and 128, not "
                                + r);
            }
        }
        int expertCount = (int) metadata.get(p + "expert_count");
        return new DeepSeek4Configuration(
                quantization(metadata),
                (int) metadata.get(p + "embedding_length"),
                blocks - nextn,
                nextn,
                this.vocabulary.size(),
                modelContextLength,
                finalContextLength,
                (float) metadata.get(p + "attention.layer_norm_rms_epsilon"),
                (int) metadata.get(p + "hyper_connection.count"),
                (int) metadata.get(p + "hyper_connection.sinkhorn_iterations"),
                (float) metadata.get(p + "hyper_connection.epsilon"),
                (int) metadata.get(p + "attention.head_count"),
                (int) metadata.get(p + "attention.key_length"),
                (int) metadata.get(p + "rope.dimension_count"),
                (int) metadata.get(p + "attention.q_lora_rank"),
                (int) metadata.get(p + "attention.output_group_count"),
                (int) metadata.get(p + "attention.output_lora_rank"),
                (int) metadata.get(p + "attention.sliding_window"),
                ratios,
                (float) metadata.get(p + "rope.freq_base"),
                (float) metadata.get(p + "attention.compress_rope_freq_base"),
                (float) metadata.getOrDefault(p + "rope.scaling.factor", 1.0f),
                (int) metadata.getOrDefault(p + "rope.scaling.original_context_length", 0),
                (float) metadata.getOrDefault(p + "rope.scaling.yarn_beta_fast", 32.0f),
                (float) metadata.getOrDefault(p + "rope.scaling.yarn_beta_slow", 1.0f),
                (int) metadata.get(p + "attention.indexer.head_count"),
                (int) metadata.get(p + "attention.indexer.key_length"),
                (int) metadata.get(p + "attention.indexer.top_k"),
                expertCount,
                (int) metadata.get(p + "expert_used_count"),
                perLayer(metadata.get(p + "expert_feed_forward_length"), 1)[0],
                (int) metadata.get(p + "expert_shared_count"),
                expertChunkSize(expertCount),
                (float) metadata.getOrDefault(p + "expert_weights_scale", 1.0f),
                (boolean) metadata.getOrDefault(p + "expert_weights_norm", false),
                swigluLimit(metadata.get(p + "swiglu_clamp_exp")),
                (int) metadata.getOrDefault(p + "hash_layer_count", 0));
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

    /** The rotations are computed from the configuration: no tables. */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(DeepSeek4Configuration config) {
        return new Pair<>(new float[0], new float[0]);
    }

    @Override
    protected DeepSeek4 createModel(
            DeepSeek4Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        int bos = (int) metadata.getOrDefault("tokenizer.ggml.bos_token_id", 0);
        int eos = (int) metadata.getOrDefault("tokenizer.ggml.eos_token_id", 1);
        return new DeepSeek4(
                config,
                tokenizer,
                weights,
                new DeepSeekV4ChatFormat((DeepSeekV3Tokenizer) tokenizer, vocabulary, bos, eos));
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DeepSeek4Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        Function<String, FloatTensor> top =
                name -> ModelLoader.loadTensor(required(tensorEntries, name));
        return new DeepSeek4StandardWeights(
                ModelLoader.loadTensor(tokenEmbeddings),
                layerWeights(
                        config,
                        withExpertStacks(tensorEntries, config, false),
                        new FloatTensor[0],
                        ModelLoader::loadTensor),
                top.apply("output_hc_fn.weight"),
                top.apply("output_hc_base.weight"),
                top.apply("output_hc_scale.weight"),
                top.apply("output_norm.weight"),
                ModelLoader.loadTensor(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    /**
     * Device weights, every tensor as the file holds it except the BF16 matrices of the layers —
     * the compressors, the indexer's head weights and the router — which become F32, the form the
     * kernels read them in.
     */
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            DeepSeek4Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        Function<String, TornadoTensor> top =
                name -> ModelLoader.loadTornadoTensorNative(required(tensorEntries, name));
        return new DeepSeek4TornadoWeights(
                ModelLoader.loadTornadoTensorNative(tokenEmbeddings),
                layerWeights(
                        config,
                        withExpertStacks(tensorEntries, config, true),
                        new TornadoTensor[0],
                        DeepSeek4ModelLoader::deviceTensor),
                top.apply("output_hc_fn.weight"),
                top.apply("output_hc_base.weight"),
                top.apply("output_hc_scale.weight"),
                top.apply("output_norm.weight"),
                ModelLoader.loadTornadoTensorNative(outputWeight),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    /** A layer tensor on the device: BF16 widened to F32, anything else as the file holds it. */
    private static TornadoTensor deviceTensor(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.BF16) {
            return ModelLoader.loadTornadoTensorNative(entry);
        }
        FloatTensor host =
                ModelLoader.loadTensor(
                        new GGMLTensorEntry(
                                entry.mappedFile(),
                                entry.name(),
                                entry.ggmlType(),
                                entry.shape(),
                                TornadoTensorLoader.withoutArrayHeader(entry.memorySegment())));
        float[] values = new float[host.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = host.getFloat(i);
        }
        return TornadoTensorLoader.fromFloats(values);
    }

    /**
     * The entries with each layer's expert tensors also split into stacks of {@link
     * DeepSeek4Configuration#expertChunkSize} experts, named {@code ffn_*_exps.<stack>.weight}. A
     * stack is a slice of the mapped file; for the device it is mapped again with room for the
     * native array header in front, as {@link GGUF#loadTensorsTornado} maps a whole tensor.
     */
    private Map<String, GGMLTensorEntry> withExpertStacks(
            Map<String, GGMLTensorEntry> entries, DeepSeek4Configuration config, boolean device) {
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
    static <T> DeepSeek4LayerWeights<T> layerWeights(
            DeepSeek4Configuration config,
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
                qA = s.next(),
                qANorm = s.next(),
                qB = s.next(),
                kv = s.next(),
                kvNorm = s.next(),
                outputA = s.next(),
                outputB = s.next(),
                sinks = s.next(),
                compKv = s.next(),
                compGate = s.next(),
                compApe = s.next(),
                compNorm = s.next(),
                indexQ = s.next(),
                indexWeights = s.next(),
                indexKv = s.next(),
                indexGate = s.next(),
                indexApe = s.next(),
                indexNorm = s.next(),
                router = s.next(),
                routerBias = s.next(),
                shGate = s.next(),
                shUp = s.next(),
                shDown = s.next(),
                embedNorm = s.next(),
                hiddenNorm = s.next(),
                inputProjection = s.next(),
                headNorm = s.next();
        int[][] tokenExperts = new int[n][];
        List<List<T>> gateExps = new ArrayList<>(),
                upExps = new ArrayList<>(),
                downExps = new ArrayList<>();
        int chunks =
                (config.expertCount() + config.expertChunkSize() - 1) / config.expertChunkSize();

        for (int l = 0; l < n; l++) {
            String b = "blk." + l + ".";
            Function<String, T> get = name -> load.apply(required(entries, b + name));
            hcAttnFn[l] = get.apply("hc_attn_fn.weight");
            hcAttnBase[l] = get.apply("hc_attn_base.weight");
            hcAttnScale[l] = get.apply("hc_attn_scale.weight");
            hcFfnFn[l] = get.apply("hc_ffn_fn.weight");
            hcFfnBase[l] = get.apply("hc_ffn_base.weight");
            hcFfnScale[l] = get.apply("hc_ffn_scale.weight");
            attnNorm[l] = get.apply("attn_norm.weight");
            ffnNorm[l] = get.apply("ffn_norm.weight");
            qA[l] = get.apply("attn_q_a.weight");
            qANorm[l] = get.apply("attn_q_a_norm.weight");
            qB[l] = get.apply("attn_q_b.weight");
            kv[l] = get.apply("attn_kv.weight");
            kvNorm[l] = get.apply("attn_kv_a_norm.weight");
            outputA[l] = get.apply("attn_output_a.weight");
            outputB[l] = get.apply("attn_output_b.weight");
            sinks[l] = get.apply("attn_sinks.weight");
            if (config.ratio(l) != 0) {
                compKv[l] = get.apply("attn_compressor_kv.weight");
                compGate[l] = get.apply("attn_compressor_gate.weight");
                compApe[l] = get.apply("attn_compressor_ape.weight");
                compNorm[l] = get.apply("attn_compressor_norm.weight");
            }
            if (config.hasIndexer(l)) {
                indexQ[l] = get.apply("indexer.attn_q_b.weight");
                indexWeights[l] = get.apply("indexer.proj.weight");
                indexKv[l] = get.apply("indexer_compressor_kv.weight");
                indexGate[l] = get.apply("indexer_compressor_gate.weight");
                indexApe[l] = get.apply("indexer_compressor_ape.weight");
                indexNorm[l] = get.apply("indexer_compressor_norm.weight");
            }
            router[l] = get.apply("ffn_gate_inp.weight");
            if (config.isHashRouted(l)) {
                tokenExperts[l] = ints(required(entries, b + "ffn_gate_tid2eid.weight"));
            } else {
                routerBias[l] = get.apply("exp_probs_b.bias");
            }
            List<T> g = new ArrayList<>(), u = new ArrayList<>(), d = new ArrayList<>();
            for (int c = 0; c < chunks; c++) {
                g.add(get.apply("ffn_gate_exps." + c + ".weight"));
                u.add(get.apply("ffn_up_exps." + c + ".weight"));
                d.add(get.apply("ffn_down_exps." + c + ".weight"));
            }
            gateExps.add(g);
            upExps.add(u);
            downExps.add(d);
            shGate[l] = get.apply("ffn_gate_shexp.weight");
            shUp[l] = get.apply("ffn_up_shexp.weight");
            shDown[l] = get.apply("ffn_down_shexp.weight");
            if (l >= config.numberOfLayers()) {
                embedNorm[l] = get.apply("nextn.enorm.weight");
                hiddenNorm[l] = get.apply("nextn.hnorm.weight");
                inputProjection[l] = get.apply("nextn.eh_proj.weight");
                if (entries.containsKey(b + "nextn.shared_head_norm.weight")) {
                    headNorm[l] = get.apply("nextn.shared_head_norm.weight");
                }
            }
        }
        return new DeepSeek4LayerWeights<>(
                new DeepSeek4LayerWeights.Mhc<>(
                        hcAttnFn, hcAttnBase, hcAttnScale, hcFfnFn, hcFfnBase, hcFfnScale),
                attnNorm,
                ffnNorm,
                new DeepSeek4LayerWeights.Attention<>(
                        qA, qANorm, qB, kv, kvNorm, outputA, outputB, sinks),
                new DeepSeek4LayerWeights.Compressor<>(compKv, compGate, compApe, compNorm),
                new DeepSeek4LayerWeights.Indexer<>(
                        indexQ, indexWeights, indexKv, indexGate, indexApe, indexNorm),
                new DeepSeek4LayerWeights.FeedForward<>(
                        router,
                        routerBias,
                        tokenExperts,
                        gateExps,
                        upExps,
                        downExps,
                        shGate,
                        shUp,
                        shDown),
                new DeepSeek4LayerWeights.Nextn<>(
                        embedNorm, hiddenNorm, inputProjection, headNorm));
    }

    // @formatter:on

    /** An I32 tensor's values, row-major: a hash-routed layer's token-to-expert table. */
    private static int[] ints(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.I32) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + entry.name()
                            + " should be I32, not "
                            + entry.ggmlType());
        }
        MemorySegment data = entry.memorySegment();
        int count = 1;
        for (int d : entry.shape()) {
            count *= d;
        }
        long start = data.byteSize() - (long) count * Integer.BYTES;
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            values[i] = data.get(ValueLayout.JAVA_INT_UNALIGNED, start + (long) i * Integer.BYTES);
        }
        return values;
    }

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
                            + "deepseek4 expects "
                            + name
                            + ", which this file does not carry");
        }
        return entry;
    }
}
