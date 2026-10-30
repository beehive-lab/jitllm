package org.beehive.jitllm.model.loader;

import static org.beehive.jitllm.model.loader.ModelLoader.*;
import static org.beehive.jitllm.model.loader.ModelLoader.loadArrayOfTensors;
import static org.beehive.jitllm.model.loader.ModelLoader.loadTensor;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.format.GGUF.GGUFTensorInfo;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.format.Gemma4ChatFormat;
import org.beehive.jitllm.model.gemma4.Gemma4;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.beehive.jitllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;

/**
 * Loader for Gemma 4 models (e.g. Gemma-4-E2B-It).
 *
 * <p>Gemma 4 needs two distinct precomputed RoPE tables (sliding-window vs. full/global attention
 * layers use different bases and head dimensions, and full-attention layers additionally apply a
 * per-dimension frequency scaling stored in the {@code rope_freqs} tensor), so RoPE frequencies are
 * computed directly here -- where the tensor entries are available -- rather than through the
 * generic {@link #precomputeRopeFrequencies} hook.
 */
public class Gemma4ModelLoader extends AbstractModelLoader<Gemma4, Gemma4Configuration> {

    public Gemma4ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Gemma4Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Gemma4Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("gemma4.context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;
        int numberOfLayers = (int) metadata.get("gemma4.block_count");
        // One count for every layer, or one per layer (the 31B: 16 sliding-window, 4 global).
        Object kvHeads = metadata.get("gemma4.attention.head_count_kv");
        int[] kvHeadsPerLayer = kvHeads instanceof int[] perLayer ? perLayer : null;
        // The same for the feed-forward width: E2B gives one per layer, the 31B one for all.
        Object ffn = metadata.get("gemma4.feed_forward_length");
        int[] feedForwardLength;
        if (ffn instanceof int[] perLayer) {
            feedForwardLength = perLayer;
        } else {
            feedForwardLength = new int[numberOfLayers];
            java.util.Arrays.fill(feedForwardLength, (int) ffn);
        }
        int maxKvHeads =
                kvHeadsPerLayer == null
                        ? (int) kvHeads
                        : java.util.Arrays.stream(kvHeadsPerLayer).max().orElse(1);

        return new Gemma4Configuration(
                getModelQuantization(metadata),
                (int) metadata.get("gemma4.embedding_length"),
                numberOfLayers,
                (int) metadata.get("gemma4.attention.head_count"),
                maxKvHeads,
                (int) metadata.get("gemma4.attention.key_length_swa"),
                (int) metadata.get("gemma4.attention.key_length"),
                feedForwardLength,
                (boolean[]) metadata.get("gemma4.attention.sliding_window_pattern"),
                (int) metadata.get("gemma4.attention.sliding_window"),
                (int) metadata.get("gemma4.attention.shared_kv_layers"),
                (int) metadata.get("gemma4.embedding_length_per_layer_input"),
                vocabulary.size(),
                modelContextLength,
                finalContextLength,
                (float) metadata.get("gemma4.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("gemma4.rope.freq_base"),
                (float) metadata.get("gemma4.rope.freq_base_swa"),
                (float) metadata.get("gemma4.final_logit_softcapping"),
                kvHeadsPerLayer);
    }

    /**
     * Wraps a tensor too large for int indexing so the weight sets can hold it without holding a
     * format type. Gemma-4's per-layer token embedding table is the only such tensor today.
     */
    private static LongIndexedTensor longIndexed(GGMLTensorEntry entry) {
        return new LongIndexedTensor(
                entry.memorySegment(), DataTypeMapping.sourceType(entry.ggmlType()));
    }

    // @formatter:on

    /**
     * Gemma4 needs two RoPE tables computed with tensor data (rope_freqs); see {@link #ropeTables}.
     */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Gemma4Configuration config) {
        return null;
    }

    @Override
    protected Gemma4 createModel(Gemma4Configuration config, Tokenizer tokenizer, Weights weights) {
        return new Gemma4(
                config, tokenizer, weights, new Gemma4ChatFormat((Gemma4Tokenizer) tokenizer));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Gemma4Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        final boolean ple = config.hasPerLayerEmbeddings();
        RopeTables ropeTables = computeRopeTables(tensorEntries, config);

        return new Gemma4StandardWeights(
                loadTensor(tokenEmbeddings),
                tensorEntries.containsKey("output.weight")
                        ? loadTensor(tensorEntries.get("output.weight"))
                        : loadTensor(tokenEmbeddings),
                loadTensor(tensorEntries.get("output_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                // Absent on the 31B's global layers, whose values are their keys.
                loadOptionalTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                // The per-layer input embeddings and their projections, where the model has them.
                ple
                        ? loadArrayOfTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight"))
                        : new FloatTensor[nl],
                ple
                        ? loadArrayOfTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".proj.weight"))
                        : new FloatTensor[nl],
                ple
                        ? loadArrayOfTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight"))
                        : new FloatTensor[nl],
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),
                ple ? longIndexed(tensorEntries.get("per_layer_token_embd.weight")) : null,
                ple ? loadTensor(tensorEntries.get("per_layer_model_proj.weight")) : null,
                ple ? loadTensor(tensorEntries.get("per_layer_proj_norm.weight")) : null,
                new ArrayFloatTensor(ropeTables.realSwa),
                new ArrayFloatTensor(ropeTables.imagSwa),
                new ArrayFloatTensor(ropeTables.realFull),
                new ArrayFloatTensor(ropeTables.imagFull),
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Gemma4Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        final boolean ple = config.hasPerLayerEmbeddings();
        // On CUDA and Metal every tensor stays as the file stores it: the quantized layers read
        // each
        // one by its own type, so the projections need not agree, and the model reports its
        // projections' type. Elsewhere they must agree, and a Q4_0 trunk is kept while anything
        // else
        // is converted to Q8_0 as before. The projections, not the output projection, decide: a
        // Q4_0 file holds Q4_0 projections and a Q4_K token_embd.
        boolean keep = TornadoPlanProvider.retainsNative(TornadoDevices.current().backend());
        DataType projections = keep ? null : projectionType(tensorEntries, config);
        boolean retain = keep || projections == DataType.Q4_0;
        RopeTables ropeTables = computeRopeTables(tensorEntries, config);

        TornadoTensor[] wq =
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight"));
        TornadoTensor[] wk =
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight"));
        // Absent on the 31B's global layers, whose values are their keys.
        TornadoTensor[] wv =
                loadOptionalProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight"));
        TornadoTensor[] wo =
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight"));
        TornadoTensor[] w1 =
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight"));
        TornadoTensor[] w3 =
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight"));
        TornadoTensor[] w2 =
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight"));
        DataType weightType;
        if (keep) {
            weightType = keptType(wq, wk, wo, w1, w3, w2);
        } else {
            weightType =
                    retain
                            ? DataType.Q4_0
                            : DataTypeMapping.materializedType(
                                    outputWeight.ggmlType(), ExecutionTarget.GPU);
        }

        Gemma4TornadoWeights weights =
                new Gemma4TornadoWeights(
                        keep
                                ? loadTornadoTensorNative(tokenEmbeddings)
                                : loadTornadoTensor(tokenEmbeddings),
                        loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                        wq,
                        wk,
                        wv,
                        wo,
                        loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                        loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                        loadArrayOfTornadoTensors(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                        loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                        w1,
                        w3,
                        w2,
                        loadArrayOfTornadoTensors(
                                nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                        ple
                                ? loadProjections(
                                        keep,
                                        nl,
                                        i -> tensorEntries.get("blk." + i + ".inp_gate.weight"))
                                : new TornadoTensor[nl],
                        ple
                                ? loadProjections(
                                        keep,
                                        nl,
                                        i -> tensorEntries.get("blk." + i + ".proj.weight"))
                                : new TornadoTensor[nl],
                        ple
                                ? loadArrayOfTornadoTensors(
                                        nl,
                                        i -> tensorEntries.get("blk." + i + ".post_norm.weight"))
                                : new TornadoTensor[nl],
                        loadArrayOfTornadoTensorsNullable(
                                nl,
                                i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),
                        ple
                                ? longIndexed(
                                        stripTornadoArrayHeader(
                                                tensorEntries.get("per_layer_token_embd.weight")))
                                : null,
                        ple
                                ? loadTornadoTensor(
                                        tensorEntries.get("per_layer_model_proj.weight"))
                                : null,
                        ple
                                ? loadTornadoTensor(tensorEntries.get("per_layer_proj_norm.weight"))
                                : null,
                        loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                        TornadoTensorLoader.fromFloats(ropeTables.realSwa),
                        TornadoTensorLoader.fromFloats(ropeTables.imagSwa),
                        TornadoTensorLoader.fromFloats(ropeTables.realFull),
                        TornadoTensorLoader.fromFloats(ropeTables.imagFull),
                        loadOutput(
                                keep,
                                retain,
                                tensorEntries.containsKey("output.weight")
                                        ? tensorEntries.get("output.weight")
                                        : tokenEmbeddings),
                        weightType);
        return weights;
    }

    // @formatter:on

    /**
     * The output projection: as the file stores it where every tensor is kept ({@code keep}). Next
     * to Q4_0 projections elsewhere it stays Q4_K, read by the packed-integer Q4_K kernel, rather
     * than becoming a Q8_0 copy of nearly twice the bytes.
     */
    private static TornadoTensor loadOutput(boolean keep, boolean retain, GGMLTensorEntry entry) {
        if (keep) {
            return loadTornadoTensorNative(entry);
        }
        if (retain) {
            return loadTornadoTensorRetainingQ4_K(entry);
        }
        return loadTornadoTensor(entry);
    }

    /**
     * Tensor entries produced by {@link GGUF#loadTensorsTornado} prefix every {@code
     * memorySegment()} with a 16-byte TornadoVM array header (so the data can be wrapped as a
     * native array without copying) -- but {@code per_layer_token_embd} is kept as a raw entry and
     * addressed with byte-offset arithmetic that assumes the segment starts at the tensor's actual
     * data, mirroring the CPU path's {@link ModelLoader#copyEmbeddingRow} over a {@link
     * GGUF#loadTensorsStandard}-produced entry, which has no such header. Slice past the header so
     * both code paths see the same layout.
     *
     * <p>How wide that header is is the backend's knowledge, not this loader's.
     */
    private static GGMLTensorEntry stripTornadoArrayHeader(GGMLTensorEntry entry) {
        return new GGMLTensorEntry(
                entry.mappedFile(),
                entry.name(),
                entry.ggmlType(),
                entry.shape(),
                TornadoTensorLoader.withoutArrayHeader(entry.memorySegment()));
    }

    /**
     * Like {@link ModelLoader#loadArrayOfTornadoTensors}, but tolerates missing entries (Gemma4's
     * optional per-layer output scale).
     */
    private static TornadoTensor[] loadArrayOfTornadoTensorsNullable(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            array[i] = (entry == null) ? null : loadTornadoTensor(entry);
        }
        return array;
    }

    /** {@link #loadProjections} where a layer may not carry the tensor: null at that layer. */
    private static TornadoTensor[] loadOptionalProjections(
            boolean retain, int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            if (entry != null) {
                array[i] = retain ? loadTornadoTensorNative(entry) : loadTornadoTensor(entry);
            }
        }
        return array;
    }

    /** Host tensors where a layer may not carry the tensor: null at that layer. */
    private static FloatTensor[] loadOptionalTensors(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            array[i] = entry == null ? null : loadTensor(entry);
        }
        return array;
    }

    /**
     * Loads a per-layer projection either as the file holds it or materialized, by one decision
     * taken for the whole trunk.
     */
    private static TornadoTensor[] loadProjections(
            boolean retain, int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        return retain
                ? loadArrayOfTornadoTensorsNative(size, getTensorEntry)
                : loadArrayOfTornadoTensors(size, getTensorEntry);
    }

    /**
     * The representation this file's trunk projections agree on.
     *
     * <p>{@code ffn_down} is deliberately excluded from the agreement. A real Q4_0 quantizer leaves
     * it Q4_1 on the first few blocks — four of thirty-five here — and those tasks read it through
     * a kernel chosen from that tensor's own type, so a disagreement there is expected rather than
     * a malformed file. Everything else must agree, because the plan is admitted on one
     * representation and the two block layouts are 18 and 20 bytes: reading one as the other yields
     * weights of plausible magnitude and fluent, wrong text.
     */
    private static DataType projectionType(
            Map<String, GGMLTensorEntry> entries, Gemma4Configuration config) {
        DataType agreed = null;
        String agreedName = null;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            for (String kind :
                    new String[] {
                        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up"
                    }) {
                String name = "blk." + l + "." + kind + ".weight";
                GGMLTensorEntry entry = entries.get(name);
                if (entry == null) {
                    continue;
                }
                DataType type = DataTypeMapping.sourceType(entry.ggmlType());
                if (agreed == null) {
                    agreed = type;
                    agreedName = name;
                } else if (agreed != type) {
                    throw new ModelLoadException(
                            DiagnosticCode.MODEL_MALFORMED.prefix()
                                    + "gemma4 projections disagree about their representation: "
                                    + agreedName
                                    + " is "
                                    + agreed
                                    + " and "
                                    + name
                                    + " is "
                                    + type
                                    + ". They are read by tasks with different block layouts, and"
                                    + " there is no model-wide representation to admit a plan on."
                                    + " Neither is converted to the other.");
                }
            }
        }
        return agreed;
    }

    private record RopeTables(
            float[] realSwa, float[] imagSwa, float[] realFull, float[] imagFull) {}

    /**
     * Computes the two RoPE frequency tables Gemma4 needs.
     *
     * <p>Sliding-window layers use {@code rope_theta_swa} with {@code headDimSwa} and no extra
     * scaling. Full/global-attention layers use {@code rope_theta} with {@code headDimFull},
     * additionally dividing each rotation angle by the corresponding entry of the (single, shared)
     * {@code rope_freqs} tensor -- this is how the GGUF encodes "partial RoPE" (entries are 1.0 for
     * the active low-frequency dimensions and effectively infinite for the inactive ones, which
     * zeroes out their rotation).
     *
     * <p>Sized to the configured context rather than the model's: no position reaches past the
     * key/value capacity, and at the model's 131072 positions the four tables are 400 MB of host
     * memory and as much again on the device per graph that binds them.
     */
    private RopeTables computeRopeTables(
            Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config) {
        Pair<float[], float[]> swa =
                precomputeFreqsCisWithFactors(
                        config.contextLength(), config.headDimSwa(), config.ropeThetaSwa(), null);

        // rope_freqs.weight is intentionally excluded from tensorEntries by
        // GGUF.loadTensorsStandard/
        // loadTensorsTornado (it isn't needed by most architectures), so read it directly here.
        float[] freqFactors = readFloat32TensorDirect("rope_freqs.weight");
        Pair<float[], float[]> full =
                precomputeFreqsCisWithFactors(
                        config.contextLength(),
                        config.headDimFull(),
                        config.ropeTheta(),
                        freqFactors);

        return new RopeTables(swa.first(), swa.second(), full.first(), full.second());
    }

    /**
     * Reads a small F32 tensor's raw data directly from the GGUF file, bypassing the {@code
     * tensorEntries} map.
     */
    private float[] readFloat32TensorDirect(String tensorName) {
        GGUFTensorInfo info = gguf.getTensorInfos().get(tensorName);
        if (info == null) {
            return null;
        }
        if (info.ggmlType() != GGMLType.F32) {
            throw new UnsupportedOperationException(
                    "Expected F32 tensor for " + tensorName + ", got " + info.ggmlType());
        }

        int numberOfElements = 1;
        for (int dimension : info.dimensions()) {
            numberOfElements *= dimension;
        }

        long byteOffset = gguf.getTensorDataOffset() + info.offset();
        ByteBuffer buffer =
                ByteBuffer.allocate(numberOfElements * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try {
            while (buffer.hasRemaining()) {
                if (gguf.getFileChannel().read(buffer, byteOffset + buffer.position()) < 0) {
                    throw new EOFException("Unexpected end of file while reading " + tensorName);
                }
            }
        } catch (IOException e) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "Failed to read "
                            + tensorName
                            + " from GGUF file",
                    e);
        }
        buffer.flip();

        float[] result = new float[numberOfElements];
        buffer.asFloatBuffer().get(result);
        return result;
    }

    /**
     * Like {@link RopeFrequencies#precomputeFreqsCis}, but allows dividing each pair's frequency by
     * a per-dimension scaling factor (NeoX-style RoPE).
     */
    private static Pair<float[], float[]> precomputeFreqsCisWithFactors(
            int contextLength, int headSize, double theta, float[] freqFactors) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                int pairIndex = i / 2;
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (freqFactors != null) {
                    freq = freq / freqFactors[pairIndex];
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}
