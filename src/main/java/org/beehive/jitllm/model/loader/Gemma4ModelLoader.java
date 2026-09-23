package org.beehive.jllm.model.loader;

import static org.beehive.jllm.model.loader.ModelLoader.*;
import static org.beehive.jllm.model.loader.ModelLoader.loadArrayOfTensors;
import static org.beehive.jllm.model.loader.ModelLoader.loadTensor;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.function.IntFunction;
import org.beehive.jllm.auxiliary.Pair;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jllm.format.DataTypeMapping;
import org.beehive.jllm.format.GGMLTensorEntry;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.format.GGUF;
import org.beehive.jllm.format.GGUF.GGUFTensorInfo;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.jllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jllm.model.format.Gemma4ChatFormat;
import org.beehive.jllm.model.gemma4.Gemma4;
import org.beehive.jllm.model.gemma4.Gemma4Configuration;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.ExecutionTarget;
import org.beehive.jllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.beehive.jllm.tokenizer.Vocabulary;

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

        return new Gemma4Configuration(
                getModelQuantization(metadata),
                (int) metadata.get("gemma4.embedding_length"),
                numberOfLayers,
                (int) metadata.get("gemma4.attention.head_count"),
                (int) metadata.get("gemma4.attention.head_count_kv"),
                (int) metadata.get("gemma4.attention.key_length_swa"),
                (int) metadata.get("gemma4.attention.key_length"),
                (int[]) metadata.get("gemma4.feed_forward_length"),
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
                (float) metadata.get("gemma4.final_logit_softcapping"));
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
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
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
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".proj.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight")),
                loadArrayOfTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),
                longIndexed(tensorEntries.get("per_layer_token_embd.weight")),
                loadTensor(tensorEntries.get("per_layer_model_proj.weight")),
                loadTensor(tensorEntries.get("per_layer_proj_norm.weight")),
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
        // What the trunk projections actually are, which is not what the output projection is: a
        // Q4_0 file holds Q4_0 projections and a Q4_K token_embd, so asking the output tensor gives
        // the plan the wrong representation to be admitted on.
        DataType projections = projectionType(tensorEntries, config);
        boolean retain = projections == DataType.Q4_0;
        DataType weightType =
                retain
                        ? DataType.Q4_0
                        : DataTypeMapping.materializedType(
                                outputWeight.ggmlType(), ExecutionTarget.GPU);
        RopeTables ropeTables = computeRopeTables(tensorEntries, config);

        return new Gemma4TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadProjections(retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadProjections(
                        retain, nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".proj.weight")),
                loadArrayOfTornadoTensors(
                        nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight")),
                loadArrayOfTornadoTensorsNullable(
                        nl, i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),
                longIndexed(
                        stripTornadoArrayHeader(tensorEntries.get("per_layer_token_embd.weight"))),
                loadTornadoTensor(tensorEntries.get("per_layer_model_proj.weight")),
                loadTornadoTensor(tensorEntries.get("per_layer_proj_norm.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                TornadoTensorLoader.fromFloats(ropeTables.realSwa),
                TornadoTensorLoader.fromFloats(ropeTables.imagSwa),
                TornadoTensorLoader.fromFloats(ropeTables.realFull),
                TornadoTensorLoader.fromFloats(ropeTables.imagFull),
                tensorEntries.containsKey("output.weight")
                        ? loadTornadoTensor(tensorEntries.get("output.weight"))
                        : loadTornadoTensor(tokenEmbeddings),
                weightType);
    }

    // @formatter:on

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
     */
    private RopeTables computeRopeTables(
            Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config) {
        Pair<float[], float[]> swa =
                precomputeFreqsCisWithFactors(
                        config.contextLengthModel(),
                        config.headDimSwa(),
                        config.ropeThetaSwa(),
                        null);

        // rope_freqs.weight is intentionally excluded from tensorEntries by
        // GGUF.loadTensorsStandard/
        // loadTensorsTornado (it isn't needed by most architectures), so read it directly here.
        float[] freqFactors = readFloat32TensorDirect("rope_freqs.weight");
        Pair<float[], float[]> full =
                precomputeFreqsCisWithFactors(
                        config.contextLengthModel(),
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
