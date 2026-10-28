package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.WeightFootprint;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/**
 * The memory preflight predicts a Qwen3 file at the size the loader gives it: the file's own bytes
 * where the loader keeps its tensors (CUDA, Metal), and Q8_0 where it converts them, because both
 * ask {@code TornadoPlanProvider.retainsNative}.
 *
 * <p>The file is synthetic and holds descriptors only, which is all the prediction reads: two
 * layers of 256 x 256 projections in the Q4_K_M layout, {@code attn_v} and {@code ffn_down} Q6_K in
 * the first layer and Q4_K in the second, and a Q6_K {@code token_embd}.
 */
public class Qwen3RetentionFootprintTest {

    private static final int LAYERS = 2;
    private static final int DIM = 256;
    private static final long ELEMENTS = (long) DIM * DIM;
    private static final long Q4_K_BYTES = ELEMENTS / 256 * 144;
    private static final long Q6_K_BYTES = ELEMENTS / 256 * 210;
    private static final long Q8_0_BYTES = ELEMENTS / 32 * 34;
    private static final long Q4_0_BYTES = ELEMENTS / 32 * 18;
    private static final long Q4_1_BYTES = ELEMENTS / 32 * 20;
    private static final long Q5_K_BYTES = ELEMENTS / 256 * 176;
    private static final long NORM_BYTES = (long) DIM * Float.BYTES;

    private static final String[] Q4_K = {"attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"};

    private static final String[] NORMS = {"attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"};

    @Rule public final TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void aKeptFileIsPredictedAtItsOwnSize() throws Exception {
        long perLayer = 10 * Q4_K_BYTES + 2 * Q6_K_BYTES + 2 * Q4_K_BYTES + 8 * NORM_BYTES;
        Path file = qwen3Q4_K_MFile();
        for (BackendId backend : new BackendId[] {BackendId.CUDA, BackendId.METAL}) {
            WeightFootprint footprint = footprint(file, backend);
            assertEquals(backend + " per layer", perLayer, footprint.perLayerBytes());
            assertEquals(backend + " token_embd", Q6_K_BYTES, footprint.globalBytes());
        }
    }

    @Test
    public void aConvertedFileIsPredictedAtQ8_0() throws Exception {
        WeightFootprint footprint = footprint(qwen3Q4_K_MFile(), BackendId.OPENCL);
        assertEquals(14 * Q8_0_BYTES + 8 * NORM_BYTES, footprint.perLayerBytes());
        assertEquals(Q8_0_BYTES, footprint.globalBytes());
    }

    /**
     * A Q4_0 file with Q4_1 in one layer's {@code ffn_down}, as unsloth's Qwen3-0.6B-Q4_0 has, and
     * a Q5_K {@code attn_v}: each kept at its own size.
     */
    @Test
    public void aMixedFileIsPredictedAtItsOwnSize() throws Exception {
        Map<String, GGMLType> tensors = new LinkedHashMap<>();
        long expected = 0;
        for (int layer = 0; layer < LAYERS; layer++) {
            String blk = "blk." + layer + ".";
            for (String kind :
                    new String[] {"attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"}) {
                tensors.put(blk + kind + ".weight", GGMLType.Q4_0);
                expected += Q4_0_BYTES;
            }
            tensors.put(blk + "attn_v.weight", GGMLType.Q5_K);
            tensors.put(blk + "ffn_down.weight", layer == 0 ? GGMLType.Q4_1 : GGMLType.Q8_0);
            expected += Q5_K_BYTES + (layer == 0 ? Q4_1_BYTES : Q8_0_BYTES);
            for (String norm : NORMS) {
                tensors.put(blk + norm + ".weight", GGMLType.F32);
                expected += NORM_BYTES;
            }
        }
        tensors.put("token_embd.weight", GGMLType.Q6_K);
        Path file = folder.newFile("qwen3-mixed-descriptors.gguf").toPath();
        Files.write(file, headerOnlyGguf(tensors));
        WeightFootprint footprint = footprint(file, BackendId.CUDA);
        assertEquals(expected, footprint.perLayerBytes());
        assertEquals(Q6_K_BYTES, footprint.globalBytes());
    }

    private static WeightFootprint footprint(Path file, BackendId backend) throws Exception {
        return ModelLoader.weightFootprint(
                file,
                sourceTypes ->
                        TornadoPlanRegistry.deviceRetention(
                                ArchitectureId.of("qwen3"), sourceTypes, LAYERS, backend));
    }

    private Path qwen3Q4_K_MFile() throws Exception {
        Map<String, GGMLType> tensors = new LinkedHashMap<>();
        for (int layer = 0; layer < LAYERS; layer++) {
            String blk = "blk." + layer + ".";
            for (String kind : Q4_K) {
                tensors.put(blk + kind + ".weight", GGMLType.Q4_K);
            }
            GGMLType mixed = layer == 0 ? GGMLType.Q6_K : GGMLType.Q4_K;
            tensors.put(blk + "attn_v.weight", mixed);
            tensors.put(blk + "ffn_down.weight", mixed);
            for (String norm : NORMS) {
                tensors.put(blk + norm + ".weight", GGMLType.F32);
            }
        }
        tensors.put("token_embd.weight", GGMLType.Q6_K);
        Path file = folder.newFile("qwen3-q4_k_m-descriptors.gguf").toPath();
        Files.write(file, headerOnlyGguf(tensors));
        return file;
    }

    /** A version 3 GGUF header with no metadata and these tensors' descriptors, and no data. */
    private static byte[] headerOnlyGguf(Map<String, GGMLType> tensors) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer header = little(4 + 4 + 8 + 8);
        header.putInt(0x46554747).putInt(3).putLong(tensors.size()).putLong(0);
        out.writeBytes(header.array());
        long offset = 0;
        for (Map.Entry<String, GGMLType> tensor : tensors.entrySet()) {
            byte[] name = tensor.getKey().getBytes(StandardCharsets.UTF_8);
            boolean norm = tensor.getValue() == GGMLType.F32;
            ByteBuffer info = little(8 + name.length + 4 + (norm ? 8 : 16) + 4 + 8);
            info.putLong(name.length).put(name);
            if (norm) {
                info.putInt(1).putLong(DIM);
            } else {
                info.putInt(2).putLong(DIM).putLong(DIM);
            }
            info.putInt(tensor.getValue().ordinal()).putLong(offset);
            out.writeBytes(info.array());
            // Offsets are multiples of the default alignment; the data itself is never written.
            offset += 1 << 20;
        }
        return out.toByteArray();
    }

    private static ByteBuffer little(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }
}
