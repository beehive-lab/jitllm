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
 * The memory preflight predicts a Llama Q4_0 file at the size the loader gives it: the file's own
 * bytes where the loader keeps every tensor (CUDA, Metal), and on OpenCL Q4_0 where the loader
 * keeps Q4_0 for the Q4_0 layers, because both ask {@code LlamaPlanProvider} the same questions.
 *
 * <p>The file is synthetic and holds descriptors only, which is all the prediction reads: two
 * layers of 256 x 256 projections, all Q4_0 except one Q4_1 {@code ffn_down}, as {@code
 * llama-quantize ... Q4_0} writes some of them.
 */
public class LlamaRetentionFootprintTest {

    private static final int LAYERS = 2;
    private static final int DIM = 256;
    private static final long PROJECTION_ELEMENTS = (long) DIM * DIM;
    private static final long Q4_0_BYTES = PROJECTION_ELEMENTS / 32 * 18;
    private static final long Q8_0_BYTES = PROJECTION_ELEMENTS / 32 * 34;
    private static final long Q4_1_BYTES = PROJECTION_ELEMENTS / 32 * 20;
    private static final long NORM_BYTES = (long) DIM * Float.BYTES;

    private static final String[] PROJECTIONS = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
    };

    @Rule public final TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void aKeptFileIsPredictedAtItsOwnSize() throws Exception {
        for (BackendId backend : new BackendId[] {BackendId.CUDA, BackendId.METAL}) {
            assertEquals(
                    backend.toString(),
                    13 * Q4_0_BYTES + Q4_1_BYTES + 2 * LAYERS * NORM_BYTES,
                    perLayerBytes(backend));
        }
    }

    @Test
    public void onOpenClQ4_0IsKeptForTheQ4_0Layers() throws Exception {
        // The Q4_1 ffn_down is allowed off Metal, so the loader keeps Q4_0; the Q4_1 projection is
        // kept too, and predicted at Q8_0: an over-estimate.
        assertEquals(
                13 * Q4_0_BYTES + Q8_0_BYTES + 2 * LAYERS * NORM_BYTES,
                perLayerBytes(BackendId.OPENCL));
    }

    private long perLayerBytes(BackendId backend) throws Exception {
        Path file = llamaQ4_0File();
        WeightFootprint footprint =
                ModelLoader.weightFootprint(
                        file,
                        sourceTypes ->
                                TornadoPlanRegistry.deviceRetention(
                                        ArchitectureId.of("llama"), sourceTypes, LAYERS, backend));
        return footprint.perLayerBytes();
    }

    private Path llamaQ4_0File() throws Exception {
        Map<String, GGMLType> tensors = new LinkedHashMap<>();
        for (int layer = 0; layer < LAYERS; layer++) {
            for (String kind : PROJECTIONS) {
                boolean q4_1 = layer == 1 && kind.equals("ffn_down");
                tensors.put(
                        "blk." + layer + "." + kind + ".weight",
                        q4_1 ? GGMLType.Q4_1 : GGMLType.Q4_0);
            }
            tensors.put("blk." + layer + ".attn_norm.weight", GGMLType.F32);
            tensors.put("blk." + layer + ".ffn_norm.weight", GGMLType.F32);
        }
        Path file =
                folder.newFile("llama-q4_0-descriptors-" + System.nanoTime() + ".gguf").toPath();
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
