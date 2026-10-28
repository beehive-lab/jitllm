package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertEquals;

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
 * The memory preflight predicts a K-quant file of each family migrated after Qwen3 and Llama at the
 * size its loader gives it: the file's own bytes on CUDA and Metal, where the loader keeps every
 * tensor ({@code TornadoPlanProvider.retainsNative}), and Q8_0 where it converts them. Devstral
 * keeps the projections of a file whose layers are all Q4_K and Q6_K on every backend, and
 * elsewhere converts only its embedding and output projection.
 *
 * <p>The file is the synthetic Q4_K_M layout of {@link Qwen3RetentionFootprintTest}: descriptors
 * only, two layers of 256 x 256 projections, {@code attn_v} and {@code ffn_down} Q6_K in the first
 * layer and Q4_K in the second, and a Q6_K {@code token_embd}.
 */
public class NativeRetentionFootprintTest {

    private static final int LAYERS = 2;
    private static final int DIM = 256;
    private static final long ELEMENTS = (long) DIM * DIM;
    private static final long Q4_K_BYTES = ELEMENTS / 256 * 144;
    private static final long Q6_K_BYTES = ELEMENTS / 256 * 210;
    private static final long Q8_0_BYTES = ELEMENTS / 32 * 34;
    private static final long NORM_BYTES = (long) DIM * Float.BYTES;

    /** Both layers' projections and norms, as the file stores them. */
    private static final long KEPT_PER_LAYER =
            10 * Q4_K_BYTES + 2 * Q6_K_BYTES + 2 * Q4_K_BYTES + 4 * NORM_BYTES;

    private static final long CONVERTED_PER_LAYER = 14 * Q8_0_BYTES + 4 * NORM_BYTES;

    private static final String[] FAMILIES = {
        "mistral", "qwen2", "deepseek-r1-distill-qwen", "phi3", "granite", "devstral", "qwen2-moe"
    };

    @Rule public final TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void aKeptFileIsPredictedAtItsOwnSizeOnCudaAndMetal() throws Exception {
        Path file = q4_K_MFile();
        for (String family : FAMILIES) {
            for (BackendId backend : new BackendId[] {BackendId.CUDA, BackendId.METAL}) {
                WeightFootprint footprint = footprint(family, file, backend);
                assertEquals(family + " on " + backend, KEPT_PER_LAYER, footprint.perLayerBytes());
                assertEquals(family + " on " + backend, Q6_K_BYTES, footprint.globalBytes());
            }
        }
    }

    @Test
    public void elsewhereTheFileIsPredictedAtQ8_0() throws Exception {
        Path file = q4_K_MFile();
        for (String family : FAMILIES) {
            if (family.equals("devstral")) {
                continue;
            }
            WeightFootprint footprint = footprint(family, file, BackendId.OPENCL);
            assertEquals(family, CONVERTED_PER_LAYER, footprint.perLayerBytes());
            assertEquals(family, Q8_0_BYTES, footprint.globalBytes());
        }
    }

    @Test
    public void elsewhereDevstralKeepsItsK_QuantLayersOnly() throws Exception {
        WeightFootprint footprint = footprint("devstral", q4_K_MFile(), BackendId.OPENCL);
        assertEquals(KEPT_PER_LAYER, footprint.perLayerBytes());
        assertEquals(Q8_0_BYTES, footprint.globalBytes());
    }

    private static WeightFootprint footprint(String family, Path file, BackendId backend)
            throws Exception {
        return ModelLoader.weightFootprint(
                file,
                sourceTypes ->
                        TornadoPlanRegistry.deviceRetention(
                                ArchitectureId.of(family), sourceTypes, LAYERS, backend));
    }

    private Path q4_K_MFile() throws Exception {
        Map<String, GGMLType> tensors = new LinkedHashMap<>();
        for (int layer = 0; layer < LAYERS; layer++) {
            String blk = "blk." + layer + ".";
            for (String kind : new String[] {"attn_q", "attn_k", "attn_output", "ffn_gate"}) {
                tensors.put(blk + kind + ".weight", GGMLType.Q4_K);
            }
            tensors.put(blk + "ffn_up.weight", GGMLType.Q4_K);
            GGMLType mixed = layer == 0 ? GGMLType.Q6_K : GGMLType.Q4_K;
            tensors.put(blk + "attn_v.weight", mixed);
            tensors.put(blk + "ffn_down.weight", mixed);
            for (String norm : new String[] {"attn_norm", "ffn_norm"}) {
                tensors.put(blk + norm + ".weight", GGMLType.F32);
            }
        }
        tensors.put("token_embd.weight", GGMLType.Q6_K);
        Path file = folder.newFile("q4_k_m-descriptors-" + System.nanoTime() + ".gguf").toPath();
        Files.write(file, Qwen3RetentionFootprintTest.headerOnlyGguf(tensors));
        return file;
    }
}
