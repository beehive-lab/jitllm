package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The one rule for whether a Qwen3 load keeps its K-quant tensors: the Q4_K_M layout, and nothing
 * else. Eligible files are the two fixtures' layouts; every other mix keeps converting to Q8_0.
 */
public class Qwen3Q4_KRetentionTest {

    private static final int LAYERS = 2;

    private static final String[] Q4_K = {"attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"};

    /**
     * Qwen3-0.6B-Q4_K_M's layout: a tied Q6_K {@code token_embd}, the trunk projections Q4_K, and
     * {@code attn_v} and {@code ffn_down} Q6_K in one layer and Q4_K in the other.
     */
    private static Map<String, DataType> q4_k_mFile() {
        Map<String, DataType> types = new HashMap<>();
        for (int layer = 0; layer < LAYERS; layer++) {
            String blk = "blk." + layer + ".";
            for (String kind : Q4_K) {
                types.put(blk + kind + ".weight", DataType.Q4_K);
            }
            DataType mixed = layer == 0 ? DataType.Q6_K : DataType.Q4_K;
            types.put(blk + "attn_v.weight", mixed);
            types.put(blk + "ffn_down.weight", mixed);
            for (String norm :
                    new String[] {"attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"}) {
                types.put(blk + norm + ".weight", DataType.F32);
            }
        }
        types.put("token_embd.weight", DataType.Q6_K);
        types.put("output_norm.weight", DataType.F32);
        return types;
    }

    private static Map<String, DataType> with(String name, DataType type) {
        Map<String, DataType> types = q4_k_mFile();
        types.put(name, type);
        return types;
    }

    private static Map<String, DataType> without(String name) {
        Map<String, DataType> types = q4_k_mFile();
        types.remove(name);
        return types;
    }

    @Test
    public void theQ4_K_MLayoutIsRetained() {
        assertTrue(Qwen3PlanProvider.retainsQ4_K(q4_k_mFile(), LAYERS, BackendId.CUDA));
        // Qwen3-1.7B-Q4_K_M: a Q4_K token_embd and its own Q6_K output projection.
        Map<String, DataType> untied = with("token_embd.weight", DataType.Q4_K);
        untied.put("output.weight", DataType.Q6_K);
        assertTrue(Qwen3PlanProvider.retainsQ4_K(untied, LAYERS, BackendId.CUDA));
    }

    @Test
    public void anyOtherQuantizationKeepsConverting() {
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("blk.0.ffn_down.weight", DataType.Q5_K), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("blk.1.attn_v.weight", DataType.Q4_0), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("blk.1.attn_q.weight", DataType.Q8_0), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("token_embd.weight", DataType.Q8_0), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("output.weight", DataType.Q8_0), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("token_embd.weight", DataType.F16), LAYERS, BackendId.CUDA));
    }

    @Test
    public void aTrunkProjectionOtherThanQ4_KKeepsConverting() {
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("blk.0.ffn_gate.weight", DataType.Q6_K), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        with("blk.1.attn_k.weight", DataType.Q6_K), LAYERS, BackendId.CUDA));
    }

    @Test
    public void aMissingTensorKeepsConverting() {
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        without("blk.1.attn_q.weight"), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        without("blk.0.attn_v.weight"), LAYERS, BackendId.CUDA));
        assertFalse(
                Qwen3PlanProvider.retainsQ4_K(
                        without("token_embd.weight"), LAYERS, BackendId.CUDA));
        assertFalse(Qwen3PlanProvider.retainsQ4_K(Map.of(), 0, BackendId.CUDA));
    }

    @Test
    public void unquantizedAndQ8_0FilesKeepTheirPaths() {
        Map<String, DataType> f16 = new HashMap<>();
        Map<String, DataType> q8 = new HashMap<>();
        q4_k_mFile()
                .forEach(
                        (name, type) -> {
                            f16.put(name, type.isQuantized() ? DataType.F16 : type);
                            q8.put(name, type.isQuantized() ? DataType.Q8_0 : type);
                        });
        assertFalse(Qwen3PlanProvider.retainsQ4_K(f16, LAYERS, BackendId.CUDA));
        assertFalse(Qwen3PlanProvider.retainsQ4_K(q8, LAYERS, BackendId.CUDA));
    }

    /** The rule against the fixtures' own tensor tables, where they are present. */
    @Test
    public void theFixturesAreJudgedByTheirTensorTables() throws Exception {
        Map<String, Boolean> fixtures =
                Map.of(
                        "Qwen3-0.6B-Q4_K_M.gguf", true,
                        "Qwen3-1.7B-Q4_K_M.gguf", true,
                        "Qwen3-0.6B-Q8_0.gguf", false,
                        "Qwen3-0.6B-f16.gguf", false);
        int judged = 0;
        for (var fixture : fixtures.entrySet()) {
            Path path = GoldenFixture.modelsRoot().resolve(fixture.getKey());
            if (!Files.isRegularFile(path)) {
                continue;
            }
            GGUF gguf = GGUF.loadGGUFMetadata(path);
            Map<String, DataType> types = new HashMap<>();
            gguf.getTensorInfos()
                    .forEach(
                            (name, info) ->
                                    types.put(name, DataTypeMapping.sourceType(info.ggmlType())));
            int layers = ((Number) gguf.getMetadata().get("qwen3.block_count")).intValue();
            assertEquals(
                    fixture.getKey(),
                    fixture.getValue(),
                    Qwen3PlanProvider.retainsQ4_K(types, layers, BackendId.CUDA));
            judged++;
        }
        assumeTrue("no Qwen3 fixture under " + GoldenFixture.modelsRoot(), judged > 0);
    }

    /** Kept on CUDA and Metal only; anywhere else the file is converted as before. */
    @Test
    public void onlyCudaAndMetalKeepIt() {
        assertTrue(Qwen3PlanProvider.retainsQ4_K(q4_k_mFile(), LAYERS, BackendId.METAL));
        assertFalse(Qwen3PlanProvider.retainsQ4_K(q4_k_mFile(), LAYERS, BackendId.OPENCL));
        assertFalse(Qwen3PlanProvider.retainsQ4_K(q4_k_mFile(), LAYERS, BackendId.CPU));
    }

    /** The preflight keeps the K-quants exactly when the loader does. */
    @Test
    public void thePreflightKeepsTheK_QuantsExactlyWhenTheLoaderDoes() {
        Qwen3PlanProvider provider = new Qwen3PlanProvider();
        var kept = provider.deviceRetention(q4_k_mFile(), LAYERS, BackendId.CUDA);
        var converted =
                provider.deviceRetention(
                        with("blk.0.ffn_down.weight", DataType.Q5_K), LAYERS, BackendId.CUDA);
        var openCl = provider.deviceRetention(q4_k_mFile(), LAYERS, BackendId.OPENCL);

        assertEquals(DataType.Q4_K, kept.deviceType("blk.0.attn_q.weight", DataType.Q4_K));
        assertEquals(DataType.Q6_K, kept.deviceType("token_embd.weight", DataType.Q6_K));
        assertEquals(DataType.Q8_0, converted.deviceType("blk.0.attn_q.weight", DataType.Q4_K));
        assertEquals(DataType.Q8_0, converted.deviceType("token_embd.weight", DataType.Q6_K));
        assertEquals(DataType.Q8_0, openCl.deviceType("blk.0.attn_q.weight", DataType.Q4_K));
        assertEquals(DataType.F32, kept.deviceType("output_norm.weight", DataType.F32));
    }
}
