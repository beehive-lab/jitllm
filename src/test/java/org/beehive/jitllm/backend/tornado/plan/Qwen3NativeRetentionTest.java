package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The one rule for whether a Qwen3 load keeps its tensors as the file stores them: on every device
 * backend, every file. The loader and the memory preflight both ask it.
 */
public class Qwen3NativeRetentionTest {

    private static final int LAYERS = 2;

    /** A mixed file: every quantized type a kept load holds, and the F32 norms. */
    private static Map<String, DataType> mixedFile() {
        Map<String, DataType> types = new HashMap<>();
        DataType[] projections = {
            DataType.Q4_0, DataType.Q4_1, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K, DataType.Q8_0
        };
        String[] kinds = {
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
        };
        for (int layer = 0; layer < LAYERS; layer++) {
            String blk = "blk." + layer + ".";
            for (int k = 0; k < kinds.length; k++) {
                types.put(
                        blk + kinds[k] + ".weight", projections[(k + layer) % projections.length]);
            }
            for (String norm :
                    new String[] {"attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"}) {
                types.put(blk + norm + ".weight", DataType.F32);
            }
        }
        types.put("token_embd.weight", DataType.Q6_K);
        types.put("output_norm.weight", DataType.F32);
        return types;
    }

    /** Kept on every device backend; the CPU backend does not load device weights. */
    @Test
    public void everyDeviceBackendKeepsTheFile() {
        assertTrue(TornadoPlanProvider.retainsNative(BackendId.CUDA));
        assertTrue(TornadoPlanProvider.retainsNative(BackendId.METAL));
        assertTrue(TornadoPlanProvider.retainsNative(BackendId.OPENCL));
        assertFalse(TornadoPlanProvider.retainsNative(BackendId.CPU));
    }

    /** The preflight keeps each quantized type exactly where the loader does. */
    @Test
    public void thePreflightKeepsEachTypeExactlyWhereTheLoaderDoes() {
        Qwen3PlanProvider provider = new Qwen3PlanProvider();
        var kept = provider.deviceRetention(mixedFile(), LAYERS, BackendId.CUDA);
        var metal = provider.deviceRetention(mixedFile(), LAYERS, BackendId.METAL);
        var openCl = provider.deviceRetention(mixedFile(), LAYERS, BackendId.OPENCL);
        for (DataType type :
                new DataType[] {
                    DataType.Q4_0, DataType.Q4_1, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K
                }) {
            assertEquals(type, kept.deviceType("blk.0.attn_q.weight", type));
            assertEquals(type, metal.deviceType("blk.0.attn_q.weight", type));
            assertEquals(type, openCl.deviceType("blk.0.attn_q.weight", type));
        }
        assertEquals(DataType.Q8_0, kept.deviceType("blk.0.ffn_up.weight", DataType.Q8_0));
        assertEquals(DataType.F16, openCl.deviceType("token_embd.weight", DataType.F16));
        assertEquals(DataType.F32, kept.deviceType("output_norm.weight", DataType.F32));
    }

    /** Admission takes every quantized type a kept load reports, and F16. */
    @Test
    public void everyKeptTypeIsAdmitted() {
        assertEquals(
                java.util.EnumSet.of(
                        DataType.F16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                new Qwen3PlanProvider().supportedDataTypes());
    }
}
