package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The rule for whether a Llama load runs the Q4_0 layers, which the loader asks: every projection
 * Q4_0, a Q4_1 {@code ffn_down} allowed off Metal.
 */
public class LlamaQ4_0RetentionTest {

    private static final String[] PROJECTIONS = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
    };

    private static final int LAYERS = 2;

    /** Every projection of every layer Q4_0, the norms F32 and the vocabulary Q6_K. */
    private static Map<String, DataType> q4_0File() {
        Map<String, DataType> types = new HashMap<>();
        for (int layer = 0; layer < LAYERS; layer++) {
            for (String kind : PROJECTIONS) {
                types.put("blk." + layer + "." + kind + ".weight", DataType.Q4_0);
            }
            types.put("blk." + layer + ".attn_norm.weight", DataType.F32);
            types.put("blk." + layer + ".ffn_norm.weight", DataType.F32);
        }
        types.put("token_embd.weight", DataType.Q6_K);
        types.put("output_norm.weight", DataType.F32);
        return types;
    }

    @Test
    public void anAllQ4_0FileIsRetainedOnEveryBackend() {
        assertTrue(LlamaPlanProvider.retainsQ4_0(q4_0File(), LAYERS, BackendId.CUDA));
        assertTrue(LlamaPlanProvider.retainsQ4_0(q4_0File(), LAYERS, BackendId.METAL));
    }

    @Test
    public void aQ4_1DownProjectionIsRetainedOffMetalOnly() {
        Map<String, DataType> types = q4_0File();
        types.put("blk.1.ffn_down.weight", DataType.Q4_1);

        assertTrue(LlamaPlanProvider.retainsQ4_0(types, LAYERS, BackendId.CUDA));
        assertFalse(LlamaPlanProvider.retainsQ4_0(types, LAYERS, BackendId.METAL));
    }

    @Test
    public void anyOtherProjectionTypeMaterializesTheModel() {
        Map<String, DataType> q4_1Value = q4_0File();
        q4_1Value.put("blk.0.attn_v.weight", DataType.Q4_1);
        Map<String, DataType> q8 = q4_0File();
        q8.put("blk.1.ffn_up.weight", DataType.Q8_0);
        Map<String, DataType> missing = q4_0File();
        missing.remove("blk.0.attn_k.weight");

        assertFalse(LlamaPlanProvider.retainsQ4_0(q4_1Value, LAYERS, BackendId.CUDA));
        assertFalse(LlamaPlanProvider.retainsQ4_0(q8, LAYERS, BackendId.CUDA));
        assertFalse(LlamaPlanProvider.retainsQ4_0(missing, LAYERS, BackendId.CUDA));
        assertFalse(LlamaPlanProvider.retainsQ4_0(Map.of(), 0, BackendId.CUDA));
    }
}
