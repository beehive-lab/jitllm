package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Test;

/**
 * The one rule for whether a Llama load keeps its Q4_0 projections, which the loader and the memory
 * preflight both ask: every projection Q4_0, a Q4_1 {@code ffn_down} allowed off Metal, and {@code
 * jitllm.q4_0.retain} not {@code false}.
 */
public class LlamaQ4_0RetentionTest {

    private static final String PROPERTY = "jitllm.q4_0.retain";

    private static final String[] PROJECTIONS = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
    };

    private static final int LAYERS = 2;

    @After
    public void clearProperty() {
        System.clearProperty(PROPERTY);
    }

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

    @Test
    public void theRetainPropertySwitchesItOff() {
        System.setProperty(PROPERTY, "false");

        assertFalse(LlamaPlanProvider.retainsQ4_0(q4_0File(), LAYERS, BackendId.CUDA));
    }

    @Test
    public void thePreflightKeepsQ4_0ExactlyWhenTheLoaderDoes() {
        DeviceRetention retained =
                new LlamaPlanProvider().deviceRetention(q4_0File(), LAYERS, BackendId.CUDA);
        System.setProperty(PROPERTY, "false");
        DeviceRetention materialized =
                new LlamaPlanProvider().deviceRetention(q4_0File(), LAYERS, BackendId.CUDA);

        String projection = "blk.0.ffn_gate.weight";
        assertEquals(DataType.Q4_0, retained.deviceType(projection, DataType.Q4_0));
        assertEquals(DataType.Q8_0, materialized.deviceType(projection, DataType.Q4_0));
        // A retained load keeps a Q4_1 ffn_down and a Q6_K output as they are; the prediction
        // counts them at Q8_0 either way, which over-estimates and never under-estimates.
        assertEquals(DataType.Q8_0, retained.deviceType("blk.1.ffn_down.weight", DataType.Q4_1));
        assertEquals(DataType.Q8_0, retained.deviceType("output.weight", DataType.Q6_K));
        assertEquals(DataType.F32, materialized.deviceType("output_norm.weight", DataType.F32));
    }
}
