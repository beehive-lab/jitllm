package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

// @formatter:off
/**
 * What Qwen3.8-27B's device weights are actually in, after loading — not what the file says, and
 * not what the plan was admitted on.
 *
 * <p>The claim this port rests on is that nothing is materialized: five block layouts and F32 live
 * on the device at once, and each is decoded by a kernel chosen from that tensor's own
 * representation. The memory prediction, the parity result and the throughput all depend on it, and
 * none of them would fail visibly if one role were quietly promoted to Q8_0 — it would cost memory
 * and change nothing else. So it is asserted here, per role, on the real file.
 *
 * <p>It also pins the direction: the only Q8_0 tensor in this model is the MTP projection the file
 * itself stores that way, which ordinary generation never reads.
 */
// @formatter:on
public class Qwen35NativeRepresentationAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    private static final int CONTEXT_LENGTH = 512;

    @Test
    public void everyRoleIsRetainedInTheRepresentationTheFileHolds() throws Exception {
        Path modelPath = GoldenFixture.locate(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent", false);
        }

        Model model = ModelLoader.loadModel(modelPath, CONTEXT_LENGTH, true, true);
        Qwen35Configuration config = (Qwen35Configuration) model.configuration();
        Qwen35TornadoWeights weights = (Qwen35TornadoWeights) model.weights();

        assertEquals(
                "the representation the model reports is its projections'",
                DataType.Q4_0,
                weights.dataType());
        assertEquals(
                "token embeddings", DataType.Q4_0, weights.getTokenEmbeddingTable().dataType());
        assertEquals("vocabulary projection", DataType.Q6_K, weights.wclsByteArray.dataType());
        assertEquals("final norm", DataType.F32, weights.rms_final_weight_as_floatArray.dataType());

        Map<DataType, Integer> counted = new EnumMap<>(DataType.class);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            assertEquals(
                    "blk." + l + ".attn_norm",
                    DataType.F32,
                    type(weights.rms_att_weightLayered, l));
            assertEquals(
                    "blk." + l + ".post_attention_norm",
                    DataType.F32,
                    type(weights.rms_ffn_weightLayered, l));
            assertEquals("blk." + l + ".ffn_gate", DataType.Q4_0, type(weights.w1Layered, l));
            assertEquals("blk." + l + ".ffn_up", DataType.Q4_0, type(weights.w3Layered, l));
            // The quantizer left the first eight down projections in Q4_1 and the rest in Q4_0.
            assertEquals(
                    "blk." + l + ".ffn_down",
                    l < 8 ? DataType.Q4_1 : DataType.Q4_0,
                    type(weights.w2Layered, l));

            if (config.isRecurrentLayer(l)) {
                assertEquals("blk." + l + ".attn_qkv", DataType.Q4_0, type(weights.ssmQkv, l));
                assertEquals("blk." + l + ".attn_gate", DataType.Q4_0, type(weights.ssmGate, l));
                assertEquals("blk." + l + ".ssm_out", DataType.Q5_K, type(weights.ssmOut, l));
                assertEquals("blk." + l + ".ssm_alpha", DataType.F32, type(weights.ssmAlpha, l));
                assertEquals("blk." + l + ".ssm_beta", DataType.F32, type(weights.ssmBeta, l));
                assertEquals("blk." + l + ".ssm_conv1d", DataType.F32, type(weights.ssmConv1d, l));
                assertEquals("blk." + l + ".ssm_norm", DataType.F32, type(weights.ssmNorm, l));
                assertEquals("blk." + l + ".ssm_a", DataType.F32, type(weights.ssmA, l));
                assertEquals("blk." + l + ".ssm_dt.bias", DataType.F32, type(weights.ssmDtBias, l));
            } else {
                assertEquals("blk." + l + ".attn_q", DataType.Q4_0, type(weights.wqLayered, l));
                assertEquals("blk." + l + ".attn_k", DataType.Q4_0, type(weights.wkLayered, l));
                assertEquals("blk." + l + ".attn_v", DataType.Q4_0, type(weights.wvLayered, l));
                assertEquals(
                        "blk." + l + ".attn_output", DataType.Q4_0, type(weights.woLayered, l));
                assertEquals("blk." + l + ".attn_q_norm", DataType.F32, type(weights.attnQNorm, l));
                assertEquals("blk." + l + ".attn_k_norm", DataType.F32, type(weights.attnKNorm, l));
            }
            counted.merge(type(weights.w2Layered, l), 1, Integer::sum);
        }

        assertEquals("eight Q4_1 down projections", Integer.valueOf(8), counted.get(DataType.Q4_1));
        assertTrue(
                "nothing in the trunk is materialized as Q8_0",
                !counted.containsKey(DataType.Q8_0));

        System.out.printf(
                "[REPRESENTATIONS] %d recurrent + %d attention layers; ffn_down %s%n",
                config.recurrentLayerCount(), config.keyValueLayerCount(), counted);
    }

    private static DataType type(TornadoTensor[] tensors, int layer) {
        return tensors[layer].dataType();
    }
}
