package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35Int8Kernels;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The int8 path's producer/consumer contract on the batched prefill graph, checked from the built
 * task graphs: whoever writes the buffer a quantization reads must write it in the form the
 * quantizer reads (FP32), and whoever reads the FP16 buffer must have a producer of FP16.
 *
 * <p>Three mixes: everything int8 (the production shape on the 27B's Q4_0 layers); int8 gate/up
 * with an FP16 down projection (the 27B's Q4_1 layers, and any down projection the pair does not
 * take: SwiGLU's FP32 output must be converted for the FP16 consumer); FP16 gate/up with an int8
 * down projection (never produced by the shape rule on the real model, built here through the test
 * seam: SwiGLU must write FP32 for the quantizer, not FP16). The third mix is the one that once
 * read a stale buffer, when SwiGLU wrote FP16 because the gate/up pair was FP16 and the int8 down
 * projection quantized an FP32 buffer nothing had written.
 */
// @formatter:on
public class Qwen35Int8TopologyAccelTest {

    private static final int TRUNK_LAYERS = 4;
    private static final int NEXTN_LAYERS = 1;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 32;
    private static final int ATTENTION_INTERVAL = 4;
    private static final int CONV_KERNEL = 4;
    private static final int STATE_SIZE = 64;
    private static final int GROUPS = 1;
    private static final int VALUE_HEADS = 4;
    private static final int WIDTH = 128;

    @After
    public void restoreFilter() {
        Qwen35BatchPrefillLayers.int8TaskFilterForTests = task -> true;
    }

    private static Qwen35Configuration config(int dim, int hidden) {
        return new Qwen35Configuration(
                "Q8_0",
                dim,
                hidden,
                TRUNK_LAYERS,
                NEXTN_LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                ATTENTION_INTERVAL,
                CONV_KERNEL,
                STATE_SIZE,
                GROUPS,
                VALUE_HEADS,
                VALUE_HEADS * STATE_SIZE,
                16,
                512,
                32,
                32,
                1e-6f,
                1e7f);
    }

    private static TornadoTensor f32(int elements) {
        return new FP32TornadoTensor(new FloatArray(elements));
    }

    private static TornadoTensor blocked(DataType type, int elements) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1 -> 32;
                    case Q5_K, Q6_K -> 256;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        int blockBytes =
                switch (type) {
                    case Q4_0 -> 18;
                    case Q4_1 -> 20;
                    case Q5_K -> 176;
                    case Q6_K -> 210;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        ByteArray bytes = new ByteArray(elements / blockSize * blockBytes);
        return switch (type) {
            case Q4_0 -> new Q4_0TornadoTensor(bytes);
            case Q4_1 -> new Q4_1TornadoTensor(bytes);
            case Q5_K -> new Q5_KTornadoTensor(bytes);
            case Q6_K -> new Q6_KTornadoTensor(bytes);
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /** Q4_0 everywhere but block 0's down projection, which is Q4_1 as on the 27B's first eight. */
    private static Qwen35TornadoWeights weights(Qwen35Configuration config) {
        final int DIM = config.dim();
        final int HIDDEN = config.hiddenDim();
        final int BLOCKS = TRUNK_LAYERS + NEXTN_LAYERS;
        int queryGate = config.queryGateDim();
        int kvDim = config.kvDim();
        int attnDim = config.attentionOutputInputDim();
        TornadoTensor[] attnNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnGate = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnDown = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnUp = new TornadoTensor[BLOCKS];
        TornadoTensor[] wq = new TornadoTensor[BLOCKS];
        TornadoTensor[] wk = new TornadoTensor[BLOCKS];
        TornadoTensor[] wv = new TornadoTensor[BLOCKS];
        TornadoTensor[] wo = new TornadoTensor[BLOCKS];
        TornadoTensor[] qNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] kNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ssmQkv = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmGate = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmConv = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmAlpha = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmBeta = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmDtBias = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmA = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmNorm = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmOut = new TornadoTensor[TRUNK_LAYERS];
        for (int l = 0; l < BLOCKS; l++) {
            attnNorm[l] = f32(DIM);
            ffnNorm[l] = f32(DIM);
            ffnGate[l] = blocked(DataType.Q4_0, DIM * HIDDEN);
            ffnUp[l] = blocked(DataType.Q4_0, DIM * HIDDEN);
            ffnDown[l] = blocked(l == 0 ? DataType.Q4_1 : DataType.Q4_0, HIDDEN * DIM);
            if (l < TRUNK_LAYERS && config.isRecurrentLayer(l)) {
                ssmQkv[l] = blocked(DataType.Q4_0, DIM * config.deltaNetConvDim());
                ssmGate[l] = blocked(DataType.Q4_0, DIM * config.deltaNetValueDim());
                ssmConv[l] = f32(config.deltaNetConvDim() * CONV_KERNEL);
                ssmAlpha[l] = f32(DIM * VALUE_HEADS);
                ssmBeta[l] = f32(DIM * VALUE_HEADS);
                ssmDtBias[l] = f32(VALUE_HEADS);
                ssmA[l] = f32(VALUE_HEADS);
                ssmNorm[l] = f32(config.headValueDim());
                ssmOut[l] = blocked(DataType.Q5_K, config.deltaNetValueDim() * DIM);
            } else {
                wq[l] = blocked(DataType.Q4_0, DIM * queryGate);
                wk[l] = blocked(DataType.Q4_0, DIM * kvDim);
                wv[l] = blocked(DataType.Q4_0, DIM * kvDim);
                wo[l] = blocked(DataType.Q4_0, attnDim * DIM);
                qNorm[l] = f32(HEAD_DIM);
                kNorm[l] = f32(HEAD_DIM);
            }
        }
        return new Qwen35TornadoWeights(
                BLOCKS,
                blocked(DataType.Q4_0, config.vocabularySize() * DIM),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                f32(DIM),
                blocked(DataType.Q6_K, config.vocabularySize() * DIM),
                f32(config.contextLength() * HEAD_DIM),
                f32(config.contextLength() * HEAD_DIM),
                wq,
                wk,
                wv,
                wo,
                qNorm,
                kNorm,
                ssmQkv,
                ssmGate,
                ssmConv,
                ssmAlpha,
                ssmBeta,
                ssmDtBias,
                ssmA,
                ssmNorm,
                ssmOut,
                DataType.Q4_0);
    }

    private static Qwen35BatchPrefillLayers build(Qwen35Configuration config) {
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.qwen35.tensorCores", "true");
        Qwen35State state =
                (Qwen35State) State.withPrefillBatchSize(WIDTH, () -> new Qwen35State(config, -1));
        return new Qwen35BatchPrefillLayers(state, weights(config), config, WIDTH);
    }

    /** Task name to kernel method name, for the tasks of one batched layer graph. */
    private static LinkedHashMap<String, String> kernels(Qwen35BatchPrefillLayers layers, int layer)
            throws Exception {
        java.lang.reflect.Field field = ImmutableTaskGraph.class.getDeclaredField("taskGraph");
        field.setAccessible(true);
        java.lang.reflect.Field impl = TaskGraph.class.getDeclaredField("taskGraphImpl");
        impl.setAccessible(true);
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        for (ImmutableTaskGraph immutable : layers.getLayerImmutableTaskGraphs()) {
            TaskGraph graph = (TaskGraph) field.get(immutable);
            if (!graph.getTaskGraphName().equals("batchLayer_" + layer)) {
                continue;
            }
            var graphImpl =
                    (uk.ac.manchester.tornado.api.TornadoTaskGraphInterface) impl.get(graph);
            // The packages in insertion order give the graph order; the runtime task gives the
            // kernel's method name.
            java.lang.reflect.Field packages =
                    graphImpl.getClass().getDeclaredField("taskPackages");
            packages.setAccessible(true);
            @SuppressWarnings("unchecked")
            List<uk.ac.manchester.tornado.api.common.TaskPackage> list =
                    (List<uk.ac.manchester.tornado.api.common.TaskPackage>) packages.get(graphImpl);
            for (var pkg : list) {
                String id = pkg.getId();
                String bare = id.substring(id.indexOf('.') + 1);
                var task = graphImpl.getTask(bare);
                out.put(bare, task == null ? "?" : task.getTaskName());
            }
        }
        assertFalse("no graph for layer " + layer, out.isEmpty());
        return out;
    }

    private static List<String> names(GridScheduler scheduler, int layer) {
        List<String> names = new ArrayList<>();
        for (String key : scheduler.keySet()) {
            if (key.startsWith("batchLayer_" + layer + ".")) {
                names.add(key.substring(key.indexOf('.') + 1));
            }
        }
        return names;
    }

    private static void assumeInt8() {
        assumeTrue("no int8 tensor-core device", TensorCoreSupport.isInt8MmaCapable());
    }

    /** gate/up int8 with an FP16 down projection: SwiGLU's FP32 output is converted for it. */
    @Test
    public void int8GateUpWithAnFp16DownConvertsSwigluOutput() throws Exception {
        assumeInt8();
        // hidden 5120 puts gate/up on the pair; dim 256 keeps the down projection direct.
        Qwen35BatchPrefillLayers layers = build(config(256, 5120));
        GridScheduler scheduler = new GridScheduler();
        layers.updateGridScheduler(scheduler);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            assertEquals(
                    "layer " + layer + " up", "gemmInt8BlockScaledSwiGLU", k.get("ffn_up_proj"));
            assertEquals("layer " + layer + " gate", "gemmInt8BlockScaled", k.get("ffn_gate_proj"));
            assertFalse("layer " + layer + " kept a SwiGLU task", k.containsKey("ffn_swiglu"));
            assertEquals(
                    "layer "
                            + layer
                            + " converts SwiGLU's FP32 output for the FP16 down projection",
                    "convertToFP16",
                    k.get("ffn_down_fp16"));
            assertFalse(
                    "layer " + layer + " quantized for a non-int8 down",
                    k.containsKey("ffn_down_q8"));
            assertTrue(
                    "layer " + layer + " ffn_down_fp16 grid",
                    names(scheduler, layer).contains("ffn_down_fp16"));
            assertEquals(
                    "layer " + layer + " normed quantization lanes",
                    (long) WIDTH * 256,
                    scheduler.get("batchLayer_" + layer + ".ffn_rms_apply_q8").getGlobalWork()[0]);
        }
    }

    /** FP16 gate/up with an int8 down projection: SwiGLU must write FP32 for the quantizer. */
    @Test
    public void fp16GateUpWithAnInt8DownWritesFp32ForTheQuantizer() throws Exception {
        assumeInt8();
        Qwen35BatchPrefillLayers.int8TaskFilterForTests =
                task -> !task.equals("ffn_gate_proj") && !task.equals("ffn_up_proj");
        // dim 5120 puts the down projection on the pair; hidden 512 keeps gate/up direct.
        Qwen35BatchPrefillLayers layers = build(config(5120, 512));
        GridScheduler scheduler = new GridScheduler();
        layers.updateGridScheduler(scheduler);
        for (int layer = 1; layer < TRUNK_LAYERS; layer++) { // layer 0's down is Q4_1
            LinkedHashMap<String, String> k = kernels(layers, layer);
            assertEquals(
                    "layer " + layer + " down",
                    "gemmInt8BlockScaledResidual",
                    k.get("ffn_down_proj"));
            assertEquals(
                    "layer " + layer + " SwiGLU writes FP32 for the quantizer",
                    "swiGLUBatch",
                    k.get("ffn_swiglu"));
            assertEquals(
                    "layer " + layer + " quantizes SwiGLU's output",
                    "quantizeActivationsQ8Warp",
                    k.get("ffn_down_q8"));
            assertFalse(
                    "layer " + layer + " built an FP16 conversion nothing reads",
                    k.containsKey("ffn_down_fp16"));
            assertEquals(
                    "layer " + layer + " hb quantization lanes",
                    (long) WIDTH * 512,
                    scheduler.get("batchLayer_" + layer + ".ffn_down_q8").getGlobalWork()[0]);
            assertEquals(
                    "layer " + layer + " down decode lanes (a lane per word of four)",
                    5120L * 512 / 4,
                    scheduler.get("batchLayer_" + layer + ".ffn_down_proj_dequant")
                            .getGlobalWork()[0]);
        }
        LinkedHashMap<String, String> q41 = kernels(layers, 0);
        assertEquals(
                "the Q4_1 down stays on the FP16 pair",
                "gemmMMATiledBResidual",
                q41.get("ffn_down_proj"));
        // With FP16 gate/up and an FP16 down, SwiGLU writes the FP16 input itself: no conversion,
        // no quantization.
        assertEquals(
                "the Q4_1 layer's SwiGLU writes FP16", "swiGLUBatchFP16", q41.get("ffn_swiglu"));
        assertFalse(q41.containsKey("ffn_down_fp16"));
        assertFalse(q41.containsKey("ffn_down_q8"));
    }

    /** Everything int8 on a shape where gate/up and down both take the pair. */
    @Test
    public void allInt8QuantizesEachActivationOnceForItsConsumers() throws Exception {
        assumeInt8();
        Qwen35BatchPrefillLayers layers = build(config(5120, 5120));
        for (int layer = 1; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            assertEquals("gemmInt8BlockScaled", k.get("ffn_gate_proj"));
            assertEquals("gemmInt8BlockScaledSwiGLU", k.get("ffn_up_proj"));
            assertEquals("gemmInt8BlockScaledResidual", k.get("ffn_down_proj"));
            assertEquals("quantizeActivationsQ8Warp", k.get("ffn_down_q8"));
            assertEquals("quantizeActivationsQ8Warp", k.get("ffn_rms_apply_q8"));
            assertEquals("quantizeActivationsQ8Warp", k.get("attn_rms_apply_q8"));
            assertFalse(k.containsKey("ffn_swiglu"));
            assertFalse(k.containsKey("ffn_down_fp16"));
            // Graph order: the quantization of an activation precedes every consumer of it and
            // follows the producer, on the one shared byte buffer.
            List<String> order = new ArrayList<>(k.keySet());
            assertTrue(order.indexOf("ffn_rms_apply") < order.indexOf("ffn_rms_apply_q8"));
            assertTrue(order.indexOf("ffn_rms_apply_q8") < order.indexOf("ffn_gate_proj"));
            assertTrue(order.indexOf("ffn_up_proj") < order.indexOf("ffn_down_q8"));
            assertTrue(order.indexOf("ffn_down_q8") < order.indexOf("ffn_down_proj"));
        }
        assertEquals(64, Qwen35Int8Kernels.I8_BK);
    }
}
