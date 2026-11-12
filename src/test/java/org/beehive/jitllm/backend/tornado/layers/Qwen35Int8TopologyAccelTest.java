package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.After;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

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
        return new FP32TornadoTensor(new FloatArray(elements), Shape.of(elements));
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
            case Q4_0 -> new Q4_0TornadoTensor(bytes, Shape.of(elements));
            case Q4_1 -> new Q4_1TornadoTensor(bytes, Shape.of(elements));
            case Q5_K -> new Q5_KTornadoTensor(bytes, Shape.of(elements));
            case Q6_K -> new Q6_KTornadoTensor(bytes, Shape.of(elements));
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /** A Q4_0 projection of {@code rows} outputs over {@code cols} inputs. */
    private static TornadoTensor q40(int rows, int cols) {
        return new Q4_0TornadoTensor(new ByteArray(rows * cols / 32 * 18), Shape.of(cols, rows));
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
            ffnGate[l] = q40(HIDDEN, DIM);
            ffnUp[l] = q40(HIDDEN, DIM);
            ffnDown[l] = l == 0 ? blocked(DataType.Q4_1, HIDDEN * DIM) : q40(DIM, HIDDEN);
            if (l < TRUNK_LAYERS && config.isRecurrentLayer(l)) {
                ssmQkv[l] = q40(config.deltaNetConvDim(), DIM);
                ssmGate[l] = q40(config.deltaNetValueDim(), DIM);
                ssmConv[l] = f32(config.deltaNetConvDim() * CONV_KERNEL);
                ssmAlpha[l] = f32(DIM * VALUE_HEADS);
                ssmBeta[l] = f32(DIM * VALUE_HEADS);
                ssmDtBias[l] = f32(VALUE_HEADS);
                ssmA[l] = f32(VALUE_HEADS);
                ssmNorm[l] = f32(config.headValueDim());
                ssmOut[l] = blocked(DataType.Q5_K, config.deltaNetValueDim() * DIM);
            } else {
                wq[l] = q40(queryGate, DIM);
                wk[l] = q40(kvDim, DIM);
                wv[l] = q40(kvDim, DIM);
                wo[l] = q40(DIM, attnDim);
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
        return build(config, false);
    }

    private static Qwen35BatchPrefillLayers build(Qwen35Configuration config, boolean packed) {
        return build(config, weights(config), packed, null);
    }

    private static Qwen35BatchPrefillLayers build(
            Qwen35Configuration config,
            Qwen35TornadoWeights weights,
            boolean packed,
            Qwen35BatchFeedForward feedForward) {
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.qwen35.tensorCores", "true");
        Qwen35State state =
                (Qwen35State) State.withPrefillBatchSize(WIDTH, () -> new Qwen35State(config, -1));
        if (packed) {
            // The Q4_0 projections whose shape divides into tiles, as a plan that packs holds them.
            List<TornadoTensor> tiled = new ArrayList<>();
            for (TornadoTensor[] role : weights.projections()) {
                for (TornadoTensor w : role) {
                    if (w != null
                            && w.dataType() == DataType.Q4_0
                            && WeightPrepacking.fits(w.shape())) {
                        tiled.add(w);
                    }
                }
            }
            state.workspace.prepacking = WeightPrepacking.of(tiled);
        }
        return new Qwen35BatchPrefillLayers(
                state, weights, config, WIDTH, 0, config.numberOfLayers(), feedForward);
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
        assertEquals(64, Int8GemmKernels.I8_BK);
    }

    // @formatter:off
    /**
     * Packed Q4_0 at the 0.8B's widths (dim 1024, FFN 3584), every output width below the FP16
     * pair's threshold: a packed weight has only the int8 GEMM, so each packed projection's input
     * must be quantized even where an unpacked one of that shape would take the direct FP16 kernel,
     * and it stores rather than adds, so its residual pass needs a grid of its own. The attention
     * output and the down projection once read whatever the scratch held from the previous
     * quantization, their residual passes ran without a grid, and the model generated garbage at
     * prefill width 128.
     */
    // @formatter:on
    @Test
    public void packedProjectionsBelowThePairThresholdQuantizeTheirInputsAndAddTheirResiduals()
            throws Exception {
        assumeInt8();
        Qwen35Configuration config = config(1024, 3584);
        Qwen35BatchPrefillLayers layers = build(config, true);
        GridScheduler scheduler = new GridScheduler();
        layers.updateGridScheduler(scheduler);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> grids = names(scheduler, layer);
            for (String task : kernels(layers, layer).keySet()) {
                assertTrue("layer " + layer + " " + task + " has no grid", grids.contains(task));
            }
        }
        for (int layer = 1; layer < TRUNK_LAYERS; layer++) { // layer 0's down is Q4_1
            LinkedHashMap<String, String> k = kernels(layers, layer);
            List<String> order = new ArrayList<>(k.keySet());
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get("ffn_gate_proj"));
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get("ffn_up_proj"));
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get("ffn_down_proj"));
            assertFalse("layer " + layer + " kept a SwiGLU task", k.containsKey("ffn_swiglu"));
            assertFalse(
                    "layer " + layer + " converted for an FP16 down",
                    k.containsKey("ffn_down_fp16"));
            assertEquals(
                    "layer " + layer + " quantizes SwiGLU's output for the packed down",
                    "quantizeActivationsQ8Warp",
                    k.get("ffn_down_q8"));
            assertTrue(order.indexOf("ffn_up_proj") < order.indexOf("ffn_down_q8"));
            assertTrue(order.indexOf("ffn_down_q8") < order.indexOf("ffn_down_proj"));
            assertTrue(order.indexOf("ffn_down_proj") < order.indexOf("ffn_down_residual"));
            assertEquals(
                    "layer " + layer + " hb quantization lanes",
                    (long) WIDTH * 3584,
                    scheduler.get("batchLayer_" + layer + ".ffn_down_q8").getGlobalWork()[0]);
        }
        int attention = ATTENTION_INTERVAL - 1;
        assertFalse(config.isRecurrentLayer(attention));
        LinkedHashMap<String, String> k = kernels(layers, attention);
        List<String> order = new ArrayList<>(k.keySet());
        assertEquals("gemmInt8Q4_0Packed", k.get("attn_output_proj"));
        assertEquals(
                "the gated attention output is quantized for the packed projection",
                "quantizeActivationsQ8Warp",
                k.get("attn_output_q8"));
        assertFalse(k.containsKey("attn_output_fp16"));
        assertTrue(order.indexOf("attn_output_gate") < order.indexOf("attn_output_q8"));
        assertTrue(order.indexOf("attn_output_q8") < order.indexOf("attn_output_proj"));
        assertTrue(order.indexOf("attn_output_proj") < order.indexOf("attn_output_residual"));
    }

    /** An F16 weight of {@code rows} outputs over {@code cols} inputs. */
    private static TornadoTensor f16(int rows, int cols) {
        return new FP16TornadoTensor(new HalfFloatArray(rows * cols), Shape.of(cols, rows));
    }

    /** A Q8_0 weight of {@code rows} outputs over {@code cols} inputs. */
    private static TornadoTensor q8(int rows, int cols) {
        return new Q8_0TornadoTensor(new ByteArray(rows * cols / 32 * 34), Shape.of(cols, rows));
    }

    /** Every projection of the mixer branch (attention or recurrent) of {@code weights} as F16. */
    private static void mixerF16(Qwen35Configuration config, Qwen35TornadoWeights weights) {
        int dim = config.dim();
        for (int l = 0; l < TRUNK_LAYERS + NEXTN_LAYERS; l++) {
            if (l < TRUNK_LAYERS && config.isRecurrentLayer(l)) {
                weights.ssmQkv[l] = f16(config.deltaNetConvDim(), dim);
                weights.ssmGate[l] = f16(config.deltaNetValueDim(), dim);
                weights.ssmAlpha[l] = f16(VALUE_HEADS, dim);
                weights.ssmBeta[l] = f16(VALUE_HEADS, dim);
                weights.ssmOut[l] = f16(dim, config.deltaNetValueDim());
            } else {
                weights.wqLayered[l] = f16(config.queryGateDim(), dim);
                weights.wkLayered[l] = f16(config.kvDim(), dim);
                weights.wvLayered[l] = f16(config.kvDim(), dim);
                weights.woLayered[l] = f16(dim, config.attentionOutputInputDim());
            }
        }
    }

    /** Every dense feed-forward projection of {@code weights} as F16. */
    private static void feedForwardF16(Qwen35Configuration config, Qwen35TornadoWeights weights) {
        for (int l = 0; l < TRUNK_LAYERS + NEXTN_LAYERS; l++) {
            weights.w1Layered[l] = f16(config.hiddenDim(), config.dim());
            weights.w3Layered[l] = f16(config.hiddenDim(), config.dim());
            weights.w2Layered[l] = f16(config.dim(), config.hiddenDim());
        }
    }

    private static boolean quantizes(LinkedHashMap<String, String> kernels) {
        return kernels.containsValue("quantizeActivationsQ8Warp");
    }

    /**
     * A whole F16 model at a width that allocates the int8 scratch: no task reads the scratch, so
     * no layer quantizes and no grid names a quantization, while the FP16 copies the projections
     * read stay. With and without native libraries, since cuBLAS and the JIT GEMM read the same
     * copy.
     */
    @Test
    public void f16LayersQuantizeNothing() throws Exception {
        assumeInt8();
        String previous = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        try {
            for (boolean nativeLibraries : List.of(false, true)) {
                System.setProperty(
                        ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY,
                        Boolean.toString(nativeLibraries));
                Qwen35Configuration config = config(1024, 3584);
                Qwen35TornadoWeights weights = weights(config);
                mixerF16(config, weights);
                feedForwardF16(config, weights);
                Qwen35BatchPrefillLayers layers = build(config, weights, false, null);
                GridScheduler scheduler = new GridScheduler();
                layers.updateGridScheduler(scheduler);
                for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
                    String at = "native=" + nativeLibraries + " layer " + layer;
                    LinkedHashMap<String, String> k = kernels(layers, layer);
                    assertFalse(at + " quantized for no reader: " + k, quantizes(k));
                    assertEquals(at, "convertNormedToFP16", k.get("attn_rms_apply_fp16"));
                    assertEquals(at, "convertNormedToFP16", k.get("ffn_rms_apply_fp16"));
                    // A cuBLAS projection is a library task, which this map does not list.
                    assertEquals(at, nativeLibraries ? null : "gemmMMA", k.get("ffn_gate_proj"));
                    for (String grid : names(scheduler, layer)) {
                        assertFalse(at + " kept the grid of " + grid, grid.endsWith("_q8"));
                    }
                }
            }
        } finally {
            if (previous == null) {
                System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
            } else {
                System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, previous);
            }
        }
    }

    /**
     * The decision follows each norm's own readers. An F16 mixer with prepacked Q4_0 gate and up
     * drops the mixer norm's quantization only, and the feed-forward norm's still precedes its int8
     * reader; the reverse keeps the mixer norm's, and the int8 down projection still quantizes its
     * own input.
     */
    @Test
    public void eachNormQuantizesForItsOwnInt8Readers() throws Exception {
        assumeInt8();
        Qwen35Configuration config = config(1024, 3584);

        Qwen35TornadoWeights f16Mixer = weights(config);
        mixerF16(config, f16Mixer);
        Qwen35BatchPrefillLayers layers = build(config, f16Mixer, true, null);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            List<String> order = new ArrayList<>(k.keySet());
            assertFalse("layer " + layer, k.containsKey("attn_rms_apply_q8"));
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get("ffn_gate_proj"));
            assertTrue(order.indexOf("ffn_rms_apply") < order.indexOf("ffn_rms_apply_q8"));
            assertTrue(order.indexOf("ffn_rms_apply_q8") < order.indexOf("ffn_gate_proj"));
        }

        Qwen35TornadoWeights f16FeedForward = weights(config);
        feedForwardF16(config, f16FeedForward);
        layers = build(config, f16FeedForward, true, null);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            List<String> order = new ArrayList<>(k.keySet());
            String reader = config.isRecurrentLayer(layer) ? "ssm_qkv_proj" : "attn_q_proj";
            assertFalse("layer " + layer, k.containsKey("ffn_rms_apply_q8"));
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get(reader));
            assertTrue(order.indexOf("attn_rms_apply") < order.indexOf("attn_rms_apply_q8"));
            assertTrue(order.indexOf("attn_rms_apply_q8") < order.indexOf(reader));
        }

        Qwen35TornadoWeights int8Down = weights(config);
        feedForwardF16(config, int8Down);
        mixerF16(config, int8Down);
        for (int l = 0; l < TRUNK_LAYERS + NEXTN_LAYERS; l++) {
            int8Down.w2Layered[l] = q40(config.dim(), config.hiddenDim());
        }
        layers = build(config, int8Down, true, null);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            List<String> order = new ArrayList<>(k.keySet());
            assertFalse("layer " + layer, k.containsKey("ffn_rms_apply_q8"));
            assertEquals("layer " + layer, "gemmInt8Q4_0Packed", k.get("ffn_down_proj"));
            assertTrue(order.indexOf("ffn_swiglu") < order.indexOf("ffn_down_q8"));
            assertTrue(order.indexOf("ffn_down_q8") < order.indexOf("ffn_down_proj"));
        }
    }

    /**
     * One int8 reader is enough. A single prepacked Q4_0 query projection among F16 attention
     * projections, and Q8_0 alpha and beta among F16 recurrent projections, which fuse only against
     * the chunk the norm quantized.
     */
    @Test
    public void oneInt8ReaderKeepsTheNormsQuantization() throws Exception {
        assumeInt8();
        Qwen35Configuration config = config(1024, 3584);
        Qwen35TornadoWeights weights = weights(config);
        mixerF16(config, weights);
        feedForwardF16(config, weights);
        for (int l = 0; l < TRUNK_LAYERS + NEXTN_LAYERS; l++) {
            if (l < TRUNK_LAYERS && config.isRecurrentLayer(l)) {
                weights.ssmAlpha[l] = q8(VALUE_HEADS, config.dim());
                weights.ssmBeta[l] = q8(VALUE_HEADS, config.dim());
            } else {
                weights.wqLayered[l] = q40(config.queryGateDim(), config.dim());
            }
        }
        Qwen35BatchPrefillLayers layers = build(config, weights, true, null);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            List<String> order = new ArrayList<>(k.keySet());
            String reader = config.isRecurrentLayer(layer) ? "ssm_alpha_beta" : "attn_q_proj";
            assertEquals(
                    "layer " + layer,
                    config.isRecurrentLayer(layer) ? "alphaBetaQ8_0" : "gemmInt8Q4_0Packed",
                    k.get(reader));
            assertTrue(order.indexOf("attn_rms_apply") < order.indexOf("attn_rms_apply_q8"));
            assertTrue(order.indexOf("attn_rms_apply_q8") < order.indexOf(reader));
            assertFalse("layer " + layer, k.containsKey("ffn_rms_apply_q8"));
        }
    }

    /** A family's own feed-forward reads the quantized chunk, whatever the dense weights are. */
    @Test
    public void aFamilysFeedForwardKeepsItsQuantization() throws Exception {
        assumeInt8();
        Qwen35Configuration config = config(1024, 3584);
        Qwen35TornadoWeights weights = weights(config);
        mixerF16(config, weights);
        feedForwardF16(config, weights);
        Qwen35BatchFeedForward readsTheQuantizedChunk =
                new Qwen35BatchFeedForward() {
                    @Override
                    public void appendTasks(TaskGraph layer, KernelContext context, int index) {}

                    @Override
                    public List<Object> layerWeights(int index) {
                        return List.of();
                    }

                    @Override
                    public Object[] scratch() {
                        return new Object[] {new FloatArray(1)};
                    }

                    @Override
                    public void addWorkerGrids(GridScheduler scheduler, String prefix) {}
                };
        Qwen35BatchPrefillLayers layers = build(config, weights, false, readsTheQuantizedChunk);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            LinkedHashMap<String, String> k = kernels(layers, layer);
            assertEquals("layer " + layer, "quantizeActivationsQ8Warp", k.get("ffn_rms_apply_q8"));
            assertFalse("layer " + layer, k.containsKey("attn_rms_apply_q8"));
        }
    }
}
