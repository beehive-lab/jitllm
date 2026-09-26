package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MMAKernels;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
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
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The shape of a {@code qwen35} plan, on the smallest topology that has both layer kinds.
 *
 * <p>The real model is 64 layers of 5120 dimensions and does not fit in a unit test, but every
 * structural question about the plan is answerable at 1/80th of the size: how many graphs there
 * are, how many tasks each holds, which mixer a block gets, which kernel each weight-reading task
 * was bound to, and whether the MTP block leaked into ordinary generation. Getting those wrong on
 * the 27B costs a multi-minute load before the failure appears — and a wrong dispatch does not fail
 * at all, it produces fluent, wrong text.
 *
 * <p>The synthetic model is <b>mixed on purpose</b>, in the same places the real one is: Q4_1 down
 * projections on the early blocks, Q5_K recurrent outputs, a Q6_K vocabulary projection, Q4_0
 * everywhere else, F32 norms and SSM parameters.
 */
// @formatter:on
public class Qwen35GraphTopologyAccelTest {

    private static final int DIM = 256;
    private static final int HIDDEN = 512;
    private static final int TRUNK_LAYERS = 8;
    private static final int NEXTN_LAYERS = 1;
    private static final int BLOCKS = TRUNK_LAYERS + NEXTN_LAYERS;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 32;
    private static final int ATTENTION_INTERVAL = 4;

    private static final int CONV_KERNEL = 4;
    private static final int STATE_SIZE = 64;
    private static final int GROUPS = 4;
    private static final int VALUE_HEADS = 4;
    private static final int INNER = 256;

    private static Qwen35Configuration config() {
        return config(TRUNK_LAYERS);
    }

    /** The same fixture with a chosen trunk depth, so a remainder group can be exercised. */
    private static Qwen35Configuration config(int trunkLayers) {
        return config(trunkLayers, HIDDEN);
    }

    /** The same fixture with a chosen feed-forward width, so a GEMM-eligible one can be built. */
    private static Qwen35Configuration config(int trunkLayers, int hidden) {
        return config(trunkLayers, hidden, STATE_SIZE);
    }

    /**
     * The same fixture with a chosen recurrent state width; the delta-net inner width scales with
     * it so each value head's width — which is the width the scan's state has — equals it.
     */
    private static Qwen35Configuration config(int trunkLayers, int hidden, int stateSize) {
        return new Qwen35Configuration(
                "Q8_0",
                DIM,
                hidden,
                trunkLayers,
                NEXTN_LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                ATTENTION_INTERVAL,
                CONV_KERNEL,
                stateSize,
                GROUPS,
                VALUE_HEADS,
                VALUE_HEADS * stateSize,
                16,
                512,
                32,
                32,
                1e-6f,
                1e7f);
    }

    // ---- synthetic tensors --------------------------------------------------

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
        if (elements % blockSize != 0) {
            fail(type + " needs a whole number of blocks; " + elements + " is not one");
        }
        ByteArray bytes = new ByteArray(elements / blockSize * blockBytes);
        return switch (type) {
            case Q4_0 -> new Q4_0TornadoTensor(bytes);
            case Q4_1 -> new Q4_1TornadoTensor(bytes);
            case Q5_K -> new Q5_KTornadoTensor(bytes);
            case Q6_K -> new Q6_KTornadoTensor(bytes);
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /**
     * Weights shaped like the real file: per-layer arrays indexed by absolute block, {@code null}
     * at blocks of the other kind, and the same representation per role that the 27B holds.
     */
    private static Qwen35TornadoWeights weights(Qwen35Configuration config) {
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
            // The 27B holds Q4_1 down projections on its first eight blocks only.
            ffnDown[l] = blocked(l < 2 ? DataType.Q4_1 : DataType.Q4_0, HIDDEN * DIM);
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

    private static Qwen35FFNLayers build(Qwen35Configuration config) {
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Qwen35State state = new Qwen35State(config, -1);
            return new Qwen35FFNLayers(
                    "qwen35FFN", state, weights(config), config, SchedulerType.NVIDIA);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    // @formatter:off
    /**
     * Whether the packed-integer path is in play for this JVM, read exactly as {@code
     * Qwen35FFNLayers} reads it: the device capability, and the escape hatch the exact-comparison
     * tests set.
     *
     * <p>Two facts, not one, and the second is why this is a method rather than a capability test.
     * {@code -Djitllm.qwen35.packedIntegerDot=false} puts every packed projection back on the
     * floating-point kernels, so a suite run that way builds a different plan — and the cases below
     * assert <b>that</b> plan rather than skipping. It is the existing escape hatch and not a new
     * option: no other switch selects this.
     */
    // @formatter:on
    private static boolean packedPathEnabled() {
        return org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                        .capabilities()
                        .supports(
                                org.beehive.jitllm.runtime.backend.DeviceCapability
                                        .PACKED_INTEGER_DOT)
                && !"false"
                        .equalsIgnoreCase(
                                System.getProperty("jitllm.qwen35.packedIntegerDot", "true"));
    }

    // @formatter:off
    /**
     * The delta-net norms are dispatched a workgroup per head, not a lane per head.
     *
     * <p>Both reductions were registered one thread per head — sixteen threads for the whole L2
     * norm, forty-eight for the gated norm — and the fix is a grid, so a grid is what this pins.
     * The task names did not change when the kernels did, which is exactly why asserting on names
     * would not have caught a silent revert to the narrow dispatch.
     *
     * <p>Read off the plan the engine actually builds, not recomputed from the config: global work
     * must be the whole activation and local work the head's width.
     */
    // @formatter:on
    @Test
    public void theDeltaNetNormsOwnAHeadPerWorkgroup() {
        Qwen35Configuration config = config();
        GridScheduler scheduler = new GridScheduler();
        build(config).updateGridScheduler(scheduler);

        int keyHeads = config.numberOfKeyHeads();
        int keyDim = config.headKeyDim();
        int valueHeads = config.numberOfValueHeads();
        int valueDim = config.headValueDim();
        // The widths this family has, and the condition the shared tree needs. If either stops
        // being a power of two the engine falls back and this expectation is the wrong one.
        assertEquals("the key head's width", 0, keyDim & (keyDim - 1));
        assertEquals("the value head's width", 0, valueDim & (valueDim - 1));

        int recurrent = -1;
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            if (config.isRecurrentLayer(layer)) {
                recurrent = layer;
                break;
            }
        }
        assertTrue("no recurrent layer to inspect", recurrent >= 0);
        String prefix = "layer_" + recurrent + ".";
        for (String task : new String[] {"ssm_l2norm_q", "ssm_l2norm_k"}) {
            WorkerGrid grid = scheduler.get(prefix + task);
            assertEquals(task + " global work", keyHeads * keyDim, (int) grid.getGlobalWork()[0]);
            assertEquals(task + " local work", keyDim, (int) grid.getLocalWork()[0]);
        }
        // The delta rule's dispatch follows the device's workgroup limit: eight lanes a column
        // where 8 * width fits, two where only 2 * width fits, the elementwise default
        // otherwise. Asserted on the grid because the task name is the same either way, and
        // against the same decision the builder took for this device.
        long limit =
                Qwen35FFNLayers.deltaRuleWorkGroupLimit(
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current());
        var geometry = Qwen35FFNLayers.selectDeltaRuleGeometry(valueDim, limit);
        WorkerGrid expectedDelta = Qwen35FFNLayers.deltaRuleWorker(geometry, config);
        WorkerGrid delta = scheduler.get(prefix + "ssm_delta_rule");
        assertEquals(
                "ssm_delta_rule global work (" + geometry + ", limit " + limit + ")",
                expectedDelta.getGlobalWork()[0],
                delta.getGlobalWork()[0]);
        assertEquals(
                "ssm_delta_rule local work (" + geometry + ")",
                expectedDelta.getLocalWork()[0],
                delta.getLocalWork()[0]);
        if (limit >= 8L * valueDim) {
            assertEquals(
                    "a device admitting 8 x width takes the eight-part kernel",
                    Qwen35FFNLayers.DeltaRuleGeometry.SPLIT8,
                    geometry);
        }

        WorkerGrid gated = scheduler.get(prefix + "ssm_gated_norm");
        assertEquals(
                "ssm_gated_norm global work",
                valueHeads * valueDim,
                (int) gated.getGlobalWork()[0]);
        assertEquals("ssm_gated_norm local work", valueDim, (int) gated.getLocalWork()[0]);
    }

    // @formatter:off
    /**
     * The batched plan's decode layers are laid out two to a graph.
     *
     * <p>Decode submits one graph at a time, so pairing adjacent layers halves the submissions.
     * What this pins is the part that could silently go wrong: the graph count, that each graph
     * really holds two layers, that the two layers' tasks do not collide on a grid key, and that
     * they are registered in layer order. The task names themselves are unchanged for the first
     * layer of each graph, so only the second one's keys are new.
     */
    // @formatter:on
    @Test
    public void theBatchedDecodeLayersArePairedTwoToAGraph() {
        Qwen35Configuration config = config();
        Qwen35FFNLayersBatchDecode decode = buildBatchDecode(config);

        int layers = config.numberOfLayers();
        int group = decode.layersPerGraph();
        int expectedGraphs = (layers + group - 1) / group;
        assertEquals(
                "one decode graph per group of layers",
                expectedGraphs,
                decode.getFFNLayerImmutableTaskGraphs().size());

        GridScheduler scheduler = new GridScheduler();
        decode.updateGridScheduler(scheduler);

        // Grid keys are graphName.taskName. Group them by graph and check the shape.
        Map<String, List<String>> byGraph = new LinkedHashMap<>();
        Set<String> all = new LinkedHashSet<>();
        for (String key : scheduler.keySet()) {
            assertTrue("a grid key is registered twice: " + key, all.add(key));
            int dot = key.indexOf('.');
            byGraph.computeIfAbsent(key.substring(0, dot), g -> new ArrayList<>())
                    .add(key.substring(dot + 1));
        }
        assertEquals("one graph per pair", expectedGraphs, byGraph.size());

        // The scheduler's key set is not ordered, so the names are checked as a set and each
        // graph is looked up by name; build order is pinned separately by the last graph's id.
        Set<String> expectedNames = new LinkedHashSet<>();
        for (int g = 0; g < expectedGraphs; g++) {
            expectedNames.add("layer_" + (group * g));
        }
        assertEquals("graphs are named for their first layer", expectedNames, byGraph.keySet());
        assertEquals(
                "the logits graph chains from the last decode graph",
                "layer_" + (group * (expectedGraphs - 1)),
                decode.getLastFFNLayerTaskGraphID());

        for (int g = 0; g < expectedGraphs; g++) {
            String name = "layer_" + (group * g);
            List<String> tasks = byGraph.get(name);
            assertEquals(
                    "no task name repeats inside " + name,
                    tasks.size(),
                    new LinkedHashSet<>(tasks).size());
            // Every slot this graph actually holds must be represented, and no other.
            int held = Math.min(group, layers - group * g);
            for (int slot = 1; slot < group; slot++) {
                String slotPrefix = "l" + slot + "_";
                boolean present = tasks.stream().anyMatch(t -> t.startsWith(slotPrefix));
                assertEquals(
                        name
                                + " slot "
                                + slot
                                + " should be "
                                + (slot < held ? "present" : "absent"),
                        slot < held,
                        present);
            }
        }
    }

    // @formatter:off
    /**
     * Which graph a paired decode layer consumes its predecessor's output from.
     *
     * <p>The check that was missing when pairing first went in, and the one that would have caught
     * the failure it caused. A layer that is not the first in its graph has its producer inside the
     * same graph and must not consume at all; a layer that <b>is</b> first consumes from the graph
     * holding the previous layer — which is {@code layer_0} for layer 2, not {@code layer_1}.
     * Resolving it as the literal {@code "layer_" + (layerIndex - 1)} names a graph that does not
     * exist once layers are grouped, and the only thing that caught it was a 99% elementwise
     * failure in a two-minute device parity run.
     */
    // @formatter:on
    @Test
    public void aPairedLayerConsumesFromTheGraphHoldingItsPredecessor() {
        Qwen35Configuration config = config();
        Qwen35FFNLayersBatchDecode decode = buildBatchDecode(config);
        int layers = config.numberOfLayers();
        int group = decode.layersPerGraph();
        assertTrue("this fixture must span more than one graph", layers > group);

        // The graph names that exist, read off the plan this family actually built.
        GridScheduler scheduler = new GridScheduler();
        decode.updateGridScheduler(scheduler);
        Set<String> builtGraphNames = new LinkedHashSet<>();
        for (String key : scheduler.keySet()) {
            builtGraphNames.add(key.substring(0, key.indexOf('.')));
        }
        assertEquals("one graph per group", (layers + group - 1) / group, builtGraphNames.size());

        for (int layer = 0; layer < layers; layer++) {
            int slot = layer % group;
            assertEquals(
                    "layer " + layer + " belongs to its group's graph",
                    "layer_" + (layer - slot),
                    decode.layerGraphName(layer));
            assertEquals(
                    "only the first layer of a graph owns the graph's inputs",
                    slot == 0,
                    decode.firstLayerOfGraph(layer));
            assertEquals(
                    "only the last layer of a graph publishes its outputs",
                    slot == group - 1 || layer == layers - 1,
                    decode.lastLayerOfGraph(layer));

            if (slot == 0 && layer > 0) {
                // The group boundary: the producer is the previous GROUP's graph, never the
                // preceding layer's own index, which is not a graph name at all.
                String producer = decode.layerGraphName(layer - 1);
                assertEquals(
                        "layer " + layer + " must consume from the previous group's graph",
                        "layer_" + (layer - group),
                        producer);
                // Checked against the graphs the family actually built, not against another
                // computed string: a producer that names no graph is the failure mode.
                assertTrue(
                        "layer "
                                + layer
                                + " names a producer that was never built: "
                                + producer
                                + " (built: "
                                + builtGraphNames
                                + ")",
                        builtGraphNames.contains(producer));
            } else if (layer > 0) {
                assertEquals(
                        "a layer inside a graph shares it with its predecessor",
                        decode.layerGraphName(layer - 1),
                        decode.layerGraphName(layer));
            }
        }
    }

    // @formatter:off
    /**
     * A trunk depth that does not divide by the group size leaves a smaller final graph.
     *
     * <p>The production depth of 64 divides by four exactly, so the remainder path would otherwise
     * never be built. Six layers at four to a graph gives one full graph and one holding two, and
     * the second graph must still own its own inputs and outputs.
     */
    // @formatter:on
    @Test
    public void aRemainderGroupTakesASmallerGraph() {
        Qwen35Configuration config = config(6);
        Qwen35FFNLayersBatchDecode decode = buildBatchDecode(config);
        int group = decode.layersPerGraph();
        assumeTrue("this case needs a remainder", 6 % group != 0);

        assertEquals(
                "a full graph and a remainder graph",
                2,
                decode.getFFNLayerImmutableTaskGraphs().size());
        assertEquals("layer_0", decode.layerGraphName(0));
        assertEquals("layer_4", decode.layerGraphName(4));
        assertEquals("layer_4", decode.layerGraphName(5));
        assertEquals(
                "the logits graph chains from the remainder graph",
                "layer_4",
                decode.getLastFFNLayerTaskGraphID());

        // The remainder graph owns its edges: first layer consumes, last layer persists.
        assertTrue(decode.firstLayerOfGraph(4));
        assertFalse(decode.firstLayerOfGraph(5));
        assertFalse(decode.lastLayerOfGraph(4));
        assertTrue("the trunk's final layer ends its graph", decode.lastLayerOfGraph(5));
        assertEquals(
                "the remainder graph consumes from the full graph before it",
                "layer_0",
                decode.layerGraphName(3));

        GridScheduler scheduler = new GridScheduler();
        decode.updateGridScheduler(scheduler);
        Set<String> names = new LinkedHashSet<>();
        for (String key : scheduler.keySet()) {
            names.add(key.substring(0, key.indexOf('.')));
        }
        assertEquals(Set.of("layer_0", "layer_4"), names);
    }

    private static Qwen35FFNLayersBatchDecode buildBatchDecode(Qwen35Configuration config) {
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Qwen35State state = new Qwen35State(config, -1);
            return new Qwen35FFNLayersBatchDecode(
                    "qwen35FFN", state, weights(config), config, SchedulerType.NVIDIA);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static List<String> taskNames(GridScheduler scheduler, int layer) {
        List<String> names = new ArrayList<>();
        for (String key : scheduler.keySet()) {
            if (key.startsWith("layer_" + layer + ".")) {
                names.add(key.substring(key.indexOf('.') + 1));
            }
        }
        return names;
    }

    // @formatter:off
    /**
     * Exactly which projections read a quantized activation, and — more to the point — which do
     * not.
     *
     * <p>The packed-integer path is eligible when three things hold at once: the weights are Q4_0,
     * the projection folds no residual, and the buffer it reads still holds the activation the
     * branch quantized. The third is a fact about <b>ordering</b>, and it is the one that can go
     * wrong silently: {@code wrapXb} is written twice more inside a layer, by the attention
     * branch's gated output and by the feed-forward norm, so a projection reading it after either
     * of those would consume the previous activation's quants and produce plausible, wrong numbers.
     *
     * <p>{@code attn_output_proj} is the case that proves the point. It reads {@code wrapXb} — the
     * same array — after the attention branch has overwritten it, and it must <b>not</b> be packed.
     * Today it is also excluded by folding a residual, which is exactly why this asserts the
     * outcome rather than trusting that coincidence.
     */
    // @formatter:on
    @Test
    public void onlyProjectionsReadingTheQuantizedActivationArePacked() {
        assumeTrue("the packed-integer path is not in play", packedPathEnabled());
        Qwen35Configuration config = config();
        Set<String> packed = new LinkedHashSet<>();
        Set<String> notPacked = new LinkedHashSet<>();
        for (Qwen35FFNLayers.Dispatch dispatch : build(config).dispatchInventory()) {
            (dispatch.quantizedActivation() ? packed : notPacked).add(dispatch.task());
        }

        assertEquals("the projections that read a quantized activation", expected(), packed);
        assertTrue(
                "attn_output_proj reads wrapXb after the attention branch overwrote it, so it"
                        + " cannot take the quantized activation",
                notPacked.contains("attn_output_proj"));
        if (packedPathEnabled()) {
            assertTrue(
                    "ssm_out_proj must take the packed path when the Q5_K readout projection is"
                            + " enabled",
                    packed.contains("ssm_out_proj"));
        } else {
            assertTrue(
                    "ssm_out_proj reads the delta-net readout, which nothing has quantized",
                    notPacked.contains("ssm_out_proj"));
        }
    }

    // @formatter:off
    /**
     * The fused gate/up is packed, and its activation is its own.
     *
     * <p>What it pins is the thing that would be wrong if the quantization were emitted in the
     * wrong place: {@code ffn_gate_up} packed, a second quantization present, and the branch
     * projections still packed from theirs. The feed-forward norm writes over the activation the
     * branch quantized, so a packed {@code ffn_gate_up} without its own quantization would be
     * reading the attention norm's output.
     *
     * <p>Where the capability holds the quantization is carried by the norm's <b>apply</b> rather
     * than by a task of its own, so what this checks for is the apply — there must be one per
     * branch, and the feed-forward's must be distinct from the attention norm's.
     */
    // @formatter:on
    @Test
    public void theFeedForwardPacksAgainstItsOwnQuantization() {
        assumeTrue("the packed-integer path is not in play", packedPathEnabled());
        Qwen35Configuration config = config();
        Set<String> packed = new LinkedHashSet<>();
        for (Qwen35FFNLayers.Dispatch dispatch : build(config).dispatchInventory()) {
            if (dispatch.quantizedActivation()) {
                packed.add(dispatch.task());
            }
        }
        assertTrue("ffn_gate_up did not take the packed path", packed.contains("ffn_gate_up"));
        assertTrue("the branch projections lost theirs", packed.contains("ssm_qkv_proj"));

        GridScheduler scheduler = new GridScheduler();
        build(config).updateGridScheduler(scheduler);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            assertTrue(
                    "layer "
                            + layer
                            + " packs the feed-forward without quantizing its activation:"
                            + " "
                            + tasks,
                    tasks.contains("ffn_rms_apply"));
            assertTrue(
                    "layer " + layer + " lost the branch's quantization",
                    tasks.contains("attn_rms_apply"));
            assertFalse(
                    "layer " + layer + " still emits a standalone quantization task: " + tasks,
                    tasks.contains("xb_quantize") || tasks.contains("ffn_xb_quantize"));
        }
    }

    // @formatter:off
    /**
     * The delta-net readout's quantization: present exactly where it belongs, or absent entirely.
     *
     * <p>Placement is the thing that can go wrong silently here, and it is asserted through the
     * consequences the builder records rather than through a task ordering the grid scheduler does
     * not preserve. The readout's quantization overwrites the same three scratch arrays the
     * branch's projections read, and emitting it clears the provenance flag those projections
     * consult. So a quantization emitted <b>before</b> {@code ssm_qkv_proj} and {@code
     * ssm_gate_proj} would show up as those two losing the packed path, and one emitted after
     * {@code ssm_out_proj} would show up as {@code ssm_out_proj} not taking it. Requiring all three
     * packed at once pins the task to the window between them.
     *
     * <p>The other half is the layer kind: only a recurrent layer has a readout, so only a
     * recurrent layer may carry this task.
     *
     * <p>This runs in both directions rather than skipping in one: on a device without the
     * capability, or under the {@code packedIntegerDot} escape hatch, it asserts the floating-point
     * topology instead — no quantization tasks at all, and every projection including {@code
     * ssm_out_proj} on the kernels that decode to floats.
     */
    // @formatter:on
    @Test
    public void theDeltaNetReadoutQuantizationSitsWhereItBelongs() {
        Qwen35Configuration config = config();
        Qwen35FFNLayers layers = build(config);
        GridScheduler scheduler = layers.updateGridScheduler(new GridScheduler());

        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            boolean recurrent = config.isRecurrentLayer(layer);
            boolean expected = packedPathEnabled() && recurrent;
            assertEquals(
                    "layer "
                            + layer
                            + (recurrent ? " (recurrent)" : " (attention)")
                            + " tasks "
                            + tasks,
                    expected,
                    tasks.contains("ssm_out_quantize"));
        }

        Set<String> packed = new LinkedHashSet<>();
        Set<String> notPacked = new LinkedHashSet<>();
        for (Qwen35FFNLayers.Dispatch dispatch : layers.dispatchInventory()) {
            (dispatch.quantizedActivation() ? packed : notPacked).add(dispatch.task());
        }

        if (packedPathEnabled()) {
            assertTrue("ssm_out_proj lost the packed path", packed.contains("ssm_out_proj"));
            assertTrue(
                    "ssm_qkv_proj lost its quantization, so the readout's was emitted too early",
                    packed.contains("ssm_qkv_proj"));
            assertTrue(
                    "ssm_gate_proj lost its quantization, so the readout's was emitted too early",
                    packed.contains("ssm_gate_proj"));
            assertTrue(
                    "the feed-forward reads its own quantization, not the readout's",
                    packed.contains("ffn_gate_up"));
        } else {
            // The fallback plan, asserted rather than assumed: nothing packed anywhere, and none
            // of the quantization tasks emitted. The two norm applies are still there, plain.
            assertTrue(
                    "no projection may read a quantized activation without the packed path: "
                            + packed,
                    packed.isEmpty());
            assertTrue(
                    "ssm_out_proj must stay on the floating-point kernel",
                    notPacked.contains("ssm_out_proj"));
            for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
                List<String> tasks = taskNames(scheduler, layer);
                assertFalse(
                        "layer " + layer + " quantized an activation",
                        tasks.contains("xb_quantize"));
                assertFalse(
                        "layer " + layer + " quantized the feed-forward's activation",
                        tasks.contains("ffn_xb_quantize"));
                assertTrue(
                        "layer " + layer + " lost the attention norm's apply",
                        tasks.contains("attn_rms_apply"));
                assertTrue(
                        "layer " + layer + " lost the feed-forward norm's apply",
                        tasks.contains("ffn_rms_apply"));
            }
        }
    }

    /** The packed set: the branch's five, and the feed-forward against its own quantization. */
    private static Set<String> expected() {
        Set<String> names =
                new LinkedHashSet<>(
                        Set.of(
                                "attn_q_proj",
                                "attn_k_proj",
                                "attn_v_proj",
                                "ssm_qkv_proj",
                                "ssm_gate_proj"));
        names.add("ffn_gate_up");
        // ffn_down, against its own quantization of the SwiGLU output.
        names.add("ffn_down_proj");
        if (packedPathEnabled()) {
            // ssm_out, against its own quantization of the delta-net readout.
            names.add("ssm_out_proj");
        }
        return names;
    }

    // ---- batched prefill: which projections reach the tensor cores ----------

    /** The batch width the batched-prefill plan is built for here; a whole number of MMA rows. */
    private static final int PREFILL_BATCH = 32;

    private static Qwen35BatchPrefillLayers buildBatched(Qwen35Configuration config) {
        return buildBatched(config, PREFILL_BATCH);
    }

    private static Qwen35BatchPrefillLayers buildBatched(Qwen35Configuration config, int width) {
        String previousDevice = System.getProperty("use.tornadovm");
        String previousCores = System.getProperty("jitllm.qwen35.tensorCores");
        System.setProperty("use.tornadovm", "true");
        // Read into a static final when Qwen35BatchPrefillLayers first loads, which is here: no
        // test above this one touches that class.
        System.setProperty("jitllm.qwen35.tensorCores", "true");
        try {
            Qwen35State state =
                    (Qwen35State)
                            State.withPrefillBatchSize(width, () -> new Qwen35State(config, -1));
            return new Qwen35BatchPrefillLayers(state, weights(config), config, width);
        } finally {
            restore("use.tornadovm", previousDevice);
            restore("jitllm.qwen35.tensorCores", previousCores);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    /** The tasks only the tensor-core branch of a batched layer adds. */
    private static List<String> batchTaskNames(GridScheduler scheduler, int layer) {
        List<String> names = new ArrayList<>();
        for (String key : scheduler.keySet()) {
            if (key.startsWith("batchLayer_" + layer + ".")) {
                names.add(key.substring(key.indexOf('.') + 1));
            }
        }
        return names;
    }

    // @formatter:off
    /**
     * The {@code ffn_down} of <b>every</b> block reaches the tensor cores, in both of the
     * representations this family holds it in.
     *
     * <p>It did not. The condition that opened the branch asked for {@code Q4_0} while the choice
     * of kernel inside it asked whether the tensor was {@code Q4_1}, so the Q4_1 blocks — the first
     * eight of the 27B — took the scalar path and the Q4_1 tensor-core kernel was unreachable. A
     * kernel test could not see it: the kernel was correct, and nothing dispatched to it. This
     * asserts the production dispatch instead, on the same mixed model the rest of this class uses.
     *
     * <p>{@code ffn_down_residual} is the marker. The scalar path folds the residual into its
     * kernel; only the tensor-core branch, whose store overwrites, adds it as a pass of its own, so
     * the task exists exactly when the projection is on the tensor cores. (The FP16 conversion
     * {@code ffn_down_fp16} was the marker before SwiGLU learned to write the FP16 buffer itself
     * when the gate/up projections are on the tensor cores too.)
     */
    // @formatter:on
    @Test
    public void everyFfnDownReachesTheTensorCoresWhateverItsRepresentation() {
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        Qwen35Configuration config = config();
        GridScheduler scheduler = new GridScheduler();
        buildBatched(config).updateGridScheduler(scheduler);

        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            DataType representation = layer < 2 ? DataType.Q4_1 : DataType.Q4_0;
            List<String> tasks = batchTaskNames(scheduler, layer);
            assertTrue(
                    "layer "
                            + layer
                            + " holds its ffn_down as "
                            + representation
                            + " and did not take the tensor-core path; its tasks are "
                            + tasks,
                    tasks.contains("ffn_down_residual"));
            assertFalse(
                    "layer " + layer + " converts hb although SwiGLU wrote it as FP16",
                    tasks.contains("ffn_down_fp16"));
        }
    }

    // @formatter:off
    /**
     * The batched feed-forward projection is two single-panel tensor-core tasks, not one fused
     * two-panel task.
     *
     * <p>What this pins is the dispatch and the grids, which a kernel test cannot see: both tasks
     * present in every batched layer, the fused task gone, the SwiGLU that consumes their two
     * buffers still there, and each projection launched on the MMA geometry — one warp per {@code
     * BM x BN} output tile — rather than on the scalar matrix-vector grid it would get if it were
     * missing from the tensor-core task map.
     */
    // @formatter:on
    @Test
    public void theBatchedFeedForwardProjectsGateAndUpSeparately() {
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        Qwen35Configuration config = config();
        GridScheduler scheduler = new GridScheduler();
        buildBatched(config).updateGridScheduler(scheduler);

        long expectedGlobal =
                (long) (PREFILL_BATCH / Qwen35MMAKernels.BM)
                        * (config.hiddenDim() / Qwen35MMAKernels.BN)
                        * Qwen35MMAKernels.LOCAL;
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = batchTaskNames(scheduler, layer);
            assertTrue(
                    "layer " + layer + " lost the gate projection: " + tasks,
                    tasks.contains("ffn_gate_proj"));
            assertTrue(
                    "layer " + layer + " lost the up projection: " + tasks,
                    tasks.contains("ffn_up_proj"));
            assertTrue(
                    "layer " + layer + " lost the SwiGLU that joins them",
                    tasks.contains("ffn_swiglu"));
            assertFalse(
                    "layer " + layer + " still dispatches the fused two-panel projection",
                    tasks.contains("ffn_gate_up"));

            for (String task : new String[] {"ffn_gate_proj", "ffn_up_proj"}) {
                WorkerGrid grid = scheduler.get("batchLayer_" + layer + "." + task);
                assertEquals(
                        "layer " + layer + " " + task + " global work",
                        expectedGlobal,
                        grid.getGlobalWork()[0]);
                assertEquals(
                        "layer " + layer + " " + task + " local work",
                        Qwen35MMAKernels.LOCAL,
                        grid.getLocalWork()[0]);
            }
        }
    }

    // @formatter:off
    /**
     * Which batched projections reach the tensor cores at each width, and which do not.
     *
     * <p>Two things this pins that no parity test does. First, **the selection is per width**:
     * `mmaEligible` needs the chunk to fill whole 16-row MMA tiles, so 32 and 64 take the
     * tensor-core path and a width of 8 cannot — the scalar tiled kernels stay reachable and are
     * still the only path at an ineligible width. Second, the plan really contains the MMA tasks
     * and their one-warp-per-tile grids, rather than a property having been set somewhere.
     *
     * <p>{@code attn_output_proj} was asserted to be on the **scalar** path, because it folds a
     * residual. It now takes the same three-task shape `ffn_down` and `ssm_out` take — convert,
     * project, add back — and this case asserts that shape and its grid instead.
     */
    // @formatter:on
    @Test
    public void theBatchedPlanSelectsTensorCoresPerWidth() {
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        Qwen35Configuration config = config();

        for (int width : new int[] {32, 64}) {
            GridScheduler scheduler = new GridScheduler();
            buildBatched(config, width).updateGridScheduler(scheduler);

            long expectedProjection =
                    (long) (width / Qwen35MMAKernels.BM)
                            * (config.hiddenDim() / Qwen35MMAKernels.BN)
                            * Qwen35MMAKernels.LOCAL;
            for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
                List<String> tasks = batchTaskNames(scheduler, layer);
                for (String task :
                        new String[] {
                            "ffn_gate_proj",
                            "ffn_up_proj",
                            "ffn_swiglu",
                            "ffn_down_proj",
                            "ffn_down_residual"
                        }) {
                    assertTrue(
                            "width "
                                    + width
                                    + " layer "
                                    + layer
                                    + " is missing "
                                    + task
                                    + ": "
                                    + tasks,
                            tasks.contains(task));
                }
                assertFalse(
                        "width " + width + " layer " + layer + " kept the fused gate/up task",
                        tasks.contains("ffn_gate_up"));

                for (String task : new String[] {"ffn_gate_proj", "ffn_up_proj"}) {
                    WorkerGrid grid = scheduler.get("batchLayer_" + layer + "." + task);
                    assertEquals(
                            "width " + width + " " + task + " global work",
                            expectedProjection,
                            grid.getGlobalWork()[0]);
                    assertEquals(
                            "width " + width + " " + task + " local work",
                            Qwen35MMAKernels.LOCAL,
                            grid.getLocalWork()[0]);
                }

                if (config.isRecurrentLayer(layer)) {
                    assertTrue(
                            "width " + width + " layer " + layer + " lost the Q5_K readout path",
                            tasks.contains("ssm_out_fp16")
                                    && tasks.contains("ssm_out_proj")
                                    && tasks.contains("ssm_out_residual"));
                } else {
                    // The attention branch's output projection folds a residual, so it takes the
                    // same three-task shape ffn_down and ssm_out do: convert, project, add back.
                    assertTrue(
                            "width "
                                    + width
                                    + " layer "
                                    + layer
                                    + " lost the attention output"
                                    + " projection's tensor-core tasks: "
                                    + tasks,
                            tasks.contains("attn_output_fp16")
                                    && tasks.contains("attn_output_proj")
                                    && tasks.contains("attn_output_residual"));
                    WorkerGrid grid = scheduler.get("batchLayer_" + layer + ".attn_output_proj");
                    assertEquals(
                            "width " + width + " attn_output_proj global work",
                            (long) (width / Qwen35MMAKernels.BM)
                                    * (config.dim() / Qwen35MMAKernels.BN)
                                    * Qwen35MMAKernels.LOCAL,
                            grid.getGlobalWork()[0]);
                }
            }
        }

        // An ineligible width: whole 16-row tiles are the condition, and 8 does not fill one.
        GridScheduler scalar = new GridScheduler();
        buildBatched(config, 8).updateGridScheduler(scalar);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = batchTaskNames(scalar, layer);
            assertTrue(
                    "width 8 layer " + layer + " must keep the fused scalar gate/up: " + tasks,
                    tasks.contains("ffn_gate_up"));
            assertFalse(
                    "width 8 layer " + layer + " reached the tensor cores",
                    tasks.contains("ffn_gate_proj")
                            || tasks.contains("ffn_down_fp16")
                            || tasks.contains("ffn_down_residual")
                            || tasks.contains("attn_output_fp16"));
        }
    }

    // @formatter:off
    /**
     * Which Q4_0 projections take the dequantize-then-GEMM pair, and at which widths.
     *
     * <p>The pair needs the chunk to fill whole 128-row GEMM tiles and the projection to be at
     * least 5,120 outputs wide, so with a 5,120-wide feed-forward: at width 128 the gate and up
     * projections gain a {@code _dequant} task and the GEMM's two-dimensional grid, the 256-wide
     * down projection stays on the direct kernel, and the scratch exists; at width 64 nothing
     * changes and no scratch is allocated. The direct kernel's grid is what a projection left off
     * the pair would carry, so the grid is asserted, not just the task name.
     */
    // @formatter:on
    @Test
    public void theBatchedPlanSelectsDequantizeThenGemmPerWidthAndWidthOfOutput() {
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        Qwen35Configuration wide = config(TRUNK_LAYERS, 5120);
        assertTrue(Qwen35Configuration.dequantGemmWidth(128));
        assertFalse(Qwen35Configuration.dequantGemmWidth(64));

        GridScheduler paired = new GridScheduler();
        buildBatched(wide, 128).updateGridScheduler(paired);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = batchTaskNames(paired, layer);
            for (String task : new String[] {"ffn_gate_proj", "ffn_up_proj"}) {
                assertTrue(
                        "width 128 layer "
                                + layer
                                + " "
                                + task
                                + " has no dequantization: "
                                + tasks,
                        tasks.contains(task + "_dequant"));
                WorkerGrid gemm = paired.get("batchLayer_" + layer + "." + task);
                assertEquals(
                        "width 128 " + task + " GEMM rows of work",
                        (128 / 128) * 256L,
                        gemm.getGlobalWork()[0]);
                assertEquals(
                        "width 128 " + task + " GEMM column tiles",
                        5120L / 128,
                        gemm.getGlobalWork()[1]);
                // The int8 decoder takes a lane per word of four weights; the FP16 decoder a
                // lane per packed byte of two. On a tensor-core device the Q4_0 pairs are int8.
                assertEquals(
                        "width 128 " + task + " dequantization lanes",
                        5120L * wide.dim() / 4,
                        paired.get("batchLayer_" + layer + "." + task + "_dequant")
                                .getGlobalWork()[0]);
            }
            assertFalse(
                    "width 128 layer " + layer + " put the 256-wide down projection on the pair",
                    tasks.contains("ffn_down_proj_dequant"));
            // On the pair the up GEMM writes silu(gate) * up itself: no SwiGLU task.
            assertFalse(
                    "width 128 layer " + layer + " kept a SwiGLU task beside the fused up GEMM",
                    tasks.contains("ffn_swiglu"));
        }

        GridScheduler direct = new GridScheduler();
        buildBatched(wide, 64).updateGridScheduler(direct);
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = batchTaskNames(direct, layer);
            assertFalse(
                    "width 64 layer " + layer + " took the pair: " + tasks,
                    tasks.contains("ffn_gate_proj_dequant"));
            assertEquals(
                    "width 64 ffn_gate_proj direct grid",
                    (64L / Qwen35MMAKernels.BM)
                            * (5120 / Qwen35MMAKernels.BN)
                            * Qwen35MMAKernels.LOCAL,
                    direct.get("batchLayer_" + layer + ".ffn_gate_proj").getGlobalWork()[0]);
        }
    }

    /**
     * The batched delta-rule scan's dispatch by value-head width, which is the width of the scan's
     * state: 128, this family's, takes the shared-state scan (32-lane groups); the fixture's 64
     * keeps the per-lane scan (128-lane groups). Same task name either way, so the grid is what is
     * asserted.
     */
    @Test
    public void theBatchedDeltaRuleScanIsSharedForThe128WideState() {
        // The expectation below is CUDA's warp scan. Off CUDA this family's batched plan is not
        // built at all on OpenCL (BatchPrefillSupport), and the grid it would take is untested.
        assumeTrue(
                "the warp-scan grid is CUDA's",
                org.beehive.jitllm.runtime.backend.BackendId.CUDA.equals(
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend()));
        for (int stateSize : new int[] {64, 128}) {
            Qwen35Configuration config = config(TRUNK_LAYERS, HIDDEN, stateSize);
            GridScheduler scheduler = new GridScheduler();
            buildBatched(config, PREFILL_BATCH).updateGridScheduler(scheduler);
            // On CUDA the 128-wide state takes the warp-per-column scan: a warp per column in
            // 128-lane groups; the 64-wide one keeps the per-lane scan in 128-lane groups.
            int expectedLocal = 128;
            long expectedGlobal = (long) VALUE_HEADS * stateSize * (stateSize == 128 ? 32 : 1);
            int found = 0;
            for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
                WorkerGrid grid = scheduler.get("batchLayer_" + layer + ".ssm_delta_rule");
                if (grid == null) {
                    continue;
                }
                assertEquals(
                        "state " + stateSize + " layer " + layer + " delta-rule global work",
                        expectedGlobal,
                        grid.getGlobalWork()[0]);
                assertEquals(
                        "state " + stateSize + " layer " + layer + " delta-rule local work",
                        (long) expectedLocal,
                        grid.getLocalWork()[0]);
                found++;
            }
            assertTrue("no batched delta-rule scan built for state " + stateSize, found > 0);
        }
    }

    // ---- the assertions -----------------------------------------------------

    /**
     * One graph per trunk layer, and the MTP block is not among them.
     *
     * <p>{@code numberOfLayers()} is the trunk; {@code numberOfBlocks()} counts the draft head with
     * it. Building the draft head into the generation plan would run it every token, for a
     * prediction nothing consumes.
     */
    @Test
    public void thereIsOneGraphPerTrunkLayerAndNoDraftHead() {
        Qwen35Configuration config = config();
        Qwen35FFNLayers layers = build(config);

        assertEquals(
                "one graph per trunk layer",
                TRUNK_LAYERS,
                layers.getFFNLayerImmutableTaskGraphs().size());
        assertEquals(
                "the last graph is the last trunk layer",
                "layer_7",
                layers.getLastFFNLayerTaskGraphID());

        GridScheduler scheduler = layers.updateGridScheduler(new GridScheduler());
        assertTrue(
                "the MTP block must not appear in ordinary generation",
                taskNames(scheduler, TRUNK_LAYERS).isEmpty());
    }

    /** Each layer kind builds its own mixer's tasks, and only those. */
    @Test
    public void eachBlockGetsTheMixerItsKindDeclares() {
        Qwen35Configuration config = config();
        GridScheduler scheduler = build(config).updateGridScheduler(new GridScheduler());

        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            boolean recurrent = config.isRecurrentLayer(layer);
            assertEquals(
                    "layer " + layer + " kind", (layer + 1) % ATTENTION_INTERVAL != 0, recurrent);

            assertEquals(
                    "layer " + layer + " delta-net tasks",
                    recurrent,
                    tasks.contains("ssm_delta_rule"));
            assertEquals("layer " + layer + " convolution", recurrent, tasks.contains("ssm_conv"));
            assertEquals("layer " + layer + " attention", !recurrent, tasks.contains("attention"));
            assertEquals("layer " + layer + " rotation", !recurrent, tasks.contains("attn_rope"));
            assertEquals(
                    "layer " + layer + " key/value append",
                    !recurrent,
                    tasks.contains("attn_kv_append"));

            // Both kinds run the same dense feed-forward and the same two normalizations.
            assertTrue("layer " + layer + " feed-forward", tasks.contains("ffn_gate_up"));
            assertTrue("layer " + layer + " down projection", tasks.contains("ffn_down_proj"));
            assertTrue("layer " + layer + " input norm", tasks.contains("attn_rms_reduce"));
            assertTrue("layer " + layer + " post-attention norm", tasks.contains("ffn_rms_reduce"));
        }
    }

    /**
     * The task count per layer kind, and the plan's total.
     *
     * <p>An exact number rather than a bound: this is the count that decides whether the real
     * model's plan is 1,300 tasks or 13,000, and a task silently gained or lost is a change in what
     * the layer computes.
     */
    @Test
    public void theTaskCountsAreTheOnesTheTopologyDeclares() {
        Qwen35Configuration config = config();
        GridScheduler scheduler = build(config).updateGridScheduler(new GridScheduler());

        int recurrentTasks = 0;
        int attentionTasks = 0;
        int total = 0;
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            int tasks = taskNames(scheduler, layer).size();
            total += tasks;
            if (config.isRecurrentLayer(layer)) {
                recurrentTasks = tasks;
            } else {
                attentionTasks = tasks;
            }
        }

        // The branch's and the feed-forward's quantizations add no task of their own: each is
        // carried by its norm's apply, which is counted in the base below either way. Only the
        // two that have no apply to ride on cost a task.
        int quantize = packedPathEnabled() ? 1 : 0;
        int ffnDownQuantize = quantize;
        // And one more in a recurrent layer: the delta-net readout's own quantization, for the
        // packed Q5_K ssm_out projection. Only that layer kind has a readout.
        int ssmOutQuantize = quantize;
        assertEquals(
                "a recurrent layer's tasks", 20 + ffnDownQuantize + ssmOutQuantize, recurrentTasks);
        assertEquals("an attention layer's tasks", 16 + ffnDownQuantize, attentionTasks);
        assertEquals("the plan's layer tasks", 6 * recurrentTasks + 2 * attentionTasks, total);
    }

    /** Every task name is unique within its graph, which is what the scheduler keys on. */
    @Test
    public void everyTaskNameIsUniqueWithinItsGraph() {
        GridScheduler scheduler = build(config()).updateGridScheduler(new GridScheduler());
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            Set<String> unique = new LinkedHashSet<>(tasks);
            assertEquals(
                    "layer " + layer + " has a duplicate task name", tasks.size(), unique.size());
        }
    }

    // @formatter:off
    /**
     * Every weight-reading task is bound to its own tensor's representation.
     *
     * <p>This is the assertion that a mixed model needs and a uniform one does not. Three roles in
     * this synthetic model are deliberately not the model-wide Q4_0 — the early {@code ffn_down}
     * are Q4_1 and every {@code ssm_out} is Q5_K — and a plan that read them as Q4_0 would decode
     * 20-byte and 176-byte blocks as 18-byte ones, producing weights of plausible magnitude.
     */
    // @formatter:on
    @Test
    public void everyTaskBindsItsOwnTensorsRepresentation() {
        Qwen35Configuration config = config();
        Qwen35FFNLayers layers = build(config);

        for (Qwen35FFNLayers.Dispatch dispatch : layers.dispatchInventory()) {
            DataType expected =
                    switch (dispatch.role()) {
                        case "ffn_down" -> dispatch.layer() < 2 ? DataType.Q4_1 : DataType.Q4_0;
                        case "ssm_out" -> DataType.Q5_K;
                        case "ssm_alpha", "ssm_beta" -> DataType.F32;
                        default -> DataType.Q4_0;
                    };
            assertEquals(
                    "layer "
                            + dispatch.layer()
                            + " task "
                            + dispatch.task()
                            + " reads "
                            + dispatch.role(),
                    expected,
                    dispatch.representation());
        }

        Set<DataType> bound = new LinkedHashSet<>();
        layers.dispatchInventory().forEach(d -> bound.add(d.representation()));
        assertTrue("the Q4_1 down projections must be read as Q4_1", bound.contains(DataType.Q4_1));
        assertTrue(
                "the Q5_K recurrent outputs must be read as Q5_K", bound.contains(DataType.Q5_K));
        assertTrue("the F32 SSM projections must be read as F32", bound.contains(DataType.F32));
        assertFalse(
                "nothing here is materialized as Q8_0 to find a kernel",
                bound.contains(DataType.Q8_0));
    }

    /**
     * Changing one tensor's representation changes what the plan compiles.
     *
     * <p>The legacy path has no compiled-program cache to key, so the identity that matters is the
     * one the graphs themselves carry: which kernel each task was bound to. If that did not change
     * with the tensor, then either the dispatch is not per tensor or a conversion is hiding one.
     */
    @Test
    public void aChangedRepresentationChangesTheDispatch() {
        Qwen35Configuration config = config();
        List<Qwen35FFNLayers.Dispatch> before = build(config).dispatchInventory();

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        List<Qwen35FFNLayers.Dispatch> after;
        try {
            Qwen35State state = new Qwen35State(config, -1);
            Qwen35TornadoWeights mixed = weights(config);
            // One recurrent block's output projection, re-quantized.
            mixed.ssmOut[0] = blocked(DataType.Q4_0, config.deltaNetValueDim() * DIM);
            after =
                    new Qwen35FFNLayers("qwen35FFN", state, mixed, config, SchedulerType.NVIDIA)
                            .dispatchInventory();
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }

        assertEquals("the same tasks either way", before.size(), after.size());
        assertFalse("the dispatch must not be identical", before.equals(after));
        for (int i = 0; i < before.size(); i++) {
            boolean isChangedTensor =
                    before.get(i).layer() == 0 && before.get(i).role().equals("ssm_out");
            if (isChangedTensor) {
                assertEquals(DataType.Q5_K, before.get(i).representation());
                assertEquals(DataType.Q4_0, after.get(i).representation());
            } else {
                assertEquals("only the changed tensor's task changed", before.get(i), after.get(i));
            }
        }
    }
}
