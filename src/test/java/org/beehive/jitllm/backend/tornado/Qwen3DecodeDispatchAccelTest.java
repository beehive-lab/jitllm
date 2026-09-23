package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.Options;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.scheduling.Fp16GemvReductionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;

// @formatter:off
/**
 * The decode half of a Qwen3 FP16 batched plan must be built the way its measurements were taken.
 *
 * <p>Two decode changes carry performance claims, and a build that quietly stopped selecting either
 * would keep generating correct text while being a fifth to a quarter slower. Neither is visible in
 * the output, so neither is caught by a correctness test.
 *
 * <ul>
 *   <li><b>Layer grouping.</b> Twenty-eight layers in seven graphs, not twenty-eight. Each graph
 *       submission ends in a device wait, and at depth zero those waits were 810 µs of a 3.49 ms
 *       token.
 *   <li><b>The warp-butterfly reduction.</b> Four matrix-vector kernels per layer and the
 *       vocabulary projection have shuffle-reducing twins that {@link Fp16GemvReductionPolicy}
 *       selects.
 * </ul>
 *
 * <h2>Skip, or fail</h2>
 *
 * <p>This test <b>fails</b> rather than skips whenever it is on the configuration the decode
 * numbers were taken on: the CUDA backend with the Qwen3-0.6B FP16 fixture present. Revoking {@code
 * SHUFFLE_REDUCED_FP16_GEMV}, or removing either selection, is a failure there and not a quiet
 * pass. It skips only for a genuinely unsupported environment — a missing fixture, or a backend
 * that is not the one this claim is about — and says which.
 *
 * <p>That distinction is the point. An earlier version assumed the capability was granted and then
 * asserted the same thing, so a build with the grant removed reported success by skipping.
 *
 * <h2>What is observed</h2>
 *
 * <p>Everything here is read off the built plan's own grid scheduler, which holds one entry per
 * task the plan contains:
 *
 * <ul>
 *   <li><b>Grouping</b> from the number of distinct {@code layer_<n>} graphs and the slot prefixes
 *       within them.
 *   <li><b>The four layer kernels</b> from their task names. The shuffle-reducing form is suffixed
 *       {@code _warp}, so each of the four is checked in each layer independently — not inferred
 *       from the vocabulary projection or from anything else.
 *   <li><b>The vocabulary projection</b> from its worker grid, which is the only thing that
 *       distinguishes its two kernels: one 32-lane workgroup per row against a scaled local size.
 * </ul>
 *
 * <p>The two are checked separately and both are required, so dropping the layer selection while
 * leaving the vocabulary selection in place fails here — which is a real defect shape, because the
 * two are chosen by different classes reading the same policy.
 *
 * <p>The grouping figure is asserted as a relationship, not as a literal seven, so a model with a
 * different layer count does not need this test edited.
 */
// @formatter:on
public class Qwen3DecodeDispatchAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String KV_FP16_PROPERTY = "jitllm.kvcache.fp16";
    private static final int BATCH = 128;
    private static final int CONTEXT = 512;

    /** The shipped grouping, read from the family rather than restated; see its javadoc for why. */
    private static final int EXPECTED_LAYERS_PER_GRAPH = Qwen3FP16FFNLayersDecode.LAYERS_PER_GRAPH;

    @Test
    public void decodeGroupsItsLayerGraphsAndReducesWithTheShuffle() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousKv = System.getProperty(KV_FP16_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(KV_FP16_PROPERTY, "true");
        try {
            // Resolving the device initializes the backend, so it has to happen with the GPU
            // property already set; that is why this is here and not beside the fixture check.
            BackendId backend = TornadoDevices.current().id().backend();
            boolean designatedConfiguration = BackendId.CUDA.equals(backend);
            assumeTrue(
                    "this runs on the "
                            + backend
                            + " backend; the decode selection this asserts is a claim about CUDA,"
                            + " where it was measured, so there is nothing to check here",
                    designatedConfiguration);

            Options options =
                    new Options(
                            model,
                            "dispatch",
                            null,
                            null,
                            false,
                            0.0f,
                            1.0f,
                            42,
                            CONTEXT,
                            false,
                            false,
                            true,
                            true,
                            BATCH);
            Model loaded = ModelLoader.loadModel(options);
            State state = loaded.createNewState();
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, loaded);
            try {
                var scheduler = PlanDispatchEvidence.gridSchedulerIfAvailable(plan);

                // --- grouping -------------------------------------------------------------
                PlanDispatchEvidence.DecodeGrouping grouping =
                        PlanDispatchEvidence.qwen3DecodeGrouping(scheduler);
                int layers = loaded.configuration().numberOfLayers();
                int expectedGraphs =
                        (layers + EXPECTED_LAYERS_PER_GRAPH - 1) / EXPECTED_LAYERS_PER_GRAPH;
                assertEquals(
                        "decode layer graphs: each one costs a submission and a device wait, and"
                                + " the grouping is what the decode measurements were taken with",
                        expectedGraphs,
                        grouping.layerGraphs());
                assertEquals(
                        "layers sharing one decode graph",
                        EXPECTED_LAYERS_PER_GRAPH,
                        grouping.layersPerGraph());
                assertTrue(
                        "grouping must reduce submissions below one graph per layer, or it is not"
                                + " doing the thing it was measured doing",
                        grouping.layerGraphs() < layers);

                // --- the policy is in force on the configuration it was measured on ---------
                assertTrue(
                        "this is the CUDA backend with the Qwen3 FP16 fixture present, which is"
                                + " the configuration the decode numbers were taken on, so the"
                                + " shuffle-reduced FP16 GEMV preference must hold here. It does not:"
                                + " either the capability grant or the policy has been changed",
                        Fp16GemvReductionPolicy.preferShuffleReduction());

                // --- the four layer kernels, each one, in each layer ------------------------
                PlanDispatchEvidence.LayerReduction reduction =
                        PlanDispatchEvidence.qwen3DecodeLayerReduction(scheduler);
                assertEquals(
                        "every decode layer slot must appear: " + reduction.describe(),
                        layers,
                        reduction.layers());
                assertTrue(
                        "the layer matrix-vector kernels did not all install their"
                                + " shuffle-reducing variants: "
                                + reduction.describe(),
                        reduction.allShuffleReduced());

                // --- the decode attention kernel, and the grid it must be paired with -------
                // Qwen3FP16FFNLayers passes config.numberOfHeadsValue() as nEmbdHead, which for
                // this family equals headSize(); asking through the Configuration interface keeps
                // the test off the concrete class.
                if (LaneAttentionPolicy.laneCooperativeAttention(
                        loaded.configuration().headSize())) {
                    PlanDispatchEvidence.DecodeAttention attention =
                            PlanDispatchEvidence.qwen3DecodeAttention(scheduler);
                    assertEquals(
                            "every decode layer must install the lane-cooperative attention kernel"
                                    + " on this configuration: "
                                    + attention.describe(),
                            layers,
                            attention.laneTasks());
                    assertEquals(
                            "no decode layer may be left on the per-key kernel: "
                                    + attention.describe(),
                            0,
                            attention.perKeyTasks());
                    assertEquals(
                            "the lane-cooperative kernel partitions a 128-wide head across 32 lanes"
                                    + " and strides the key range by warp, so its group must be exactly"
                                    + " WARPS_PER_GROUP warps; any other size computes a wrong answer"
                                    + " without failing",
                            LaneAttentionPolicy.WARPS_PER_GROUP * 32L,
                            attention.laneLocalWork());
                }

                // --- the vocabulary projection, which a different class selects -------------
                PlanDispatchEvidence.VocabularyProjection vocab =
                        PlanDispatchEvidence.qwen3VocabularyProjection(scheduler);
                assertTrue(
                        "logits.vocab_proj was built with local work "
                                + vocab.localWork()
                                + ", which is the shared-memory kernel's grid: the policy holds but"
                                + " the vocabulary projection did not select its shuffle-reducing"
                                + " twin",
                        vocab.shuffleReduced());
                assertEquals(
                        "the shuffle-reducing vocabulary kernel launches one 32-lane workgroup per"
                                + " vocabulary row",
                        (long) loaded.configuration().vocabularySize() * 32L,
                        vocab.globalWork());
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(KV_FP16_PROPERTY, previousKv);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
