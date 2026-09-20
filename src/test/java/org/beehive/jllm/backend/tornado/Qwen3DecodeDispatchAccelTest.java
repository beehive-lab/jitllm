package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.Options;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

// @formatter:off
/**
 * The decode half of a Qwen3 FP16 batched plan must be built the way its measurements were taken.
 *
 * <p>Two decode changes carry performance claims, and a build that quietly stopped selecting either
 * would keep generating correct text while being a quarter to a third slower. Neither is visible in
 * the output, so neither is caught by a correctness test.
 *
 * <ul>
 *   <li><b>Layer grouping.</b> Twenty-eight layers in seven graphs, not twenty-eight. Each graph
 *       submission ends in a device wait, and at depth zero those waits were 810 µs of a 3.49 ms
 *       token. Read off the grid scheduler, which keys every task by {@code graphName.taskName}.
 *   <li><b>The warp-butterfly reduction.</b> The vocabulary projection and four layer matrix-vector
 *       kernels have shuffle-reducing twins that {@code WARP_SHUFFLE_GEMV_FP16} selects. The
 *       witness is the vocabulary projection's worker grid; see below.
 * </ul>
 *
 * <h2>Why the vocabulary projection is the witness for all five kernels</h2>
 *
 * <p>The four layer kernels cannot be observed from a scheduler at all: their shuffle-reducing and
 * shared-memory forms share a task name <i>and</i> a worker grid, because both run on the same
 * 32-wide group. The vocabulary projection does not — {@code matrixVectorGenericSimd32} assumes
 * exactly one 32-lane workgroup per output row, while the shared-memory kernel scales the local
 * size by {@code THREAD_SCALE_FOR_LOGITS}. So {@code logits.vocab_proj} with a local size of 32 is
 * a fact about the built plan, not a restatement of a system property.
 *
 * <p>That fact then carries the other four, because both selections read the same capability.
 * {@code LogitsFP16Layer.useSimd32Reduction()} is {@code SUBGROUP_SHUFFLE_32 ||
 * WARP_SHUFFLE_GEMV_FP16}; {@code Qwen3FP16FFNLayers.useWarpMatmul} is {@code WARP_SHUFFLE ||
 * SUBGROUP_SHUFFLE_32 || WARP_SHUFFLE_GEMV_FP16}. The test asserts that neither of the other two
 * capabilities is granted here, so a 32-lane vocabulary grid can only have come from {@code
 * WARP_SHUFFLE_GEMV_FP16} — and that grant is a disjunct of {@code useWarpMatmul}, which is
 * therefore true in the same process. Revoking the capability breaks the assertion; so does
 * removing the disjunct from the logits layer.
 *
 * <p>On a device where one of the other two capabilities <i>is</i> granted (Metal has {@code
 * SUBGROUP_SHUFFLE_32}) the witness no longer isolates anything, and the reduction half is skipped
 * rather than asserted on weaker evidence.
 *
 * <p>The grouping figure is asserted as a relationship, not as a literal seven, so a model with a
 * different layer count does not need this test edited.
 */
// @formatter:on
public class Qwen3DecodeDispatchAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String KV_FP16_PROPERTY = "jllm.kvcache.fp16";
    private static final int BATCH = 128;
    private static final int CONTEXT = 512;

    /** What Qwen3FP16FFNLayersDecode.layersPerGraph() returns; see its javadoc for why four. */
    private static final int EXPECTED_LAYERS_PER_GRAPH = 4;

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

                assumeTrue(
                        "this device is not granted the shuffle-reducing matrix-vector kernels,"
                                + " so the reduction half of this test does not apply here",
                        SchedulerDetectionService.isWarpShuffleGemvFp16Supported());
                assumeTrue(
                        "another capability could also have selected the 32-lane vocabulary grid"
                                + " here, so it no longer isolates WARP_SHUFFLE_GEMV_FP16 and proves"
                                + " nothing about the layer kernels",
                        !SchedulerDetectionService.isWarpShuffleSupported()
                                && !SchedulerDetectionService.isSubgroupShuffle32Supported());

                PlanDispatchEvidence.VocabularyProjection vocab =
                        PlanDispatchEvidence.qwen3VocabularyProjection(scheduler);
                assertTrue(
                        "logits.vocab_proj was built with local work "
                                + vocab.localWork()
                                + ", which is the shared-memory kernel's grid: the capability is"
                                + " granted but the vocabulary projection did not select its"
                                + " shuffle-reducing twin",
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
