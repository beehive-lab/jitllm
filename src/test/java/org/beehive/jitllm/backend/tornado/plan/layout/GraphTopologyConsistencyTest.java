package org.beehive.jllm.backend.tornado.plan.layout;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.junit.Test;

/**
 * The memory model multiplies per-layer weight bytes by the number of layer graph families, because
 * the Tornado runtime holds object state per task graph and so allocates a device buffer per graph
 * that binds an array. Getting that count wrong under-predicts by roughly the size of the model —
 * measured at 1856 MiB for Llama-3.2-1B-F16 — and an under-prediction admits a load that then dies
 * part-allocated.
 *
 * <p><b>What this test is for.</b> The count used to be a literal in the memory model. A new mode,
 * or a layout that grew a second family, would have kept reporting {@code EXACT} while predicting
 * the memory of a different plan. Three things now prevent that, and this pins all three:
 *
 * <ol>
 *   <li>each layout declares its own family count beside its graph indices;
 *   <li>{@code totalGraphs == sum(layerFamilyGraphCounts) + nonLayerGraphs}, together with one
 *       count per declared family, is checked — so a layout cannot grow a family without either
 *       stating its graphs or failing here. A family need not build one graph per layer: grouping
 *       layers changes that term without changing the family count, which is what the memory model
 *       multiplies by;
 *   <li>{@link TornadoGraphTopology}'s switches have no {@code default}, so a new {@link
 *       ExecutionMode} does not compile until it answers.
 * </ol>
 */
public class GraphTopologyConsistencyTest {

    /** Every selectable topology's declared families agree with the graphs it lays out. */
    @Test
    public void everySelectableTopologyIsSelfConsistent() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            for (int layers : new int[] {1, 16, 32, 80}) {
                assertTrue(
                        mode
                                + " at "
                                + layers
                                + " layers: totalGraphs ("
                                + TornadoGraphTopology.totalGraphs(mode, layers)
                                + ") must equal the graphs its families contribute ("
                                + java.util.Arrays.toString(
                                        TornadoGraphTopology.layerFamilyGraphCounts(mode, layers))
                                + ") plus nonLayerGraphs ("
                                + TornadoGraphTopology.nonLayerGraphs(mode, layers)
                                + "). A layout that grew a layer family without updating its"
                                + " layerGraphFamilies() fails here rather than under-predicting"
                                + " memory silently.",
                        TornadoGraphTopology.verify(mode, layers));
            }
        }
        assertTrue(TornadoGraphTopology.verifyAll());
    }

    /**
     * The family counts themselves, pinned per mode.
     *
     * <p>Exact equalities rather than "batched is larger", because the number is the multiplier on
     * the largest component of the memory plan. It is also the measurement's own conclusion:
     * batched prefill costs 1872 MiB more than single-token against 1856 MiB of per-layer weights,
     * which is one extra family and not two.
     */
    @Test
    public void theFamilyCountsAreTheMeasuredOnes() {
        assertEquals(
                "single-token binds each layer's weights once",
                1,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.STANDARD, 16));
        assertEquals(
                "sequential prefill reuses the decode layer graphs, so also once",
                1,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.PREFILL_DECODE, 16));
        assertEquals(
                "batched prefill builds a second layer family and binds the weights twice",
                2,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.BATCH_PREFILL_DECODE, 16));
    }

    /**
     * Sequential prefill lays out the same number of graphs as single-token.
     *
     * <p>Recorded because it was got wrong once: an earlier note claimed sequential prefill had the
     * same 2N+3 graphs as batched prefill and explained its equal memory cost by the families
     * differing. The graph counts differ too — N+2 against 2N+3 — and the measurement (2365 MiB
     * against 2362) is consistent with the simpler explanation.
     */
    @Test
    public void sequentialPrefillLaysOutTheSameGraphsAsSingleToken() {
        assertEquals(
                TornadoGraphTopology.totalGraphs(ExecutionMode.STANDARD, 16),
                TornadoGraphTopology.totalGraphs(ExecutionMode.PREFILL_DECODE, 16));
        assertEquals(
                "batched prefill is the topology that differs",
                2 * 16 + 3,
                TornadoGraphTopology.totalGraphs(ExecutionMode.BATCH_PREFILL_DECODE, 16));
    }

    // @formatter:off
    /**
     * A layout that groups layers into fewer decode graphs stays self-consistent.
     *
     * <p>The decode family may hold more than one layer per graph, which costs one submission
     * instead of several. That breaks the old {@code families x N + nonLayerGraphs} identity — the
     * decode family no longer contributes {@code N} graphs — so the check is now against the graphs
     * each family says it contributes. This pins that the new form still catches a layout whose
     * totals do not add up, rather than having been weakened to accept anything.
     */
    // @formatter:on
    @Test
    public void aGroupedDecodeLayoutStaysSelfConsistent() {
        for (int layers : new int[] {1, 2, 7, 16, 64, 65}) {
            for (int group : new int[] {1, 2}) {
                int decodeGraphs = (layers + group - 1) / group;
                var layout = new BatchPrefillDecodeForwardTaskGraphLayout(layers, decodeGraphs);

                assertEquals(
                        "graphs each family contributes must sum with the non-layer graphs",
                        layers + decodeGraphs + layout.nonLayerGraphs(),
                        layout.totalGraphs());
                assertEquals(
                        "a term per family",
                        layout.layerGraphFamilies(),
                        layout.layerFamilyGraphCounts().length);
                assertEquals(
                        "the decode family contributes its graph count",
                        decodeGraphs,
                        layout.layerFamilyGraphCounts()[1]);
                assertEquals(
                        "logits sits after every decode graph",
                        layers + 2 + decodeGraphs,
                        layout.logitsIdx());
                assertEquals(
                        "the first decode graph follows the decode activation",
                        layout.decodeActivationIdx() + 1,
                        layout.decodeLayerGraphIdx(0));
                assertEquals(
                        "the last decode graph sits immediately before logits",
                        layout.logitsIdx() - 1,
                        layout.decodeLayerGraphIdx(decodeGraphs - 1));
                // The batch-prefill family and everything before decode are untouched by grouping.
                assertEquals(0, layout.batchActivationIdx());
                assertEquals(1, layout.batchLayerIdx(0));
                assertEquals(layers, layout.batchLayerIdx(layers - 1));
                assertEquals(layers + 1, layout.decodeActivationIdx());
                assertEquals(layers, layout.batchLayerGraphs());
            }
        }
    }

    /** An odd layer count leaves the final decode graph holding one layer, and still adds up. */
    @Test
    public void anOddLayerCountIsLaidOutExplicitly() {
        var layout = new BatchPrefillDecodeForwardTaskGraphLayout(65, 33);
        assertEquals("33 graphs for 65 layers paired", 33, layout.decodeLayerGraphs());
        assertEquals(65 + 33 + 3, layout.totalGraphs());
        assertEquals(65 + 2 + 33, layout.logitsIdx());
    }

    /** The ungrouped constructor is what every family still gets, and it is the old layout. */
    @Test
    public void theUngroupedLayoutIsUnchanged() {
        for (int layers : new int[] {1, 16, 32, 80}) {
            var layout = new BatchPrefillDecodeForwardTaskGraphLayout(layers);
            assertEquals(layers, layout.decodeLayerGraphs());
            assertEquals(2 * layers + 3, layout.totalGraphs());
            assertEquals(2 * layers + 2, layout.logitsIdx());
            assertEquals(layers + 2, layout.decodeLayerGraphIdx(0));
            assertTrue(
                    "every family builds a graph per layer here",
                    TornadoGraphTopology.isUngrouped(ExecutionMode.BATCH_PREFILL_DECODE, layers));
            assertTrue(TornadoGraphTopology.verify(ExecutionMode.BATCH_PREFILL_DECODE, layers));
        }
    }

    // @formatter:off
    /**
     * Grouping changes graphs, not weight memory.
     *
     * <p>The family count is what the memory model multiplies by, and it counts <b>families</b>,
     * not graphs. Grouping the decode family into half as many graphs must leave it at two, so that
     * a grouped plan is not predicted to hold less of the model than it does.
     */
    // @formatter:on
    @Test
    public void groupingDoesNotChangeTheFamilyCount() {
        assertEquals(2, new BatchPrefillDecodeForwardTaskGraphLayout(64, 64).layerGraphFamilies());
        assertEquals(
                "half as many decode graphs is still two families",
                2,
                new BatchPrefillDecodeForwardTaskGraphLayout(64, 32).layerGraphFamilies());
    }

    /**
     * Every mode is answered.
     *
     * <p>A guard against the switches gaining a {@code default} in a future tidy-up, which would
     * turn "a new mode does not compile" into "a new mode silently reports one family".
     */
    @Test
    public void everyModeIsAnswered() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            assertTrue(
                    mode + " must report at least one layer family",
                    TornadoGraphTopology.layerGraphFamilies(mode, 16) >= 1);
            assertTrue(
                    mode + " must lay out more graphs than it has layers",
                    TornadoGraphTopology.totalGraphs(mode, 16) > 16);
        }
        assertEquals(
                "ExecutionMode gained a constant; TornadoGraphTopology and the memory model"
                        + " must state its layer-family count",
                3,
                ExecutionMode.values().length);
    }
}
