package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanPipeline.StageGraphs;
import org.junit.Test;

/**
 * A pipeline stage's program positions with and without prepacking graphs ({@link StageGraphs}).
 *
 * <p>No device: each case takes the positions a stage builder records among the stage's inference
 * graphs (its programs, and the graph that binds its recurrent state), moves them past the
 * prepacking graphs with {@link StageGraphs#shifted}, and checks them against graph names laid out
 * as the stage assembles them, prepacking graphs first. Initialization and the programs together
 * must cover the list once.
 */
public class PipelineStageGraphsTest {

    private static List<String> assembled(int p, List<String> inference) {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < p; i++) {
            names.add("prepack_layer_" + (3 + i));
        }
        names.addAll(inference);
        return names;
    }

    private static List<String> named(List<String> names, int[] positions) {
        return IntStream.of(positions).mapToObj(names::get).toList();
    }

    private static void coversOnce(int p, int listSize, int[]... programs) {
        Set<Integer> covered = new TreeSet<>();
        IntStream.range(0, p).forEach(covered::add);
        for (int[] program : programs) {
            for (int i : program) {
                assertTrue("initialization and the programs are disjoint", covered.add(i));
            }
        }
        assertEquals(listSize, covered.size());
    }

    @Test
    public void aLaterSingleTokenStageWithAndWithoutPrepacking() {
        List<String> inference = List.of("activationUpdate", "layer_3", "layer_4", "handoff_out");
        // As addSingleTokenStage records them: every graph is the token program, and the first
        // layer graph, added after the receive, binds the recurrent state.
        StageGraphs built =
                new StageGraphs(null, IntStream.range(0, inference.size()).toArray(), 1);
        for (int p : new int[] {0, 2}) {
            List<String> names = assembled(p, inference);
            StageGraphs graphs = built.shifted(p);

            assertNull(graphs.prefill());
            assertEquals(inference, named(names, graphs.token()));
            assertEquals("layer_3", names.get(graphs.recurrentStateGraph()));
            coversOnce(p, names.size(), graphs.token());
        }
    }

    @Test
    public void aBatchedStageWithAndWithoutPrepacking() {
        List<String> prefill =
                List.of(
                        "prefillActivation",
                        "batchPrefillLayer_3",
                        "batchPrefillLayer_4",
                        "prefillHandoff");
        List<String> decode = List.of("decodeActivation", "layer_3", "layer_4", "decodeHandoff");
        List<String> inference = new ArrayList<>(prefill);
        inference.addAll(decode);
        // As addBatchedStage records them: the prefill program, then the decode program, and its
        // first batch-prefill layer graph binds the recurrent state.
        StageGraphs built =
                new StageGraphs(
                        IntStream.range(0, prefill.size()).toArray(),
                        IntStream.range(prefill.size(), inference.size()).toArray(),
                        1);
        for (int p : new int[] {0, 2}) {
            List<String> names = assembled(p, inference);
            StageGraphs graphs = built.shifted(p);

            assertEquals(prefill, named(names, graphs.prefill()));
            assertEquals(decode, named(names, graphs.token()));
            assertEquals("batchPrefillLayer_3", names.get(graphs.recurrentStateGraph()));
            coversOnce(p, names.size(), graphs.prefill(), graphs.token());
        }
    }

    @Test
    public void theRecurrentStateOwnerIsTheRecordedGraphNotAFixedPosition() {
        // A stage whose builder added two graphs before its first layer graph.
        List<String> inference = List.of("receive", "unpack", "layer_3", "handoff_out");
        StageGraphs built =
                new StageGraphs(null, IntStream.range(0, inference.size()).toArray(), 2);
        for (int p : new int[] {0, 3}) {
            List<String> names = assembled(p, inference);
            assertEquals("layer_3", names.get(built.shifted(p).recurrentStateGraph()));
        }
    }
}
