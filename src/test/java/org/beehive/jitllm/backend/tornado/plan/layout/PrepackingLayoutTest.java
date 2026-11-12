package org.beehive.jitllm.backend.tornado.plan.layout;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.Test;

/**
 * The forward-plan layouts with and without prepacking graphs: initialization first, then the
 * inference graphs, every index computed by the layout.
 *
 * <p>No device: each case lays out graph names in the order the forward plan assembles its graphs
 * (prepacking graphs, then the inference graphs) and checks that the layout's indices select the
 * intended names, and that the initialization and inference ranges are disjoint and cover the list.
 */
public class PrepackingLayoutTest {

    private static final int N = 4;

    /**
     * {@code p} prepacking graph names, then {@code inference}, as a forward plan assembles them.
     */
    private static List<String> assembled(int p, List<String> inference) {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < p; i++) {
            names.add("prepack_layer_" + i);
        }
        names.addAll(inference);
        return names;
    }

    private static List<String> singleTokenInference() {
        List<String> names = new ArrayList<>();
        names.add("activation");
        IntStream.range(0, N).forEach(i -> names.add("layer_" + i));
        names.add("logits");
        return names;
    }

    /** The initialization range is [0, first) and holds exactly the prepacking graphs. */
    private static void checkRanges(
            List<String> names, int p, int first, int inference, int total) {
        assertEquals(names.size(), total);
        assertEquals(p, first);
        assertEquals(total - p, inference);
        for (int i = 0; i < total; i++) {
            boolean initialization = i < first;
            assertEquals(
                    "graph " + i + " (" + names.get(i) + ") is initialization iff prepacking",
                    initialization,
                    names.get(i).startsWith("prepack_"));
        }
    }

    private static void checkSelection(List<String> names, List<Integer> selected, int first) {
        for (int i : selected) {
            assertTrue("an inference step selects only inference graphs", i >= first);
            assertFalse(names.get(i).startsWith("prepack_"));
        }
    }

    @Test
    public void singleTokenWithAndWithoutPrepacking() {
        for (int p : new int[] {0, 3}) {
            SingleTokenForwardTaskGraphLayout layout = new SingleTokenForwardTaskGraphLayout(N, p);
            List<String> names = assembled(p, singleTokenInference());
            checkRanges(
                    names,
                    p,
                    layout.firstInferenceIdx(),
                    layout.inferenceGraphCount(),
                    layout.totalGraphs());
            for (int i = 0; i < p; i++) {
                assertEquals("prepack_layer_" + i, names.get(layout.prepackingIdx(i)));
            }
            assertEquals("activation", names.get(layout.activationIdx()));
            for (int i = 0; i < N; i++) {
                assertEquals("layer_" + i, names.get(layout.layerIdx(i)));
            }
            assertEquals("logits", names.get(layout.logitsIdx()));
            assertEquals(
                    "the logits are the last graph", layout.totalGraphs() - 1, layout.logitsIdx());
            List<Integer> token = new ArrayList<>();
            token.add(layout.activationIdx());
            IntStream.range(0, N).forEach(i -> token.add(layout.layerIdx(i)));
            token.add(layout.logitsIdx());
            checkSelection(names, token, layout.firstInferenceIdx());
            assertEquals(
                    "the recurrent state goes to the first layer graph",
                    "layer_0",
                    names.get(layout.layerIdx(0)));
        }
    }

    @Test
    public void prefillDecodeWithAndWithoutPrepacking() {
        for (int p : new int[] {0, 3}) {
            PrefillDecodeForwardTaskGraphLayout layout =
                    new PrefillDecodeForwardTaskGraphLayout(N, p);
            List<String> names = assembled(p, singleTokenInference());
            checkRanges(
                    names,
                    p,
                    layout.firstInferenceIdx(),
                    layout.inferenceGraphCount(),
                    layout.totalGraphs());
            assertEquals("activation", names.get(layout.activationIdx()));
            assertEquals("layer_" + (N - 1), names.get(layout.layerIdx(N - 1)));
            assertEquals("logits", names.get(layout.logitsIdx()));
            // The warm-up runs every inference graph, from the first to the logits.
            List<Integer> warmUp =
                    IntStream.rangeClosed(layout.firstInferenceIdx(), layout.logitsIdx())
                            .boxed()
                            .toList();
            assertEquals(layout.inferenceGraphCount(), warmUp.size());
            checkSelection(names, warmUp, layout.firstInferenceIdx());
        }
    }

    @Test
    public void batchPrefillDecodeWithAndWithoutPrepacking() {
        int fallback = 2;
        int decode = 3;
        for (int p : new int[] {0, 3}) {
            BatchPrefillDecodeForwardTaskGraphLayout layout =
                    new BatchPrefillDecodeForwardTaskGraphLayout(N, N, fallback, decode, p);
            List<String> inference = new ArrayList<>();
            inference.add("batchPrefillActivation");
            IntStream.range(0, N).forEach(i -> inference.add("batchPrefillLayer_" + i));
            IntStream.range(0, fallback).forEach(g -> inference.add("fallbackLayer_" + g));
            inference.add("decodeActivation");
            IntStream.range(0, decode).forEach(g -> inference.add("layer_" + g));
            inference.add("logits");
            List<String> names = assembled(p, inference);
            checkRanges(
                    names,
                    p,
                    layout.firstInferenceIdx(),
                    layout.inferenceGraphCount(),
                    layout.totalGraphs());
            assertEquals("batchPrefillActivation", names.get(layout.batchActivationIdx()));
            assertEquals("batchPrefillLayer_" + (N - 1), names.get(layout.batchLayerIdx(N - 1)));
            assertEquals("fallbackLayer_0", names.get(layout.fallbackLayerIdx(0)));
            assertEquals("decodeActivation", names.get(layout.decodeActivationIdx()));
            assertEquals(
                    "layer_" + (decode - 1), names.get(layout.decodeLayerGraphIdx(decode - 1)));
            assertEquals("logits", names.get(layout.logitsIdx()));
            List<Integer> prefill = new ArrayList<>();
            prefill.add(layout.batchActivationIdx());
            IntStream.range(0, N).forEach(i -> prefill.add(layout.batchLayerIdx(i)));
            List<Integer> decodeToken = new ArrayList<>();
            decodeToken.add(layout.decodeActivationIdx());
            IntStream.range(0, decode).forEach(g -> decodeToken.add(layout.decodeLayerGraphIdx(g)));
            decodeToken.add(layout.logitsIdx());
            checkSelection(names, prefill, layout.firstInferenceIdx());
            checkSelection(names, decodeToken, layout.firstInferenceIdx());
        }
    }

    @Test
    public void aLayoutWithoutPrepackingIsTheOneItAlwaysWas() {
        assertEquals(
                new SingleTokenForwardTaskGraphLayout(N, 0),
                new SingleTokenForwardTaskGraphLayout(N));
        assertEquals(0, new SingleTokenForwardTaskGraphLayout(N).activationIdx());
        assertEquals(N + 1, new SingleTokenForwardTaskGraphLayout(N).logitsIdx());
        assertEquals(0, new BatchPrefillDecodeForwardTaskGraphLayout(N).batchActivationIdx());
    }
}
