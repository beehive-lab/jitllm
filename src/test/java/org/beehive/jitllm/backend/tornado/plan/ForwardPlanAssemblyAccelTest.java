package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The three forward plans as production assembles them, from lightweight components whose layer
 * graphs register weights with a real {@link WeightPrepacker}: without prepacking and with it.
 *
 * <p>Builds the graph lists only; nothing executes, so it needs the TornadoVM runtime but no
 * particular device. For each plan it checks that the list is as long as the layout says, that the
 * prepacking graphs are exactly the initialization prefix, that every role index names the graph
 * the component built for it, and that the warm-up and the steps select no prepacking graph.
 */
public class ForwardPlanAssemblyAccelTest {

    private static final int N = 3;

    @Test
    public void singleTokenPlan() throws Exception {
        for (boolean prepack : new boolean[] {false, true}) {
            Components components = new Components(prepack);
            SingleTokenForwardPlan plan =
                    new SingleTokenForwardPlan(model(), components, components.prepacker);
            SingleTokenForwardTaskGraphLayout layout = plan.getTaskGraphLayout();
            List<String> names = names(plan.getImmutableTaskGraphs());

            int p = prepack ? N : 0;
            checkPrefix(names, p, layout.firstInferenceIdx(), layout.totalGraphs(), plan);
            assertEquals(layout.totalGraphs() - p, layout.inferenceGraphCount());
            assertEquals("activation", names.get(layout.activationIdx()));
            for (int i = 0; i < N; i++) {
                assertEquals("layer_" + i, names.get(layout.layerIdx(i)));
                if (prepack) {
                    assertEquals("prepack_layer_" + i, names.get(layout.prepackingIdx(i)));
                }
            }
            assertEquals("logits", names.get(layout.logitsIdx()));
            checkNoPrepacking(
                    names, IntStream.rangeClosed(layout.activationIdx(), layout.logitsIdx()));
        }
    }

    @Test
    public void prefillDecodePlan() throws Exception {
        for (boolean prepack : new boolean[] {false, true}) {
            Components components = new Components(prepack);
            PrefillDecodeForwardPlan plan =
                    new PrefillDecodeForwardPlan(model(), components, components.prepacker);
            PrefillDecodeForwardTaskGraphLayout layout = plan.getTaskGraphLayout();
            List<String> names = names(plan.getImmutableTaskGraphs());

            int p = prepack ? N : 0;
            checkPrefix(names, p, layout.firstInferenceIdx(), layout.totalGraphs(), plan);
            assertEquals("activation", names.get(layout.activationIdx()));
            assertEquals("layer_" + (N - 1), names.get(layout.layerIdx(N - 1)));
            assertEquals("logits", names.get(layout.logitsIdx()));
            // The warm-up: from the first inference graph to the logits.
            checkNoPrepacking(
                    names, IntStream.rangeClosed(layout.firstInferenceIdx(), layout.logitsIdx()));
        }
    }

    @Test
    public void batchPrefillDecodePlan() throws Exception {
        for (boolean prepack : new boolean[] {false, true}) {
            Components components = new Components(prepack);
            BatchPrefillDecodeForwardPlan plan =
                    new BatchPrefillDecodeForwardPlan(
                            model(), components, 128, components.prepacker);
            BatchPrefillDecodeForwardTaskGraphLayout layout = plan.getTaskGraphLayout();
            List<String> names = names(plan.getImmutableTaskGraphs());

            int p = prepack ? N : 0;
            checkPrefix(names, p, layout.firstInferenceIdx(), layout.totalGraphs(), plan);
            assertEquals("batchPrefillActivation", names.get(layout.batchActivationIdx()));
            for (int i = 0; i < N; i++) {
                assertEquals("batchPrefillLayer_" + i, names.get(layout.batchLayerIdx(i)));
                if (prepack) {
                    assertEquals(
                            "prepack_batchPrefillLayer_" + i, names.get(layout.prepackingIdx(i)));
                }
            }
            assertEquals("decodeActivation", names.get(layout.decodeActivationIdx()));
            assertEquals("layer_" + (N - 1), names.get(layout.decodeLayerGraphIdx(N - 1)));
            assertEquals("logits", names.get(layout.logitsIdx()));
            checkNoPrepacking(
                    names, IntStream.rangeClosed(layout.firstInferenceIdx(), layout.logitsIdx()));
        }
    }

    /** The list holds the layout's graphs, and exactly the prepacking ones come first. */
    private static void checkPrefix(
            List<String> names, int p, int firstInference, int totalGraphs, ForwardPlan plan) {
        assertEquals("the layout counts every graph in the list", names.size(), totalGraphs);
        assertEquals(p, firstInference);
        assertEquals(p, plan.getWeightPrepacking().graphCount());
        for (int i = 0; i < names.size(); i++) {
            assertEquals(
                    "graph " + i + " (" + names.get(i) + ") is initialization iff it prepacks",
                    i < p,
                    names.get(i).startsWith("prepack_"));
        }
    }

    private static void checkNoPrepacking(List<String> names, IntStream selected) {
        selected.forEach(
                i -> assertFalse("an inference selection", names.get(i).startsWith("prepack_")));
    }

    private static List<String> names(List<ImmutableTaskGraph> graphs) throws Exception {
        Field field = ImmutableTaskGraph.class.getDeclaredField("taskGraph");
        field.setAccessible(true);
        List<String> names = new ArrayList<>();
        for (ImmutableTaskGraph g : graphs) {
            names.add(((TaskGraph) field.get(g)).getTaskGraphName());
        }
        return names;
    }

    private static Model model() {
        Configuration config =
                (Configuration)
                        Proxy.newProxyInstance(
                                Configuration.class.getClassLoader(),
                                new Class<?>[] {Configuration.class},
                                (proxy, method, args) -> {
                                    if (method.getName().equals("numberOfLayers")) {
                                        return N;
                                    }
                                    throw new UnsupportedOperationException(method.getName());
                                });
        return (Model)
                Proxy.newProxyInstance(
                        Model.class.getClassLoader(),
                        new Class<?>[] {Model.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("configuration")) {
                                return config;
                            }
                            throw new UnsupportedOperationException(method.getName());
                        });
    }

    private static TornadoTensor q4() {
        return new Q4_0TornadoTensor(new ByteArray(128 * 64 / 32 * 18), Shape.of(64, 128));
    }

    private static TornadoWeights weights() {
        TornadoTensor[] layer = {q4()};
        TornadoTensor f32 = new FP32TornadoTensor(new FloatArray(64), Shape.of(64));
        TornadoTensor[] norm = {f32};
        return new LlamaTornadoWeights(
                f32,
                norm,
                layer,
                layer,
                layer,
                layer,
                norm,
                layer,
                layer,
                layer,
                f32,
                f32,
                f32,
                f32,
                DataType.Q4_0);
    }

    /** A graph with one tiny upload, so it is not empty. */
    private static TaskGraph graph(String name) {
        return new TaskGraph(name)
                .transferToDevice(
                        uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION,
                        new FloatArray(4));
    }

    private static ActivationTaskGraph activation(String name) {
        ImmutableTaskGraph g = graph(name).snapshot();
        return new ActivationTaskGraph() {
            public ImmutableTaskGraph getImmutableTaskGraph() {
                return g;
            }

            public GridScheduler updateGridScheduler(GridScheduler scheduler) {
                return scheduler;
            }
        };
    }

    /**
     * The plan components: tiny named graphs, and, when {@code prepack}, layer graphs that each
     * declare one prepacked Q4_0 weight through {@link #prepacker}, as the real layer builders do
     * through the session's; the test hands the plan the same prepacker, as the registry does.
     */
    private static final class Components implements BatchPrefillDecodeForwardPlanComponents {

        private final WeightPrepacker prepacker = new WeightPrepacker();
        private final boolean prepack;

        Components(boolean prepack) {
            this.prepack = prepack;
        }

        /** {@code N} layer graphs {@code prefix_i}, each declaring a weight to prepack if asked. */
        private List<ImmutableTaskGraph> layers(String prefix, boolean registers) {
            List<ImmutableTaskGraph> graphs = new ArrayList<>();
            for (int i = 0; i < N; i++) {
                TaskGraph g = graph(prefix + i);
                if (registers) {
                    TornadoTensor w = q4();
                    prepacker.configureGraphInputs(g, WeightPrepacking.of(List.of(w)), w);
                }
                graphs.add(g.snapshot());
            }
            return graphs;
        }

        private TransformerLayerTaskGraphs decodeLayers(boolean registers) {
            List<ImmutableTaskGraph> graphs = layers("layer_", registers);
            return new TransformerLayerTaskGraphs() {
                public List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs() {
                    return graphs;
                }

                public GridScheduler updateGridScheduler(GridScheduler scheduler) {
                    return scheduler;
                }

                public String getLastFFNLayerTaskGraphID() {
                    return "layer_" + (N - 1);
                }
            };
        }

        private AbstractLogitsTaskGraph logits(String previous) {
            return new AbstractLogitsTaskGraph(
                    null, weights(), null, previous, SchedulerType.NVIDIA) {
                @Override
                protected TaskGraph setupLogitsTaskGraph(TornadoWeights w, Configuration c) {
                    return graph("logits");
                }

                @Override
                public GridScheduler updateGridScheduler(GridScheduler scheduler) {
                    return scheduler;
                }
            };
        }

        @Override
        public ActivationTaskGraph singleTokenActivation() {
            return activation("activation");
        }

        @Override
        public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
            return decodeLayers(prepack);
        }

        @Override
        public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
            return logits(previousGraphId);
        }

        @Override
        public ActivationTaskGraph prefillDecodeActivation() {
            return activation("activation");
        }

        @Override
        public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
            return decodeLayers(prepack);
        }

        @Override
        public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
            return logits(previousGraphId);
        }

        @Override
        public ActivationTaskGraph batchPrefillActivation(int batchSize) {
            return activation("batchPrefillActivation");
        }

        @Override
        public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
            return activation("decodeActivation");
        }

        @Override
        public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
            // As in the real batched plans: the decode layers consume the weights the batch-prefill
            // layers declared, so only those register them.
            return decodeLayers(false);
        }

        @Override
        public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
            List<ImmutableTaskGraph> graphs = layers("batchPrefillLayer_", prepack);
            return new BatchPrefillTransformerLayerTaskGraphs() {
                public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
                    return graphs;
                }

                public void updateGridScheduler(GridScheduler scheduler) {}

                public String getLastLayerTaskGraphID() {
                    return "batchPrefillLayer_" + (N - 1);
                }
            };
        }
    }
}
