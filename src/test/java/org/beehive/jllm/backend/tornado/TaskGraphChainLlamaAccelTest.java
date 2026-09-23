package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Map;
import org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * {@code --print-taskgraph-chain} on Llama 3.2 1B F16 single-token plans, the lowered one (the
 * default) and the legacy one: the property makes the plan print its chain once, at build, with
 * every graph, its transfers, its tasks with named arguments and worker grids, and the schedule.
 */
public class TaskGraphChainLlamaAccelTest {

    @Test
    public void theLoweredPlanPrintsItsChain() throws Exception {
        long before = LoweredPlanSelection.loweredPlanCount();
        String chain = printed(Map.of());
        assertTrue(
                "the default plan for this model is the lowered one",
                LoweredPlanSelection.loweredPlanCount() > before);
        check(chain);
    }

    @Test
    public void theLegacyPlanPrintsItsChain() throws Exception {
        long before = LoweredPlanSelection.loweredPlanCount();
        String chain = printed(Map.of("jllm.lowering", "off"));
        assertEquals(before, LoweredPlanSelection.loweredPlanCount());
        check(chain);
    }

    private static void check(String chain) {
        assertTrue(chain, chain.startsWith("Plan  LLAMA_3 F16 · single-token · 18 graphs"));
        assertTrue(chain, chain.contains("token    activation → 16 × layers → logits"));
        // the sixteen layer graphs collapse into one block, activation and logits stay single
        assertTrue(chain, chain.contains("[0]     activationUpdate"));
        assertTrue(
                chain,
                chain.contains("[1-16]  layer_0 … layer_15   (layers, 16 graphs, 1 layer each)"));
        assertTrue(chain, chain.contains("[17]    logits"));
        // one line per task: name, kernel, worker grid; nothing else
        assertTrue(
                chain,
                chain.matches("(?s).*task +qkv_projection +fusedQKVMatmulX +\\d+ / \\d+\n.*"));
        assertFalse(chain, chain.contains("libraryTask"));
        assertFalse(chain, chain.contains("upload once"));
        assertFalse(chain, chain.contains("(unavailable"));
        assertTrue(
                "a summary, not a dump: " + chain.lines().count() + " lines",
                chain.lines().count() < 40);
    }

    /** Builds the plan with the printer on and returns what it wrote to stderr. */
    static String printed(Map<String, String> properties) throws Exception {
        Path file = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                file != null);
        return capture(file, properties);
    }

    static String capture(Path file, Map<String, String> properties) throws Exception {
        Map<String, String> previous = new java.util.HashMap<>();
        java.util.Map<String, String> all = new java.util.HashMap<>(properties);
        all.put("use.tornadovm", "true");
        all.put(TaskGraphChainPrinter.PROPERTY, "true");
        all.forEach((k, v) -> previous.put(k, System.getProperty(k)));
        all.forEach(System::setProperty);
        StringBuilder captured = new StringBuilder();
        TornadoVMMasterPlan plan = null;
        try {
            Model model = ModelLoader.loadModel(file, 512, true, true);
            State state = model.createNewState();
            TaskGraphChainPrinter.output(captured::append);
            plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        } finally {
            TaskGraphChainPrinter.output(null);
            if (plan != null) plan.freeTornadoExecutionPlan();
            previous.forEach(
                    (k, v) -> {
                        if (v == null) System.clearProperty(k);
                        else System.setProperty(k, v);
                    });
        }
        String out = captured.toString();
        int start = out.indexOf("Plan  ");
        assertTrue("the plan printed no chain:\n" + out, start >= 0);
        return out.substring(start);
    }
}
