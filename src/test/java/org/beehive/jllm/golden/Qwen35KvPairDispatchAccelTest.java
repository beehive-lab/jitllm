package org.beehive.jllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.junit.Test;

/**
 * The production plan's key/value projection dispatch at a width and context given by {@code
 * jllm.kvDispatch.width} / {@code jllm.kvDispatch.context} (the benchmark builds contexts of prompt
 * + 8: 520 and 2056): the pair from width 512 up, the direct kernel below. One plan per JVM, as the
 * fixture demands.
 */
public class Qwen35KvPairDispatchAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
        System.setProperty("jllm.kvcache.fp16", "true");
    }

    @Test
    public void theKvProjectionsDispatchByWidth() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        assumeTrue("environment absent", modelPath != null && TupleInfo.acceleratorPresent());
        int width = Integer.getInteger("jllm.kvDispatch.width", 512);
        int context = Integer.getInteger("jllm.kvDispatch.context", 520);
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(width));
        Model model = ModelLoader.loadModel(modelPath, context, true, true);
        State state = State.withPrefillBatchSize(width, model::createNewState);
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            var config = (Qwen35Configuration) model.configuration();
            PlanDispatchEvidence.assertQwen35KvProjectionDispatch(
                    plan,
                    PlanDispatchEvidence.gridSchedulerIfAvailable(plan),
                    width,
                    config.kvDim(),
                    width >= 512);
            System.out.println(
                    "[kvdispatch] width "
                            + width
                            + " context "
                            + context
                            + ": "
                            + (width >= 512 ? "pair" : "direct")
                            + " asserted");
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }
}
