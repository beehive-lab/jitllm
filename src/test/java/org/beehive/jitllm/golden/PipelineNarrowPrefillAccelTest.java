package org.beehive.jitllm.golden;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.BatchPrefillDecodePlan;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

// @formatter:off
/**
 * The narrow prefill family of a model split across devices: two stages on one device, a plan 512
 * wide whose narrow family is 128 wide, against the same model on one device. A stage runs the same
 * layer graphs as the single-device plan and hands the hidden state on exactly, so the rows are bit
 * for bit the same, with the narrow family and without it.
 *
 * <p>Prefills of 40 tokens (one narrow chunk on every stage), 560 (a full chunk, then a narrow one)
 * and 712 (a full chunk, then one past the narrow width). The families whose batched prefill splits
 * across devices: Llama Q4_0, Gemma 4 and Qwen3.5. Both stages share one device, so this covers the
 * stage programs and their handoff, not a transfer between two devices.
 */
// @formatter:on
public class PipelineNarrowPrefillAccelTest {

    private static final int WIDTH = 512;
    private static final int CONTEXT = 1024;
    private static final int STEPS = 8;

    private final Map<String, String> previous = new HashMap<>();

    @Before
    public void properties() {
        for (String key :
                new String[] {
                    "use.tornadovm",
                    "jitllm.withPrefillDecode",
                    "jitllm.prefillBatchSize",
                    BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY,
                    DeviceSplit.DEVICES_PROPERTY,
                    DeviceSplit.TRANSPORT_PROPERTY
                }) {
            previous.put(key, System.getProperty(key));
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(WIDTH));
    }

    @After
    public void restore() {
        previous.forEach(
                (k, v) -> {
                    if (v == null) {
                        System.clearProperty(k);
                    } else {
                        System.setProperty(k, v);
                    }
                });
    }

    @Test
    public void llamaQ4_0() throws Exception {
        assertStagesMatchOneDevice(GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q4_0));
    }

    @Test
    public void gemma4Q8_0() throws Exception {
        assertStagesMatchOneDevice(GoldenFixture.locate(Fixture.GEMMA_4_E2B_Q8_0));
    }

    @Test
    public void qwen35Q8_0() throws Exception {
        assertStagesMatchOneDevice(
                GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-Q8_0.gguf"));
    }

    private void assertStagesMatchOneDevice(Path file) throws Exception {
        assumeTrue("model absent: " + file, file != null && file.toFile().isFile());
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        Model model = KvPrecisionHarness.load(file, CONTEXT, true);
        assumePipelineNarrowFamily(model);
        for (boolean narrow : new boolean[] {true, false}) {
            if (narrow) {
                System.clearProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
            } else {
                System.setProperty(
                        BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY, "false");
            }
            for (int tokens : new int[] {40, WIDTH + 48, WIDTH + 200}) {
                List<Integer> prompt = prefilling(model, tokens);
                System.clearProperty(DeviceSplit.DEVICES_PROPERTY);
                KvPrecisionHarness.Trace one =
                        KvPrecisionHarness.run(
                                model, true, DataType.F16, WIDTH, prompt, STEPS, null);
                System.setProperty(DeviceSplit.DEVICES_PROPERTY, "0:0,0:0");
                System.setProperty(DeviceSplit.TRANSPORT_PROPERTY, "host");
                KvPrecisionHarness.Trace stages =
                        KvPrecisionHarness.run(
                                model, true, DataType.F16, WIDTH, prompt, STEPS, one.tokens());
                String what =
                        file.getFileName()
                                + (narrow ? ", narrow family" : ", no narrow family")
                                + ", prefill of "
                                + tokens
                                + " tokens, two stages against one device";
                assertEquals(what + ", rows", one.rows().size(), stages.rows().size());
                for (int i = 0; i < one.rows().size(); i++) {
                    assertArrayEquals(
                            what + ", row " + i, one.rows().get(i), stages.rows().get(i), 0f);
                }
                System.out.println("[pipeline] " + what + ": bit for bit");
            }
        }
    }

    /** Skips unless the two-stage plan of this model has a narrow family 128 wide. */
    private static void assumePipelineNarrowFamily(Model model) {
        System.setProperty(DeviceSplit.DEVICES_PROPERTY, "0:0,0:0");
        System.setProperty(DeviceSplit.TRANSPORT_PROPERTY, "host");
        State state = State.withPrefillBatchSize(WIDTH, model::createNewState);
        TornadoVMMasterPlan plan;
        try {
            plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        } catch (UnsupportedOperationException refused) {
            assumeTrue("no batched pipeline here: " + refused.getMessage(), false);
            throw refused;
        }
        try {
            assertTrue("a pipeline plan", ((BatchPrefillDecodePlan) plan).overlapsPrefillChunks());
            assertEquals(
                    "the narrow family's rows",
                    128,
                    ((BatchPrefillDecodePlan) plan).narrowPrefillRows());
        } finally {
            plan.freeTornadoExecutionPlan();
            System.clearProperty(DeviceSplit.DEVICES_PROPERTY);
        }
    }

    /**
     * A prompt whose batched prefill is exactly {@code tokens} long; see NarrowChunkPrefillParity.
     */
    private static List<Integer> prefilling(Model model, int tokens) {
        List<Integer> text = KvPrecisionHarness.longPrompt(model, tokens * 2);
        boolean seedIsHead = PromptIngestion.of(model.createNewState(), text, 0).firstIndex() == 1;
        int size = seedIsHead ? tokens + 1 : tokens;
        return new ArrayList<>(text.subList(0, size));
    }
}
