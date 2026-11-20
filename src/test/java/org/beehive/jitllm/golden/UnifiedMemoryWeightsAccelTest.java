package org.beehive.jitllm.golden;

import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.WRITE;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Weights read in place from the mapped file ({@link UnifiedMemoryWeights}) give the logits of
 * weights copied to the device, bit for bit: the same JIT kernels read the same bytes. A prompt
 * that crosses several 128-token chunks, then a decode that reads what prefill wrote.
 */
public class UnifiedMemoryWeightsAccelTest {

    private static final int CONTEXT = 1024;
    private static final int PROMPT_TOKENS = 300;
    private static final int DECODE_STEPS = 16;

    /** A tensor mapped from a file is read in place; an array on the heap is uploaded. */
    @Test
    public void onlyMappedWeightsAreReadInPlace() throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = Files.createTempFile("weights", ".bin");
        try {
            Files.write(file, new byte[1 << 16]);
            ByteArray mapped;
            try (FileChannel channel = FileChannel.open(file, READ, WRITE)) {
                mapped =
                        ByteArray.fromSegmentShallow(
                                channel.map(
                                        FileChannel.MapMode.PRIVATE, 0, 1 << 16, Arena.ofAuto()));
            }
            FloatArray heap = new FloatArray(1024);
            TornadoWorkspace ws = new TornadoWorkspace();
            ws.unifiedMemory = true;
            Object[] others =
                    UnifiedMemoryWeights.consume(new TaskGraph("weights"), ws, mapped, heap);
            assertArrayEquals(new Object[] {heap}, others);
            ws.unifiedMemory = false;
            others = UnifiedMemoryWeights.consume(new TaskGraph("copies"), ws, mapped, heap);
            assertArrayEquals(new Object[] {mapped, heap}, others);
        } finally {
            Files.delete(file);
        }
    }

    @Test
    public void qwen3Q8_0SingleToken() throws Exception {
        check(Fixture.QWEN3_0_6B_Q8_0, 1);
    }

    @Test
    public void qwen3Q8_0Batched() throws Exception {
        check(Fixture.QWEN3_0_6B_Q8_0, 128);
    }

    @Test
    public void qwen3F16Batched() throws Exception {
        check(Fixture.QWEN3_0_6B_F16, 128);
    }

    @Test
    public void llamaQ8_0Batched() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0, 128);
    }

    /**
     * A mixed model: Q4_K, Q5_K and Q6_K experts, Q8_0 attention, F32 norms and convolution taps,
     * each read in place. LFM2 keeps an FP32 key/value cache on Metal.
     */
    @Test
    public void lfm2SingleToken() throws Exception {
        check(Fixture.LFM2_8B_A1B_Q4_K_XL, 1, PROMPT_TOKENS, DataType.F32);
    }

    /**
     * A plan 512 wide with its narrow family: 40 tokens run on the narrow graphs alone, 600 on a
     * full chunk and then a narrow one. Narrow graphs read the weights in place as their twins do.
     */
    @Test
    public void qwen3F16NarrowChunks() throws Exception {
        check(Fixture.QWEN3_0_6B_F16, 512, 40);
        check(Fixture.QWEN3_0_6B_F16, 512, 600);
    }

    @Test
    public void llamaQ8_0NarrowChunks() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0, 512, 40);
        check(Fixture.LLAMA_3_2_1B_Q8_0, 512, 600);
    }

    /**
     * Two plans over one model, both reading its weights in place, each with its own bindings:
     * either gives the copied plan's logits, and closing one leaves the other's weights readable.
     */
    @Test
    public void twoPlansReadTheSameWeightsInPlace() throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                file != null);
        String[] keys = {
            "use.tornadovm",
            "jitllm.withPrefillDecode",
            "jitllm.prefillBatchSize",
            "jitllm.nativeLibraries",
            UnifiedMemoryWeights.PROPERTY
        };
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", "512");
        System.setProperty("jitllm.nativeLibraries", "false");
        try {
            Model model = KvPrecisionHarness.load(file, CONTEXT, true);
            List<Integer> prompt = KvPrecisionHarness.longPrompt(model, 600);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "false");
            var copied =
                    KvPrecisionHarness.run(
                            model, true, DataType.F16, 512, prompt, DECODE_STEPS, null);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "true");
            // The copied run's storage: an FP16 key/value cache.
            StorageOptions storage = new StorageOptions(DataType.F16, false);
            State stateA =
                    State.withStorageOptions(
                            storage, () -> State.withPrefillBatchSize(512, model::createNewState));
            State stateB =
                    State.withStorageOptions(
                            storage, () -> State.withPrefillBatchSize(512, model::createNewState));
            int seed = stateA.latestToken;
            // Each plan built is closed whatever fails: plan A in the outer block unless the test
            // closed it already, plan B in the inner one.
            TornadoVMMasterPlan planA = TornadoVMMasterPlan.initializeTornadoVMPlan(stateA, model);
            boolean planAOpen = true;
            try {
                TornadoVMMasterPlan planB =
                        TornadoVMMasterPlan.initializeTornadoVMPlan(stateB, model);
                try {
                    assertRows("plan A", copied.rows(), generate(model, stateA, planA, prompt));
                    assertRows("plan B", copied.rows(), generate(model, stateB, planB, prompt));
                    planAOpen = false;
                    planA.freeTornadoExecutionPlan();
                    planB.resetSequenceState();
                    stateB.latestToken = seed;
                    assertRows(
                            "plan B after plan A closed",
                            copied.rows(),
                            generate(model, stateB, planB, prompt));
                } finally {
                    planB.freeTornadoExecutionPlan();
                }
            } finally {
                if (planAOpen) {
                    planA.freeTornadoExecutionPlan();
                }
            }
        } finally {
            for (int i = 0; i < keys.length; i++) {
                if (previous[i] == null) {
                    System.clearProperty(keys[i]);
                } else {
                    System.setProperty(keys[i], previous[i]);
                }
            }
        }
    }

    private static void assertRows(String what, List<float[]> expected, List<float[]> got) {
        assertEquals(what + ", rows", expected.size(), got.size());
        for (int r = 0; r < expected.size(); r++) {
            assertArrayEquals(what + ", row " + r, expected.get(r), got.get(r), 0f);
        }
    }

    /**
     * The greedy logits of {@code prompt} and {@link #DECODE_STEPS} decode steps on {@code plan}.
     */
    private static List<float[]> generate(
            Model model, State state, TornadoVMMasterPlan plan, List<Integer> prompt) {
        List<float[]> rows = new ArrayList<>();
        Sampler capturing =
                logits -> {
                    float[] row = new float[logits.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = logits.get(i);
                    }
                    rows.add(row);
                    return Sampler.TENSOR_ARGMAX.sampleToken(logits);
                };
        int skippedSeed = PromptIngestion.of(state, prompt, 0).firstIndex();
        int budget = prompt.size() + DECODE_STEPS - skippedSeed;
        model.generateTokensGPU(state, 0, prompt, Set.of(), budget, capturing, false, null, plan);
        return rows;
    }

    private static void check(Fixture fixture, int batch) throws Exception {
        check(fixture, batch, PROMPT_TOKENS);
    }

    private static void check(Fixture fixture, int batch, int promptTokens) throws Exception {
        check(fixture, batch, promptTokens, DataType.F16);
    }

    private static void check(Fixture fixture, int batch, int promptTokens, DataType keyValue)
            throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        String[] keys = {
            "use.tornadovm",
            "jitllm.withPrefillDecode",
            "jitllm.prefillBatchSize",
            "jitllm.nativeLibraries",
            UnifiedMemoryWeights.PROPERTY
        };
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", Boolean.toString(batch > 1));
        System.setProperty("jitllm.prefillBatchSize", Integer.toString(batch));
        // In place, MLX is not used (UnifiedMemoryWeights); the copies run the same JIT kernels.
        System.setProperty("jitllm.nativeLibraries", "false");
        try {
            Model model = KvPrecisionHarness.load(file, CONTEXT, true);
            List<Integer> prompt = KvPrecisionHarness.longPrompt(model, promptTokens);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "false");
            var copied =
                    KvPrecisionHarness.run(
                            model, true, keyValue, batch, prompt, DECODE_STEPS, null);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "true");
            var inPlace =
                    KvPrecisionHarness.run(
                            model, true, keyValue, batch, prompt, DECODE_STEPS, null);
            assertEquals(copied.executionMode(), inPlace.executionMode());
            assertArrayEquals(fixture + " tokens", copied.tokens(), inPlace.tokens());
            for (int r = 0; r < copied.rows().size(); r++) {
                assertArrayEquals(
                        fixture + " logits row " + r,
                        copied.rows().get(r),
                        inPlace.rows().get(r),
                        0f);
            }
        } finally {
            for (int i = 0; i < keys.length; i++) {
                if (previous[i] == null) {
                    System.clearProperty(keys[i]);
                } else {
                    System.setProperty(keys[i], previous[i]);
                }
            }
        }
    }
}
