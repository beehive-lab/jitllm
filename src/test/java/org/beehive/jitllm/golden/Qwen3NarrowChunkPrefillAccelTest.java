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
import java.util.Set;
import org.beehive.jitllm.backend.tornado.BatchPrefillDecodePlan;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

// @formatter:off
/**
 * Qwen3-0.6B F16's tensor-core batched prefill at a plan 512 wide, where a chunk of up to 128
 * tokens runs on the narrow family: a short prompt as its first and only chunk, and the last part
 * of a long one after a full chunk, at start position 512.
 *
 * <ul>
 *   <li>Against the host with {@link #FP16}, native libraries on and off, with an FP32 key/value
 *       cache on both sides, as the other batched parity cases compare.
 *   <li>The FP16 cache against the FP32 one on the same plan, within {@link GpuFp16KvPrecision}'s
 *       bounds: the narrow chunk's attention reads the FP16 cache through the tensor-core
 *       attention.
 *   <li>One plan through a long prompt, a short one and a reset: the long prompt's rows come back
 *       bit for bit.
 * </ul>
 *
 * <p>The rows are logits at every generated position, teacher-forced onto the host's tokens, from
 * the real generation loop: prefill in chunks, then single-token decode reading what the prefill
 * wrote.
 */
// @formatter:on
public class Qwen3NarrowChunkPrefillAccelTest extends CpuGpuParity {

    private static final int WIDTH = 512;
    private static final int CONTEXT = 1024;
    private static final int STEPS = 24;

    private static Model cpuModel;
    private static Model gpuModel;
    private static final Map<String, KvPrecisionHarness.Trace> HOST = new HashMap<>();

    private final Map<String, String> previous = new HashMap<>();

    @Before
    public void properties() {
        for (String key :
                new String[] {
                    "use.tornadovm",
                    "jitllm.withPrefillDecode",
                    "jitllm.prefillBatchSize",
                    ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY
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

    private static Path file() {
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                file != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue("tensor-core prefill", TensorCoreSupport.isTensorCoreCapableBackend());
        return file;
    }

    private static Model gpu() throws Exception {
        if (gpuModel == null) {
            gpuModel = KvPrecisionHarness.load(file(), CONTEXT, true);
        }
        return gpuModel;
    }

    /** About 580 tokens: one chunk of 512, then a tail of at most 128 on the narrow family. */
    private static List<Integer> longPrompt(Model model) {
        List<Integer> prompt = KvPrecisionHarness.longPrompt(model, 560);
        assertTrue(
                "a full chunk and then a narrow one: " + prompt.size() + " tokens",
                prompt.size() > WIDTH + 1 && prompt.size() <= WIDTH + 128);
        return prompt;
    }

    /** A short prompt: one chunk, on the narrow family. */
    private static List<Integer> shortPrompt(Model model) {
        List<Integer> prompt = KvPrecisionHarness.longPrompt(model, 40);
        assertTrue("one narrow chunk: " + prompt.size() + " tokens", prompt.size() <= 128);
        return prompt;
    }

    private static KvPrecisionHarness.Trace host(String name, List<Integer> prompt)
            throws Exception {
        KvPrecisionHarness.Trace trace = HOST.get(name);
        if (trace == null) {
            if (cpuModel == null) {
                cpuModel = KvPrecisionHarness.load(file(), CONTEXT, false);
            }
            // The host runs its own sequential loop: the batched-prefill switches are the device's.
            String batched = System.clearProperty("jitllm.withPrefillDecode");
            String width = System.clearProperty("jitllm.prefillBatchSize");
            try {
                trace =
                        KvPrecisionHarness.run(
                                cpuModel, false, DataType.F32, 1, prompt, STEPS, null);
            } finally {
                System.setProperty("jitllm.withPrefillDecode", batched);
                System.setProperty("jitllm.prefillBatchSize", width);
            }
            HOST.put(name, trace);
        }
        return trace;
    }

    private void assertHostParity(boolean nativeLibraries) throws Exception {
        System.setProperty(
                ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, String.valueOf(nativeLibraries));
        Model model = gpu();
        for (String name : new String[] {"long", "short"}) {
            List<Integer> prompt = name.equals("long") ? longPrompt(model) : shortPrompt(model);
            KvPrecisionHarness.Trace host = host(name, prompt);
            KvPrecisionHarness.Trace gpu =
                    KvPrecisionHarness.run(
                            model, true, DataType.F32, WIDTH, prompt, STEPS, host.tokens());
            assertEquals("batch-prefill-decode", gpu.executionMode());
            assertRowsWithin(
                    "Qwen3 F16 width "
                            + WIDTH
                            + ", "
                            + name
                            + " prompt ("
                            + prompt.size()
                            + " tokens), "
                            + (nativeLibraries ? "native" : "JIT"),
                    host.rows(),
                    gpu.rows(),
                    FP16);
        }
    }

    @Test
    public void nativeLibrariesAgainstTheHost() throws Exception {
        assertHostParity(true);
    }

    @Test
    public void jitKernelsAgainstTheHost() throws Exception {
        assertHostParity(false);
    }

    @Test
    public void fp16CacheAgainstFp32() throws Exception {
        Model model = gpu();
        List<Integer> prompt = longPrompt(model);
        var comparison =
                KvPrecisionHarness.compareFp16AgainstFp32(
                        model, true, WIDTH, prompt, STEPS, "batch-prefill-decode");
        System.out.println("[narrow] FP16 against FP32 cache: " + comparison);
        comparison.assertWithin("Qwen3 F16 width " + WIDTH, 0.9999, 0.01, 0.95);
    }

    @Test
    public void resetRestoresTheSequenceAcrossFamilies() throws Exception {
        Model model = gpu();
        State state = State.withPrefillBatchSize(WIDTH, model::createNewState);
        int seed = state.latestToken;
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            assertEquals(
                    "the narrow family's rows",
                    128,
                    ((BatchPrefillDecodePlan) plan).narrowPrefillRows());
            List<float[]> fresh = generate(model, state, plan, longPrompt(model));
            state.latestToken = seed;
            List<float[]> other = generate(model, state, plan, shortPrompt(model));
            assertTrue("the short prompt ran", !other.isEmpty());
            plan.resetSequenceState();
            state.latestToken = seed;
            List<float[]> again = generate(model, state, plan, longPrompt(model));
            assertEquals("rows after the reset", fresh.size(), again.size());
            for (int i = 0; i < fresh.size(); i++) {
                assertArrayEquals("row " + i + " after the reset", fresh.get(i), again.get(i), 0f);
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

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
        int budget = prompt.size() + STEPS - skippedSeed;
        model.generateTokensGPU(state, 0, prompt, Set.of(), budget, capturing, false, null, plan);
        return rows;
    }
}
