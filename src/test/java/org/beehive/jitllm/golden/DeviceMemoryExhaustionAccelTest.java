package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.InsufficientDeviceMemoryException;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * A load that does not fit the device budget fails while allocating, with {@code GPUL-MEM-001} and
 * what to change; nothing refuses it on a prediction beforehand. What it allocated is released: a
 * model that fits then loads and generates in the same process.
 *
 * <p>Example: under a 1536 MB budget, Llama-3.2-1B F16 (2.5 GB of weights) fails, then Qwen3-0.6B
 * Q8_0 runs.
 *
 * <p>TornadoVM reads its budget ({@code tornado.device.memory}, what {@code --gpu-memory} sets)
 * once, when it loads: this class sets it statically, and surefire runs each accelerator test class
 * in its own JVM. The weights are copied, since weights read in place from unified memory are not
 * charged to that budget.
 */
public class DeviceMemoryExhaustionAccelTest {

    static {
        System.setProperty("tornado.device.memory", "1536MB");
        System.setProperty(UnifiedMemoryWeights.PROPERTY, "false");
    }

    @Test
    public void anOverBudgetLoadFailsClearlyAndAFittingLoadThenRuns() throws Exception {
        Path big = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        Path small = GoldenFixture.locate(Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                big != null);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_Q8_0),
                small != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        String previousGpu = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            ModelOptions options =
                    ModelOptions.builder()
                            .contextLength(512)
                            .executionPolicy(
                                    ExecutionPolicy.builder().nativeLibraries(false).build())
                            .build();
            InsufficientDeviceMemoryException failure =
                    assertThrows(
                            InsufficientDeviceMemoryException.class, () -> generate(big, options));
            assertEquals(DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT, failure.code());
            String message = String.valueOf(failure.getMessage());
            boolean clear =
                    message.startsWith("[GPUL-MEM-001]") && message.contains("--gpu-memory");
            if (!clear) {
                failure.printStackTrace(System.out);
            }
            assertTrue("a clear allocation failure, not " + failure, clear);
            System.out.println(message);
            assertFalse("a model that fits runs afterwards", generate(small, options).isBlank());
        } finally {
            if (previousGpu == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previousGpu);
            }
        }
    }

    private static String generate(Path model, ModelOptions options) throws Exception {
        try (LocalModel loaded = LocalModels.load(model, options);
                GenerationSession session = ((TextGenerationModel) loaded).newSession()) {
            return session.generate(
                            GenerationRequest.builder()
                                    .prompt("The capital of France is")
                                    .maxNewTokens(8)
                                    .temperature(0.0f)
                                    .seed(42L)
                                    .build())
                    .text();
        }
    }
}
