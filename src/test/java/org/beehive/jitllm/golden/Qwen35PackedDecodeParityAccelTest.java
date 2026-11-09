package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.After;
import org.junit.Test;

// @formatter:off
/**
 * Qwen3.5-0.8B Q4_K_M on CUDA against the exact CPU reference, decoding on the packed integer path
 * it takes by default, and with that path off to isolate the batched prefill.
 *
 * <p>On the default path the Q8_0 {@code ssm_alpha} and {@code ssm_beta} and the Q5_K {@code
 * ssm_out} of every recurrent layer read their input quantized to eight bits in blocks of 32; no
 * other decode projection does. Each case first asserts exactly that dispatch, so a comparison
 * cannot pass on a plan that did not take the packed path. The packed envelopes are the ones this
 * family already uses for that arithmetic: {@link #Q8_0_PACKED_DECODE} for a batched plan, whose
 * packed projections run at the decoded positions only, and {@link #Q8_0_FULLY_PACKED} for the
 * single-token plan, which runs them at every position, prompt included.
 *
 * <p>The packed kernels themselves are held to a same-quantization reference by {@link
 * Qwen35PackedDecodeProjectionAccelTest}, on this file's weights. With the packed path off, the
 * prefill keeps {@link #FP16} against the exact host.
 *
 * <p>Uses the file under the fixture directory, by name; skips when it is absent or the backend is
 * not CUDA.
 */
// @formatter:on
public class Qwen35PackedDecodeParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @After
    public void restoreProperties() {
        System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        System.clearProperty(PackedActivationTasks.DISABLE_PROPERTY);
    }

    private static Path model() {
        Path file = GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf");
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        BackendId backend = TornadoDevices.current().backend();
        assumeTrue("this runs on the " + backend + " backend", BackendId.CUDA.equals(backend));
        return file;
    }

    /** The decode projections each layer ran on a packed integer kernel, by layer, with kernel. */
    private static Map<Integer, Map<String, String>> packedDecodeKernels(GoldenCapture.Result gpu) {
        Map<Integer, Map<String, String>> packed = new TreeMap<>();
        gpu.decodeProjectionKernels.forEach(
                (layer, kernels) ->
                        kernels.forEach(
                                (task, kernel) -> {
                                    if (kernel.contains("DP4A") || kernel.endsWith("Packed")) {
                                        packed.computeIfAbsent(layer, l -> new TreeMap<>())
                                                .put(task, kernel);
                                    }
                                }));
        return packed;
    }

    /**
     * Every recurrent layer's alpha, beta and readout projections on the kernels {@link
     * Qwen35PackedDecodeProjectionAccelTest} checks, and nothing else packed.
     */
    private static void assertPackedDecode(GoldenCapture.Result gpu) {
        assertEquals("decode layers recorded", gpu.layers, gpu.decodeProjectionKernels.size());
        Map<Integer, Map<String, String>> expected = new TreeMap<>();
        gpu.decodeProjectionKernels.forEach(
                (layer, kernels) -> {
                    if (kernels.containsKey("ssm_out_proj")) {
                        expected.put(
                                layer,
                                Map.of(
                                        "ssm_alpha_proj", "matrixVectorGenericQ8_0DP4A",
                                        "ssm_beta_proj", "matrixVectorGenericQ8_0DP4A",
                                        "ssm_out_proj", "matrixVectorGenericWithResidualQ5_KDP4A"));
                    }
                });
        assertEquals("recurrent layers", 18, expected.size());
        assertEquals("decode projections on the packed path", expected, packedDecodeKernels(gpu));
    }

    @Test
    public void packedDecodeAtWidth128() throws Exception {
        assertPackedDecode(
                assertParityBatched(model(), "Q4_K_M width 128", Q8_0_PACKED_DECODE, 128));
    }

    /** Three prefill chunks, the last one partial, before the packed decode. */
    @Test
    public void packedDecodeAtWidth7() throws Exception {
        assertPackedDecode(assertParityBatched(model(), "Q4_K_M width 7", Q8_0_PACKED_DECODE, 7));
    }

    @Test
    public void packedDecodeOnJitKernelsAtWidth128() throws Exception {
        Path file = model();
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "false");
        assertPackedDecode(
                assertParityBatched(file, "Q4_K_M width 128 JIT", Q8_0_PACKED_DECODE, 128));
    }

    /** The single-token plan, which runs the packed projections at every prompt position too. */
    @Test
    public void packedDecodeOnTheSingleTokenPlan() throws Exception {
        assertPackedDecode(assertParity(model(), "Q4_K_M single-token", Q8_0_FULLY_PACKED));
    }

    @Test
    public void prefillAloneAtWidth128() throws Exception {
        Path file = model();
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
        GoldenCapture.Result gpu =
                assertParityBatched(file, "Q4_K_M width 128, packed path off", FP16, 128);
        assertEquals("decode layers recorded", gpu.layers, gpu.decodeProjectionKernels.size());
        assertEquals("decode projections on the packed path", Map.of(), packedDecodeKernels(gpu));
    }
}
