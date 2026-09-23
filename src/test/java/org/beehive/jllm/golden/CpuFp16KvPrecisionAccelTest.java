package org.beehive.jllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.model.Model;
import org.junit.Test;

/**
 * The CPU's FP16 key/value cache against its FP32 one, per family, over a prompt that crosses
 * several prefill chunks and a multi-step decode that reads what the prompt wrote.
 *
 * <p>Storage is the only difference — accumulation is FP32 in both — so the bound is the rounding
 * of each cached value to half precision. Measured on the fixtures here: cosine at least 0.999998,
 * relative L2 at most 0.0022, top-1 agreement 100%; the thresholds keep a wide margin.
 */
public class CpuFp16KvPrecisionAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 1024;
    private static final int PROMPT_TOKENS = 300;
    private static final int DECODE_STEPS = 24;

    @Test
    public void llama() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0, 1);
    }

    @Test
    public void llamaChunkedPrefill() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0, 32);
    }

    @Test
    public void qwen3() throws Exception {
        check(Fixture.QWEN3_0_6B_F16, 1);
    }

    @Test
    public void qwen2() throws Exception {
        check(Fixture.QWEN2_5_0_5B_Q8_0, 1);
    }

    @Test
    public void granite() throws Exception {
        check(Fixture.GRANITE_3_2_2B_Q8_0, 1);
    }

    @Test
    public void phi3() throws Exception {
        check(Fixture.PHI3_MINI_4K_Q8_0, 1);
    }

    private static void check(Fixture fixture, int prefillBatchSize) throws Exception {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousPrefill = System.getProperty("jllm.withPrefillDecode");
        String previousWidth = System.getProperty("jllm.prefillBatchSize");
        System.setProperty(GPU_PROPERTY, "false");
        if (prefillBatchSize > 1) {
            System.setProperty("jllm.withPrefillDecode", "true");
            System.setProperty("jllm.prefillBatchSize", Integer.toString(prefillBatchSize));
        }
        try {
            Model model = KvPrecisionHarness.load(file, CONTEXT, false);
            List<Integer> prompt = KvPrecisionHarness.longPrompt(model, PROMPT_TOKENS);
            var comparison =
                    KvPrecisionHarness.compareFp16AgainstFp32(
                            model, false, prefillBatchSize, prompt, DECODE_STEPS);
            System.out.println(
                    "[kv-precision] cpu "
                            + fixture
                            + " prefill="
                            + prefillBatchSize
                            + " promptTokens="
                            + prompt.size()
                            + " "
                            + comparison);
            comparison.assertWithin(fixture + " cpu", 0.9999, 0.01, 0.95);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore("jllm.withPrefillDecode", previousPrefill);
            restore("jllm.prefillBatchSize", previousWidth);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
