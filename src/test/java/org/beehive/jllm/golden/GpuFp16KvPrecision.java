package org.beehive.jllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.model.Model;

/**
 * The GPU's FP16 key/value cache against its FP32 one, for every supported (family, weights, mode)
 * row with a fixture here: a prompt that crosses several 128-token prefill chunks, then a
 * multi-step decode that reads what prefill wrote.
 *
 * <p>The rows live in one small test class per family and weight type (the surefire fork is per
 * class, and TornadoVM keeps device memory resident across the plans of one JVM), and call {@link
 * #check}.
 *
 * <p>The FP32 and FP16 runs may take different attention kernels, so the bound covers the kernels'
 * accumulation order as well as the rounding of each cached value; the thresholds are set from
 * measurement with margin. A row whose FP16 run is identical to FP32 fails: that path never read
 * the FP16 cache.
 */
public final class GpuFp16KvPrecision {

    private GpuFp16KvPrecision() {}

    private static final int CONTEXT = 1024;
    private static final int PROMPT_TOKENS = 300;
    private static final int DECODE_STEPS = 24;
    private static final int CHUNK = 128;

    public enum Mode {
        SINGLE_TOKEN("single-token"),
        PREFILL_DECODE("prefill-decode"),
        BATCHED("batch-prefill-decode");

        /** What the plan's execution info must report for this mode. */
        final String planMode;

        Mode(String planMode) {
            this.planMode = planMode;
        }
    }

    public static void check(Fixture fixture, Mode mode) throws Exception {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        String[] keys = {"use.tornadovm", "jllm.withPrefillDecode", "jllm.prefillBatchSize"};
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jllm.withPrefillDecode", Boolean.toString(mode != Mode.SINGLE_TOKEN));
        int width = mode == Mode.BATCHED ? CHUNK : 1;
        System.setProperty("jllm.prefillBatchSize", Integer.toString(width));
        try {
            Model model = KvPrecisionHarness.load(file, CONTEXT, true);
            List<Integer> prompt = KvPrecisionHarness.longPrompt(model, PROMPT_TOKENS);
            var comparison =
                    KvPrecisionHarness.compareFp16AgainstFp32(
                            model, true, width, prompt, DECODE_STEPS, mode.planMode);
            System.out.println(
                    "[kv-precision] gpu "
                            + fixture
                            + " "
                            + mode
                            + " promptTokens="
                            + prompt.size()
                            + " "
                            + comparison);
            comparison.assertWithin(fixture + " " + mode, 0.9999, 0.01, 0.95);
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
