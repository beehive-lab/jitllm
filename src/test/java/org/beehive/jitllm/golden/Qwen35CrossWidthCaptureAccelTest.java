package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * One chunk width's logits, written out so another width's run can be held against them.
 *
 * <p>Chunk invariance across widths, with the precision configuration held fixed on both sides —
 * the question is whether ingesting the same prompt in chunks of 32 and of 64 produces the same
 * answers, not whether two arithmetics agree.
 *
 * <p><b>Exactness is the right bar, and that is a claim about the kernels rather than a hope.</b>
 * Every batched projection reduces over K with {@code for (blockIndex = 0; blockIndex < k / 32;
 * blockIndex++)} — a sequential loop whose bound is the reduction length. The chunk width decides
 * how many row tiles there are and therefore which workgroup owns which prompt row; it does not
 * enter the order in which one output accumulates. So corresponding logical positions should agree
 * bit for bit, and a tolerance here would hide exactly the addressing defect this is for.
 *
 * <p>The token sequence is forced from a fixed seed rather than sampled, so both runs traverse the
 * same positions whatever the logits do, and the generated count is odd — the final chunk is
 * partial at both widths, which is the case where an active count and a launch width differ.
 *
 * <p>Two runs, two JVMs: this fixture holds 15.5 GiB on the device and TornadoVM returns freed
 * device memory to its own provider, so a second plan in one process runs out. The width comes from
 * {@code jitllm.crossWidth.width} and the rows go to {@code jitllm.crossWidth.out}.
 */
// @formatter:on
public class Qwen35CrossWidthCaptureAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    /** Odd on purpose: neither 32 nor 64 divides it, so both widths end on a partial chunk. */
    private static final int FORCED_TOKENS = 37;

    @Test
    public void captureAtTheRequestedWidth() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println("[SKIP] environment absent");
            assumeTrue("environment absent", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] no TornadoVM device");
            assumeTrue("environment absent", false);
        }
        String out = System.getProperty("jitllm.crossWidth.out");
        assumeTrue("no jitllm.crossWidth.out given", out != null);
        int width = Integer.getInteger("jitllm.crossWidth.width", 32);

        // Fixed, so both widths walk the same positions regardless of what the logits say.
        Random random = new Random(20260911L);
        List<Integer> forced = new ArrayList<>();
        for (int i = 0; i < FORCED_TOKENS; i++) {
            forced.add(1000 + random.nextInt(50000));
        }

        GoldenCapture.assertHostLogitsAvailable();
        GoldenCapture.Result result = GoldenCapture.capture(modelPath, true, forced, width, true);

        assertTrue("nothing was captured", !result.rows.isEmpty());
        verifyDispatch(result, width);
        write(Paths.get(out), width, result);
        System.out.printf(
                "[CROSSWIDTH] width %d wrote %d rows of %d to %s%n",
                width, result.rows.size(), result.rows.get(0).length, out);
    }

    /**
     * What the capture's own plan must be built with for the file it writes to mean what its name
     * says. Nothing here; the tensor-core subclass overrides it, so a scalar-plan capture cannot be
     * filed as MMA evidence.
     */
    protected void verifyDispatch(GoldenCapture.Result result, int width) {}

    private static void write(Path path, int width, GoldenCapture.Result result)
            throws IOException {
        try (DataOutputStream stream =
                new DataOutputStream(
                        new java.io.BufferedOutputStream(Files.newOutputStream(path)))) {
            stream.writeInt(width);
            stream.writeInt(result.rows.size());
            stream.writeInt(result.rows.get(0).length);
            for (int r = 0; r < result.rows.size(); r++) {
                assertEquals(
                        "row " + r + " has a different vocabulary length",
                        result.rows.get(0).length,
                        result.rows.get(r).length);
                for (float value : result.rows.get(r)) {
                    stream.writeFloat(value);
                }
            }
            stream.writeInt(result.tokenIds.size());
            for (int token : result.tokenIds) {
                stream.writeInt(token);
            }
        }
    }
}
