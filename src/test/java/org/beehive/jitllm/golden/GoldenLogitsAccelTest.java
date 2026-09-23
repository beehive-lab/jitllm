package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Runs only under {@code -Paccel-tests}; excluded from {@code mvn test}, which must never need a
 * model or an accelerator. When the fixture or the accelerator is absent the test <b>skips with an
 * explicit marker</b> and is never reported as a pass.
 *
 * <p>Comparison is bit-exact on the pinned tuple: every compared row's SHA-256 must match, every
 * token id must match, and the retained final row must be identical under {@code
 * Float.floatToRawIntBits}. NaN or Inf anywhere fails immediately, before comparison — a NaN-vs-NaN
 * "match" must not pass.
 */
public class GoldenLogitsAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final Path GOLDEN_ROOT = Paths.get("src/test/resources/goldens");

    @Test
    public void llama3_2_1b_f16_matchesGolden() throws Exception {
        assertGolden(Fixture.LLAMA_3_2_1B_F16);
    }

    @Test
    public void llama3_2_1b_q8_0_matchesGolden() throws Exception {
        assertGolden(Fixture.LLAMA_3_2_1B_Q8_0);
    }

    private void assertGolden(Fixture fixture) throws Exception {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: fixture " + fixture.fileName, false);
        }

        Path goldenDir = GOLDEN_ROOT.resolve(fixture.goldenDirName());
        if (!Files.isDirectory(goldenDir)) {
            System.out.println(
                    "[SKIP] environment absent — no committed golden at "
                            + goldenDir
                            + "; generate with scripts/regenerate-goldens.sh");
            assumeTrue("no committed golden for " + fixture.goldenDirName(), false);
        }

        GoldenRecord expected = GoldenRecord.read(goldenDir);
        GoldenCapture.Result actual = GoldenCapture.capture(model, true);

        // NaN/Inf fails before any comparison.
        for (int r = 0; r < actual.rows.size(); r++) {
            for (float v : actual.rows.get(r)) {
                assertFalse("NaN in produced logits, row " + r, Float.isNaN(v));
                assertFalse("Inf in produced logits, row " + r, Float.isInfinite(v));
            }
        }

        assertEquals("number of compared rows", expected.rowHashes.size(), actual.rows.size());
        assertEquals("number of token ids", expected.tokenIds.size(), actual.tokenIds.size());

        boolean pinnedTuple = onPinnedTuple(expected);
        if (!pinnedTuple) {
            System.out.println(
                    "[WARN] not the pinned tuple ("
                            + expected.metadata.get("device_name")
                            + " / "
                            + expected.metadata.get("tornadovm_version")
                            + "); bit-exactness is not asserted — comparing token ids only.");
        }

        assertEquals("emitted token ids", expected.tokenIds, actual.tokenIds);

        // A configuration whose logits are not reproducible run-to-run cannot be asserted
        // bit-exact. That is recorded in the golden by the generator, which measures it, rather
        // than assumed here. Token ids above are still compared, and NaN/Inf still fails.
        boolean bitExact =
                Boolean.parseBoolean(expected.metadata.getOrDefault("bit_exact", "true"));
        if (!bitExact) {
            // Provisional envelope gate. Token equality alone is NOT accepted as sufficient:
            // on this tuple argmax and top-5 survive while top-10 membership already changes,
            // so greedy decoding hides what top-k/top-p sampling would expose.
            float[] finalRow = actual.rows.get(actual.rows.size() - 1);
            Envelope.Report report = Envelope.compare(expected.finalRow, finalRow);
            System.out.println(
                    "[ENVELOPE] "
                            + expected.metadata.get("quantization")
                            + " recorded bit_exact=false — provisional envelope gate: "
                            + report);
            assertTrue(
                    "envelope exceeded for "
                            + expected.metadata.get("quantization")
                            + " (see verification-gates.md): "
                            + report,
                    report.withinBounds());
        }

        if (pinnedTuple && bitExact) {
            List<String> expectedHashes = expected.rowHashes;
            for (int r = 0; r < actual.rows.size(); r++) {
                assertEquals(
                        "logits row " + r + " differs from golden",
                        expectedHashes.get(r),
                        GoldenRecord.hashRow(actual.rows.get(r)));
            }
            float[] finalRow = actual.rows.get(actual.rows.size() - 1);
            assertEquals("final row length", expected.finalRow.length, finalRow.length);
            for (int i = 0; i < finalRow.length; i++) {
                assertEquals(
                        "final row element " + i + " differs bitwise",
                        Float.floatToRawIntBits(expected.finalRow[i]),
                        Float.floatToRawIntBits(finalRow[i]));
            }
        }
    }

    /** Bit-exactness is asserted only on the tuple the golden was recorded on. */
    private static boolean onPinnedTuple(GoldenRecord golden) {
        String recordedVersion = golden.metadata.getOrDefault("tornadovm_version", "");
        String actualVersion = TupleInfo.tornadoVmVersion();
        String recordedDevice = golden.metadata.getOrDefault("device_name", "");
        String actualDevice = TupleInfo.deviceName();
        return recordedVersion.equals(actualVersion)
                && (recordedDevice.isEmpty() || recordedDevice.equals(actualDevice));
    }

    @Test
    public void goldensDeclareTheirExecutionFlags() throws Exception {
        Path dir = GOLDEN_ROOT.resolve(Fixture.LLAMA_3_2_1B_F16.goldenDirName());
        assumeTrue("no committed golden yet", Files.isDirectory(dir));
        GoldenRecord g = GoldenRecord.read(dir);
        assertEquals(
                "goldens must be recorded with bailout disabled (C4)",
                "false",
                g.metadata.get("recover_bailout"));
        assertEquals(
                "goldens must be recorded with host-side sampling",
                "false",
                g.metadata.get("device_sample"));
        assertTrue("prompt must be recorded verbatim", g.metadata.containsKey("prompt"));
    }
}
