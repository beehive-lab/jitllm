package org.beehive.jllm.golden;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jllm.golden.GoldenFixture.Fixture;

/**
 * Writes the committed goldens. Invoked only by {@code scripts/regenerate-goldens.sh}, never by a
 * test: goldens must never be regenerated automatically on failure.
 */
public final class GenerateGoldens {

    /** Extra captures compared against the first one before recording {@code bit_exact}. */
    private static final int REPEAT_CAPTURES = 2;

    public static void main(String[] args) throws Exception {
        String commit = System.getProperty("golden.commit", "unknown");
        Path outRoot = Paths.get(System.getProperty("golden.out", "src/test/resources/goldens"));

        GoldenCapture.assertHostLogitsAvailable();
        if (!TupleInfo.acceleratorPresent()) {
            throw new IllegalStateException(
                    "no TornadoVM device available — goldens must be recorded on the pinned tuple");
        }

        List<Fixture> fixtures = new ArrayList<>();
        for (Fixture f : Fixture.values()) {
            fixtures.add(f);
        }

        for (Fixture fixture : fixtures) {
            Path model = GoldenFixture.locate(fixture);
            if (model == null) {
                throw new IllegalStateException(GoldenFixture.absentMessage(fixture));
            }
            System.out.println("capturing " + fixture.fileName + " ...");

            String actualSha = GoldenFixture.sha256(model);
            if (!actualSha.equals(fixture.sha256)) {
                throw new IllegalStateException(
                        "fixture sha256 mismatch for "
                                + fixture.fileName
                                + "\n  pinned:   "
                                + fixture.sha256
                                + "\n  on disk:  "
                                + actualSha);
            }

            GoldenCapture.Result r = GoldenCapture.capture(model, true);
            if (r.rows.size() != GoldenCapture.TOKENS) {
                throw new IllegalStateException(
                        "expected "
                                + GoldenCapture.TOKENS
                                + " compared rows, captured "
                                + r.rows.size());
            }
            for (int i = 0; i < r.rows.size(); i++) {
                for (float v : r.rows.get(i)) {
                    if (Float.isNaN(v) || Float.isInfinite(v)) {
                        throw new IllegalStateException(
                                "NaN/Inf in captured logits at row "
                                        + i
                                        + " — refusing to write a golden");
                    }
                }
            }

            List<String> hashes = new ArrayList<>();
            r.rows.forEach(row -> hashes.add(GoldenRecord.hashRow(row)));

            // Reproducibility is measured, not assumed. Capturing a second time and comparing is
            // the only way to know whether bit-exactness is actually a property of this tuple for
            // this configuration — and it makes the golden self-correcting: when a
            // non-deterministic path is fixed, the next regeneration records bit_exact=true
            // without anyone editing a flag.
            boolean bitExact = true;
            for (int repeat = 1; repeat <= REPEAT_CAPTURES; repeat++) {
                GoldenCapture.Result again = GoldenCapture.capture(model, true);
                if (!r.tokenIds.equals(again.tokenIds)) {
                    throw new IllegalStateException(
                            "token ids are not reproducible for "
                                    + fixture.fileName
                                    + " — the golden would be meaningless");
                }
                for (int i = 0; i < r.rows.size() && bitExact; i++) {
                    bitExact = hashes.get(i).equals(GoldenRecord.hashRow(again.rows.get(i)));
                }
            }
            if (!bitExact) {
                System.out.println(
                        "  WARNING: "
                                + fixture.quantization
                                + " logits are NOT bit-reproducible run-to-run on this tuple;"
                                + " recording bit_exact=false (token ids are still compared)");
            }

            Map<String, String> meta = new LinkedHashMap<>();
            meta.put("model_file", fixture.fileName);
            meta.put("model_sha256", actualSha);
            meta.put("quantization", fixture.quantization);
            meta.put("prompt", GoldenCapture.PROMPT);
            meta.put("context_length", Integer.toString(GoldenCapture.CONTEXT_LENGTH));
            meta.put("tokens_compared", Integer.toString(r.rows.size()));
            meta.put("sampling", "greedy-argmax");
            meta.put("vocab_size", Integer.toString(r.rows.get(0).length));
            meta.put("backend", TupleInfo.backend());
            meta.put("device_name", TupleInfo.deviceName());
            meta.put("driver", TupleInfo.driver());
            meta.put("tornadovm_version", TupleInfo.tornadoVmVersion());
            meta.put(
                    "recover_bailout",
                    Boolean.toString(
                            Boolean.parseBoolean(
                                    System.getProperty("tornado.recover.bailout", "false"))));
            meta.put("device_sample", Boolean.toString(Boolean.getBoolean("jllm.deviceSample")));
            // Recorded as measured. This was hardcoded to false while GPU execution was
            // non-deterministic — a few repeats could not distinguish "reproducible" from "won the
            // race this time" (Q8_0 passed the repeat check and still diverged about 1 run in 4
            // under device-memory pressure). That defect is fixed: the RMS reduction no longer
            // combines per-workgroup partial sums without synchronization.
            //
            // The repeats below are a cheap guard, not the determinism gate. The gate is
            // golden/DivergenceRate over >= 300 identical executions; run it before trusting a
            // bit_exact=true recorded here.
            meta.put("bit_exact", Boolean.toString(bitExact));
            meta.put("captures_compared", Integer.toString(REPEAT_CAPTURES + 1));
            meta.put("payload", "row-hashes + token-ids + final row (see GoldenRecord)");
            meta.put("created_by_commit", commit);

            Path dir = outRoot.resolve(fixture.goldenDirName());
            new GoldenRecord(meta, hashes, r.tokenIds, r.rows.get(r.rows.size() - 1)).write(dir);
            System.out.println("  wrote " + dir + " (" + hashes.size() + " rows)");
        }
        System.out.println("done");
    }

    private GenerateGoldens() {}
}
