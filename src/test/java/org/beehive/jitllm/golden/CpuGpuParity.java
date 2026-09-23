package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/**
 * Runs the same fixture and prompt through the CPU path and the GPU path and compares the logits.
 * The CPU path is the reference, as specified in {@code verification-gates.md} §CPU↔GPU parity.
 * Cross-path comparison is never bit-exact — different orders of the same arithmetic, and on the
 * FP16 path a different storage format for the activations — so it uses tolerances, and NaN/Inf on
 * either side fails.
 *
 * <p><b>Four complementary gates</b>, because one worst-element assertion cannot distinguish a
 * broken kernel from one unlucky tail value out of 8.2 million:
 *
 * <ol>
 *   <li><b>Elementwise</b> {@code |gpu − cpu| ≤ atol + rtol·|cpu|}, the conventional mixed bound
 *       (the same shape as {@code torch.testing.assert_close}), with a small violation budget;
 *   <li><b>Hard ceiling</b> on the single largest absolute error, so the budget can never hide a
 *       large excursion;
 *   <li><b>Whole-vector</b> relative L2 and cosine similarity, which describe the row as a whole
 *       instead of being decided by its most extreme element;
 *   <li><b>Decision-level</b>: argmax agreement and top-k overlap. An argmax reversal is only a
 *       failure when the CPU considered the decision clear-cut — a reversed near-tie is expected
 *       between numerical paths and is reported, not tolerated silently.
 * </ol>
 *
 * <p><b>Thresholds are measured, not assumed</b>, and are per quantization: the FP16 GPU path
 * stores its normalized activations as FP16 while the CPU path keeps FP32, so its floor is set by
 * that format, whereas the Q8_0 path keeps FP32 activations and tracks the CPU far more closely.
 * They come from {@code golden/ParityProfile} on the pinned tuple and sit roughly 2× above the
 * observed worst case. Re-derive them with that tool if the tuple, the prompt or the compared-token
 * count changes; do not widen them to make a failing run pass.
 *
 * <p><b>The two absolute-scale bounds are fractions of the reference's own RMS.</b> A constant
 * {@code atol} does not transfer between families: Granite-3.2-2B's logits have an RMS of 233
 * against Llama-3.2-1B's 2.92, because Granite multiplies its logits by its {@code logit_scale}. A
 * constant calibrated on Llama leaves nearly every Llama element inside {@code atol} and tested
 * only absolutely, while putting nearly every Granite element outside it and tested only relatively
 * — so the same code reads as clean on one family and broken on the other. Normalising holds every
 * family to one standard, and it *tightens* the gate for the families whose logits are small.
 *
 * <p>The FP16 bounds were 5-20x looser until the CPU reference stopped flushing denormal FP16
 * weights to zero. Most of what this gate used to tolerate was error on the reference side, not the
 * GPU's.
 *
 * <p><b>Teacher forcing</b> is what makes the comparison meaningful: greedy decoding is
 * autoregressive, so the first near-tie that tips differently sends the two paths into different
 * contexts and every later row compares unrelated states. Forcing the GPU along the CPU's tokens
 * keeps the KV state identical at every compared position.
 *
 * <p><b>One family per subclass, and therefore per JVM.</b> Device memory a closed session frees
 * returns to TornadoVM's buffer provider but not to the driver, so a class that loads every fixture
 * exhausts the device partway through and the failures land on whichever model happened to run late
 * rather than on whichever one is wrong. Surefire forks per class, so the split is what makes each
 * result attributable.
 */
abstract class CpuGpuParity {

    /**
     * Per-configuration bounds. {@code atol} dominates for the vast majority of logits, which sit
     * near zero; {@code rtol} is what keeps the bound honest on the few large ones.
     */
    record Bounds(
            double atolPerRms,
            double rtol,
            double ceilingPerRms,
            double relL2,
            double minCosine,
            double violationFraction,
            double decisionGap) {}

    static final Bounds FP16 = new Bounds(5e-3, 1e-2, 8e-3, 2e-3, 0.99999, 1e-4, 0.5);
    static final Bounds Q8_0 = new Bounds(1.7e-4, 1e-2, 3.4e-4, 1e-4, 0.999999, 1e-4, 0.5);

    // @formatter:off
    /**
     * For a family whose Q4_0 projections read an <b>eight-bit activation</b>.
     *
     * <p>These bounds are much wider than {@link #Q8_0}'s and they are meant to be: the device is
     * no longer computing the same thing as the host reference to within rounding. It quantizes the
     * normalized activation to eight bits per block of 32 and does the dot product in packed
     * integers, which is what llama.cpp's decode does and what makes {@code dp4a} available at all.
     *
     * <p><b>Sized from measurement, on the 27B, teacher-forced over 63 rows.</b> What it costs:
     * elementwise violations 22.81%, largest absolute difference 0.5227 against a reference RMS of
     * 0.111, relative L2 2.45e-2, cosine 0.99970. What it did not cost, measured at the same time:
     * <b>zero</b> argmax disagreements across those 63 rows, top-5 4.984/5 and top-10 9.968/10, and
     * greedy generation that was token-identical over 120 tokens against the floating-point path.
     *
     * <p>So the bounds below carry roughly a factor of two over the measured magnitudes: the
     * absolute ceiling goes from 3.4e-4 of the reference RMS to 0.32 of it, the relative L2 from
     * 1e-4 to 5e-2, and the elementwise budget from 0.01% of logits to half of them. Those three
     * are weak by construction now, and saying so is the point — they no longer distinguish a
     * defect from the arithmetic, and {@code atol} and {@code rtol} are left where they were so the
     * printed violation count stays a comparable number rather than a redefined one.
     *
     * <p>What still has teeth is the pair that speaks to decisions: {@code minCosine} moves from
     * 0.999999 to 0.9994 — a real weakening, not a formality, and the number to watch — while
     * {@code decisionGap} does not move at all, because an argmax reversal where the reference was
     * not close would still be a defect.
     *
     * <p>The top-k figures say plainly that this path can reorder near-ties. Greedy decoding did
     * not notice; sampling with top-k or top-p can.
     */
    // @formatter:on
    static final Bounds Q8_0_PACKED_ACTIVATION =
            new Bounds(1.7e-4, 1e-2, 0.32, 5e-2, 0.9994, 0.5, 0.5);

    // @formatter:off
    /**
     * For the <b>batched</b> path, whose decode rows carry every packed projection while its prompt
     * goes through the prefill kernels.
     *
     * <p>A third envelope rather than a wider shared one. {@link #Q8_0_PACKED_ACTIVATION} is what
     * the batched path met when its decode rows carried two packed projections; they now carry
     * more, and widening the shared constant would weaken whatever else uses it in order to admit
     * this. {@link #Q8_0_FULLY_PACKED} is a different amount of quantization again -- every
     * position rather than only the decoded ones -- so it is not this.
     *
     * <p><b>Engineering regression limits for an accepted arithmetic, not calibrated quality
     * thresholds.</b> They say "this path still computes what it computed yesterday"; they say
     * nothing about whether what it computes is good. A future failure here is something to
     * investigate, not to widen: the envelope has now been set three times, once per change in what
     * is packed, and each time the numbers were measured first and the limits written after.
     *
     * <p>Measured on the 63-row trace with the packed branch projections, gate/up and ffn_down:
     * largest absolute difference 1.54620 against a reference RMS of 3.272, relative L2 0.0797096,
     * cosine 0.99682787, elementwise 33.13%, argmax 0/63, top-5 4.952/5, top-10 9.952/10.
     */
    // @formatter:on
    static final Bounds Q8_0_PACKED_DECODE = new Bounds(1.7e-4, 1e-2, 0.70, 0.12, 0.9955, 0.5, 0.5);

    // @formatter:off
    /**
     * The same, for the paths that pack the <b>feed-forward</b> activation as well.
     *
     * <p>{@code STANDARD} and sequential prefill quantize the feed-forward's activation at every
     * position, prompt included; the batched path packs only its decode rows and stays inside
     * {@link #Q8_0_PACKED_ACTIVATION}. Two different amounts of quantization are two different
     * envelopes, and widening the shared one to cover this would weaken a path that passes in order
     * to admit a path that does not. Only the two fully-packed cases take this, and each names the
     * mode it runs.
     *
     * <p><b>A fixture-specific regression envelope for an accepted precision tradeoff.</b> Not a
     * quality guarantee, and not a statistically calibrated threshold: it is one 63-row trace of
     * one fixture, and the headroom below is an engineering allowance rather than a measured bound
     * on variability, which has not been established.
     *
     * <p>Measured on that trace, reference RMS 3.272, against the allowance chosen for each:
     *
     * <ul>
     *   <li>largest absolute difference 0.457 of the RMS against {@code 0.70} — about 1.5x, on a
     *       single worst logit out of 15.6 million;
     *   <li>relative L2 0.0571 against {@code 0.075} — about 1.3x, and the most informative of the
     *       three, being an aggregate over a quarter of a million logits per row;
     *   <li>cosine 0.99837 against {@code 0.9975} — about 1.53x the <i>deficit</i> from one, which
     *       is the quantity that matters rather than the ratio of the similarities.
     * </ul>
     *
     * <p>Everything else is {@link #Q8_0_PACKED_ACTIVATION}'s, unchanged: the elementwise budget
     * passes at 33.13% of 50%, and the decision gap does not move, because an argmax reversal where
     * the reference was not close would still be a defect. On this trace there were none — 0/63 —
     * with top-5 4.984/5 and top-10 9.873/10.
     *
     * <p>These limits are not to be loosened again for the next optimization.
     */
    // @formatter:on
    static final Bounds Q8_0_FULLY_PACKED = new Bounds(1.7e-4, 1e-2, 0.70, 0.075, 0.9975, 0.5, 0.5);

    // @formatter:off
    /**
     * For a <b>Q4_0 file read natively whose projections take a block-quantized eight-bit
     * activation</b> — device residency and the packed-integer path together.
     *
     * <p>A fifth envelope, and the reason is the same as every other one here: this is a different
     * amount of quantization, not a looser view of {@link #Q8_0}. The device is no longer computing
     * what the host computes to within rounding — it quantizes the normalized activation to eight
     * bits per block of 32 and does the dot product in packed integers, which is what makes {@code
     * dp4a} available at all.
     *
     * <p><b>Measured on gemma-4-E2B-it-Q4_0.gguf, CUDA, teacher-forced over 63 rows</b>, against
     * the allowance chosen for each: elementwise 1.416% against 3%, largest absolute difference
     * 0.0289 of the reference RMS against 0.06, relative L2 0.00767 against 0.015, cosine deficit
     * 2.08e-5 against 5e-5. Roughly a factor of two on each — an engineering allowance, not a
     * measured bound on variability.
     *
     * <p>What did not move is the part that decides tokens: <b>0 of 63 argmax disagreements</b>,
     * top-5 4.873/5, and greedy generation token-identical to the host reference on three prompts.
     * {@code decisionGap} does not move, because an argmax reversal where the reference was not
     * close would still be a defect.
     *
     * <p><b>The elementwise budget was raised once, from 3% to 5%, and the reason is recorded
     * because the previous revision of this comment said it should not be.</b> Splitting the
     * attention window sixteen ways instead of eight changed the online-softmax merge order and
     * took the elementwise count from 1.416% to 3.153%. Nothing that bounds a magnitude moved with
     * it: the largest absolute difference went from 0.0289 of the reference RMS to 0.0347 against a
     * 0.06 ceiling, relative L2 from 0.00767 to 0.00950 against 0.015, and cosine <i>improved</i>,
     * 0.99997925 to 0.99998472. Argmax stayed at 0 of 63.
     *
     * <p>That pattern is what the elementwise count measures here: {@code atol} is 0.00406 against
     * logits whose bulk sits near zero, so a small shared shift moves many of them across the line
     * without moving the worst excursion, the aggregate, or any decision. It is the weakest of the
     * four gates on a packed path by construction — which is exactly why the other three are quoted
     * above rather than this one. A future failure in {@code ceilingPerRms}, {@code relL2}, {@code
     * minCosine} or {@code decisionGap} is something to investigate, not to widen.
     */
    // @formatter:on
    static final Bounds Q4_0_PACKED_ACTIVATION =
            new Bounds(1.7e-4, 1e-2, 0.06, 0.015, 0.99995, 0.05, 0.5);

    /** The CPU reference against the accelerator running its default single-token path. */
    void assertParity(Fixture fixture, Bounds bounds) throws Exception {
        assertParity(fixture, bounds, 1);
    }

    /**
     * The same comparison with the accelerator driven through batched prefill.
     *
     * <p>Batched prefill is a different program, not a faster one: its own MMA GEMMs for the
     * projections, its own paged-KV attention, its own fused norm kernels. Checking only the
     * single-token path leaves every prompt token a model processes in that mode unverified, which
     * is where a silent numerical defect would sit. The CPU side is unchanged — there is one
     * reference, and both accelerator modes are measured against it.
     *
     * <p>The bounds are the same. A mode that needs looser bounds to pass is a mode that computes
     * something different, and that is the finding, not the configuration.
     */
    GoldenCapture.Result assertParityBatched(Fixture fixture, Bounds bounds, int prefillBatchSize)
            throws Exception {
        return assertParity(fixture, bounds, prefillBatchSize, true);
    }

    /**
     * The same comparison with the accelerator ingesting the prompt as its own sequential phase.
     *
     * <p>{@code PREFILL_DECODE} at a batch of one. It runs the same layer graphs as {@code
     * STANDARD} with the logits graph skipped for prompt positions, so what it can disagree about
     * is the boundary rather than the arithmetic: where decode resumes, and whether anything the
     * prompt left behind — a key/value entry, a convolution window, a recurrent matrix — carried.
     */
    void assertParityPrefillDecode(Fixture fixture, Bounds bounds) throws Exception {
        assertParity(fixture, bounds, 1, true);
    }

    private void assertParity(Fixture fixture, Bounds bounds, int prefillBatchSize)
            throws Exception {
        assertParity(fixture, bounds, prefillBatchSize, false);
    }

    /**
     * @return the accelerator capture, so a caller can assert what its plan was built to dispatch
     */
    private GoldenCapture.Result assertParity(
            Fixture fixture, Bounds bounds, int prefillBatchSize, boolean separatePrefillPhase)
            throws Exception {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: fixture " + fixture.fileName, false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] environment absent — no TornadoVM device");
            assumeTrue("environment absent: no accelerator", false);
        }

        GoldenCapture.Result cpu = GoldenCapture.capture(model, false);
        GoldenCapture.Result gpu =
                GoldenCapture.capture(
                        model, true, cpu.tokenIds, prefillBatchSize, separatePrefillPhase);
        if (separatePrefillPhase) {
            System.out.printf(
                    "  %s, batch %d%n",
                    prefillBatchSize > 1 ? "batched prefill" : "sequential prefill",
                    prefillBatchSize);
        }

        assertEquals("compared row count", cpu.rows.size(), gpu.rows.size());
        // Opt-in: the accelerator's rows as raw little-endian floats, one row after another, so
        // two builds' accelerator outputs can be compared directly with each other rather than
        // only each against the CPU reference (the metrics below measure the latter).
        String dump = System.getProperty("jitllm.parity.dumpRows");
        if (dump != null) {
            dumpRows(gpu.rows, Path.of(dump));
        }

        // The two absolute-scale bounds are fractions of the reference's own RMS, not constants.
        // Families differ by two orders of magnitude here -- Granite-3.2-2B's logits have an RMS
        // of 233 against Llama-3.2-1B's 2.92, because Granite multiplies them by its logit_scale
        // -- so a constant atol tests one family relatively and the other not at all, and reads
        // as a defect in whichever family happens to have the larger logits. Normalised, the
        // measured worst-case error sits in one narrow band across every family on the pinned
        // tuple: Granite 0.0035, Llama 0.0026, Qwen3 0.0019, Phi-3 0.0009. The bounds below are
        // roughly 2x that band, which is the same margin the constants used to carry for Llama.
        double refRms = rms(cpu.rows);
        double atol = bounds.atolPerRms() * refRms;
        double maxAbsCeiling = bounds.ceilingPerRms() * refRms;
        System.out.printf(
                "  reference RMS %.4g -> atol %.4g, max-abs ceiling %.4g%n",
                refRms, atol, maxAbsCeiling);

        long elements = 0;
        long violations = 0;
        double maxAbs = 0;
        double maxRatio = 0;
        int maxAbsRow = -1;
        int maxAbsIndex = -1;
        double worstRelL2 = 0;
        int worstRelL2Row = -1;
        double minCosine = 1;
        int argmaxDisagreements = 0;
        double top5 = 0;
        double top10 = 0;
        List<String> wideReversals = new ArrayList<>();

        for (int r = 0; r < cpu.rows.size(); r++) {
            float[] ref = cpu.rows.get(r);
            float[] got = gpu.rows.get(r);
            assertEquals("vocabulary length, row " + r, ref.length, got.length);
            assertFalse("NaN/Inf in CPU logits, row " + r, Envelope.hasNonFinite(ref));
            assertFalse("NaN/Inf in GPU logits, row " + r, Envelope.hasNonFinite(got));

            double sqDiff = 0;
            double sqRef = 0;
            double sqGot = 0;
            double dot = 0;
            for (int i = 0; i < ref.length; i++) {
                double d = Math.abs((double) ref[i] - (double) got[i]);
                double tol = atol + bounds.rtol() * Math.abs((double) ref[i]);
                if (d > tol) {
                    violations++;
                }
                maxRatio = Math.max(maxRatio, d / tol);
                // Tracked independently of the violation test: when everything passes there is
                // still a largest error, and reporting it as zero would hide the headroom left.
                if (d > maxAbs) {
                    maxAbs = d;
                    maxAbsRow = r;
                    maxAbsIndex = i;
                }
                sqDiff += d * d;
                sqRef += (double) ref[i] * ref[i];
                sqGot += (double) got[i] * got[i];
                dot += (double) ref[i] * (double) got[i];
                elements++;
            }

            double relL2 = Math.sqrt(sqDiff / sqRef);
            if (Boolean.getBoolean("jitllm.parity.rows")) {
                System.out.printf(java.util.Locale.ROOT, "[PARITY-ROW] %d relL2=%.5f%n", r, relL2);
            }
            if (relL2 > worstRelL2) {
                worstRelL2 = relL2;
                worstRelL2Row = r;
            }
            minCosine = Math.min(minCosine, dot / (Math.sqrt(sqRef) * Math.sqrt(sqGot)));

            top5 += overlap(ref, got, 5);
            top10 += overlap(ref, got, 10);

            int aRef = Envelope.argmax(ref);
            int aGot = Envelope.argmax(got);
            if (aRef != aGot) {
                argmaxDisagreements++;
                // The gap between the two tokens that actually competed, on each side. This is the
                // movement the GPU needed to reverse this specific decision — a per-path
                // top1-minus-top2 margin can involve a third token entirely.
                double cpuGap = ref[aRef] - ref[aGot];
                double gpuGap = got[aGot] - got[aRef];
                String line =
                        String.format(
                                "row %d: cpu picks %d, gpu picks %d; cpu gap=%.6g gpu gap=%.6g",
                                r, aRef, aGot, cpuGap, gpuGap);
                System.out.println("  [REVERSAL] " + line);
                if (cpuGap > bounds.decisionGap()) {
                    wideReversals.add(line);
                }
            }
        }

        int rows = cpu.rows.size();
        System.out.printf(
                "[PARITY] %s rows=%d elements=%d%n", fixture.quantization, rows, elements);
        System.out.printf(
                "[PARITY]   elementwise: violations=%d (%.4g%%, budget %.4g%%) worstRatio=%.4g%n",
                violations,
                100.0 * violations / elements,
                100.0 * bounds.violationFraction(),
                maxRatio);
        System.out.printf(
                "[PARITY]   maxAbs=%.6g (row %d, token %d; ceiling %.6g)%n",
                maxAbs, maxAbsRow, maxAbsIndex, maxAbsCeiling);
        System.out.printf(
                "[PARITY]   worst relL2=%.6g (row %d; bound %.6g)  minCosine=%.8f (bound %.8f)%n",
                worstRelL2, worstRelL2Row, bounds.relL2(), minCosine, bounds.minCosine());
        System.out.printf(
                "[PARITY]   argmax disagreements=%d/%d  top5=%.3f/5  top10=%.3f/10%n",
                argmaxDisagreements, rows, top5 / rows, top10 / rows);

        assertTrue(
                String.format(
                        "%s: %d/%d elementwise violations (%.4g%%) exceed the %.4g%% budget"
                                + " at atol=%.3g rtol=%.3g; worst was %.3gx tolerance",
                        fixture.quantization,
                        violations,
                        elements,
                        100.0 * violations / elements,
                        100.0 * bounds.violationFraction(),
                        atol,
                        bounds.rtol(),
                        maxRatio),
                violations <= bounds.violationFraction() * elements);

        assertTrue(
                String.format(
                        "%s: max |cpu-gpu|=%.6g exceeds the ceiling %.6g (row %d, token %d)",
                        fixture.quantization, maxAbs, maxAbsCeiling, maxAbsRow, maxAbsIndex),
                maxAbs <= maxAbsCeiling);

        assertTrue(
                String.format(
                        "%s: relative L2 %.6g exceeds %.6g at row %d",
                        fixture.quantization, worstRelL2, bounds.relL2(), worstRelL2Row),
                worstRelL2 <= bounds.relL2());

        assertTrue(
                String.format(
                        "%s: cosine similarity %.8f is below %.8f",
                        fixture.quantization, minCosine, bounds.minCosine()),
                minCosine >= bounds.minCosine());

        assertTrue(
                String.format(
                        "%s: argmax reversed where the CPU decision was not close"
                                + " (gap > %.3g): %s",
                        fixture.quantization, bounds.decisionGap(), wideReversals),
                wideReversals.isEmpty());
        return gpu;
    }

    /** Number of shared entries between the two top-k sets. */
    /** RMS of the whole reference, which is the scale the absolute bounds are expressed in. */
    private static void dumpRows(List<float[]> rows, Path path) throws java.io.IOException {
        try (var out =
                new java.io.DataOutputStream(
                        new java.io.BufferedOutputStream(
                                java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(rows.size());
            out.writeInt(rows.get(0).length);
            for (float[] row : rows) {
                for (float v : row) {
                    out.writeFloat(v);
                }
            }
        }
        System.out.println("[PARITY] wrote accelerator rows to " + path);
    }

    private static double rms(java.util.List<float[]> rows) {
        double sq = 0;
        long n = 0;
        for (float[] row : rows) {
            for (float v : row) {
                sq += (double) v * v;
                n++;
            }
        }
        return Math.sqrt(sq / n);
    }

    private static int overlap(float[] a, float[] b, int k) {
        int[] ta = topK(a, k);
        int[] tb = topK(b, k);
        int shared = 0;
        for (int x : ta) {
            for (int y : tb) {
                if (x == y) {
                    shared++;
                    break;
                }
            }
        }
        return shared;
    }

    private static int[] topK(float[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Float.compare(v[q], v[p]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }
}
