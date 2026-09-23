package org.beehive.jitllm.golden;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.memory.TornadoMemoryModel;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.memory.BufferClass;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * Each is the smallest {@code -Dtornado.device.memory} at which the whole selected path runs,
 * bisected to a window under 1.3%, in a <b>fresh JVM per probe</b> so the backend's buffer
 * recycling cannot carry state between them. Recorded 2026-09-03, {@code Llama-3.2-1B-Instruct},
 * ctx 512, CUDA, RTX 5090 Laptop, TornadoVM 5.2.1-jdk21-dev:
 *
 * <pre>
 *   dtype  mode                 highest failing   lowest succeeding
 *   F16    single-token              2350 MiB          2362 MiB
 *   F16    sequential prefill        2356 MiB          2365 MiB
 *   F16    batched prefill (8)       4225 MiB          4234 MiB
 *   Q8_0   single-token              1240 MiB          1256 MiB
 *   Q8_0   batched prefill (8)       2252 MiB          2262 MiB
 * </pre>
 *
 * <p>These are <b>machine- and version-specific</b>, which is why this is an accelerator test with
 * a recorded tuple rather than a portable unit test. It is a check that the model's shape is right,
 * not a portable contract.
 *
 * <h2>The two things asserted, and why both</h2>
 *
 * <ol>
 *   <li><b>Never under-predict.</b> A prediction below the measured minimum would admit a load that
 *       then dies part-allocated, which is the whole failure the preflight exists to prevent. This
 *       is the assertion that matters.
 *   <li><b>Within 5% above.</b> A predictor that simply returned a huge number would satisfy the
 *       first and be useless.
 * </ol>
 */
public class MemoryPlanAccuracyAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final long MIB = 1048576L;

    /** Lowest repeatably successful budget, per the campaign in the class javadoc. */
    private record Threshold(
            Fixture fixture, ExecutionPolicy policy, String label, long lowestOkMib) {}

    @Test
    public void theModelNeverUnderPredictsAndStaysWithinFivePercent() throws Exception {
        Threshold[] cases = {
            new Threshold(Fixture.LLAMA_3_2_1B_F16, singleToken(), "F16 single-token", 2375),
            new Threshold(Fixture.LLAMA_3_2_1B_F16, sequentialPrefill(), "F16 sequential", 2390),
            new Threshold(Fixture.LLAMA_3_2_1B_F16, batchedPrefill(8), "F16 batched(8)", 2399),
            new Threshold(Fixture.LLAMA_3_2_1B_Q8_0, singleToken(), "Q8_0 single-token", 1277),
            new Threshold(Fixture.LLAMA_3_2_1B_Q8_0, batchedPrefill(8), "Q8_0 batched(8)", 1298),
        };

        StringBuilder report = new StringBuilder("\n");
        boolean anyRan = false;
        for (Threshold t : cases) {
            Path path = GoldenFixture.locate(t.fixture());
            if (path == null) {
                continue;
            }
            anyRan = true;
            MemoryPlan plan = predict(path, t.policy());
            long predictedMib = plan.predictedBudgetBytes() / MIB;
            double errorPct = 100.0 * (predictedMib - t.lowestOkMib()) / t.lowestOkMib();
            report.append(
                    String.format(
                            "  %-20s measured %5d MiB   predicted %5d MiB   %+6.2f%%%n",
                            t.label(), t.lowestOkMib(), predictedMib, errorPct));

            assertTrue(
                    t.label()
                            + ": the prediction ("
                            + predictedMib
                            + " MiB) must not be below the measured minimum successful budget ("
                            + t.lowestOkMib()
                            + " MiB). Under-predicting admits a load that then"
                            + " dies part-allocated."
                            + report,
                    predictedMib >= t.lowestOkMib());
            assertTrue(
                    t.label()
                            + ": the prediction must be within 5% above the measured minimum,"
                            + " and was "
                            + String.format("%+.2f%%", errorPct)
                            + ". A prediction that is merely large is not a plan."
                            + report,
                    errorPct <= 5.0);
        }
        assumeTrue("no fixture present", anyRan);
        System.out.println("prediction accuracy:" + report);
    }

    /**
     * A representation the device has no kernel for is predicted as what it <b>becomes</b>.
     *
     * <p>Metal parity task 12/13. {@code loadTornadoTensor} materializes Q4_0/Q4_K/Q5_K/Q6_K as
     * Q8_0, so a Q4_K file's weights occupy roughly twice their file size once loaded. {@code
     * weightFootprint} measured the file type, so it under-predicted exactly those models — the one
     * direction this class's headline assertion says a preflight must never be wrong. Found on the
     * real Devstral fixture: predicted 13.5 GiB, then died materializing Q8_0 with {@code
     * OutOfMemoryError. at TornadoTensorLoader.dequantizeToQ8_0}.
     *
     * <p>Asserted against the ratio rather than a byte count, so it holds for any Q4_K fixture:
     * Q4_K stores ~4.5 bits per weight and Q8_0 ~8.5, so the materialized footprint must be
     * substantially larger than the file's own — never equal to it, which is what the bug looked
     * like.
     */
    @Test
    public void aRepresentationWithNoDeviceKernelIsPredictedAsWhatItBecomes() throws Exception {
        Path path =
                Path.of(
                        System.getProperty("user.home"),
                        "LLMModels",
                        "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf");
        if (!java.nio.file.Files.exists(path)) {
            System.out.println("[SKIP] environment absent — no Q4_K fixture at " + path);
            assumeTrue("environment absent: no Q4_K fixture", false);
        }
        var footprint = ModelLoader.weightFootprint(path);
        long predicted = footprint.perLayerBytes() + footprint.globalBytes();
        long fileBytes = java.nio.file.Files.size(path);

        System.out.printf(
                "Q4_K materialization: file %.1f MiB, predicted device %.1f MiB (%.2fx)%n",
                fileBytes / (double) MIB, predicted / (double) MIB, predicted / (double) fileBytes);

        assertTrue(
                "a Q4_K model's device footprint must be predicted as the Q8_0 it is materialized"
                        + " into, which is substantially larger than the file; predicted "
                        + predicted / MIB
                        + " MiB against a "
                        + fileBytes / MIB
                        + " MiB file",
                predicted > fileBytes * 3 / 2);
    }

    /**
     * The multiplicity must be visible in the breakdown, not folded into a global fudge factor —
     * and it is one for the weights of a batched plan now: its decode graphs consume the copy its
     * prefill graphs uploaded. Measured (2026-09-23, TornadoVM 7.0.1-dev): Llama-3.2-1B F16
     * batched(8) runs in 2399 MiB against 2375 MiB single-token, where a second weight copy would
     * have needed ~4.2 GB; the difference is the 128-row tensor-core staging.
     */
    @Test
    public void batchedPrefillChargesItsStagingNotASecondWeightCopy() throws Exception {
        Path path = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (path == null) {
            assumeTrue("environment absent", false);
        }
        MemoryPlan single = predict(path, singleToken());
        MemoryPlan batched = predict(path, batchedPrefill(8));

        assertTrue("single-token duplicates nothing", single.duplicationBytes() == 0);
        assertTrue(
                "batched prefill shares one weight copy between its graph families; reported "
                        + batched.duplicationBytes() / MIB
                        + " MiB of duplication",
                batched.duplicationBytes() == 0);
        long staging =
                batched.components().stream()
                        .filter(c -> c.bufferClass() == BufferClass.BATCH_STAGING)
                        .mapToLong(c -> c.logicalBytes())
                        .sum();
        assertTrue(
                "the extra is the padded staging, which is what was measured; reported "
                        + staging / MIB
                        + " MiB",
                staging / MIB >= 10 && staging / MIB <= 40);
        System.out.println(batched.describe());
    }

    /**
     * Each component against the storage actually allocated for it.
     *
     * <p><b>Required because the total cannot carry this.</b> Weights are 98.9% of the footprint,
     * so a 5% total passes with an arbitrarily wrong workspace — a component that is 30x too small
     * moves the total by less than a percent. Each class is therefore measured against the real
     * arrays, deduplicated <b>by segment address</b>: the tied embedding and output weights are two
     * wrappers over one segment, and counting wrapper identities inflates the weights by 501 MiB.
     */
    @Test
    public void eachComponentMatchesTheStorageActuallyAllocated() throws Exception {
        Path path = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (path == null) {
            assumeTrue("environment absent", false);
        }
        System.setProperty("use.tornadovm", "true");
        Model model = ModelLoader.loadModel(path, 512, true, true);
        var state = model.createNewState();

        long measuredWeights = distinctSegmentBytes(model.weights());
        long measuredWorkspace = distinctSegmentBytes(state.workspace);

        MemoryPlan plan = predict(path, singleToken());
        long predictedWeights =
                plan.components().stream()
                        .filter(c -> c.bufferClass().name().startsWith("WEIGHTS"))
                        .mapToLong(c -> c.logicalBytes())
                        .sum();
        // The workspace measurement includes the key/value cache, which the plan reports as its own
        // component; compare the sum so the two sides cover the same storage.
        long predictedWorkspaceAndKv =
                plan.components().stream()
                        .filter(c -> !c.bufferClass().name().startsWith("WEIGHTS"))
                        .mapToLong(c -> c.logicalBytes())
                        .sum();

        System.out.printf(
                "%n  weights        measured %10d  predicted %10d  %+6.2f%%%n",
                measuredWeights,
                predictedWeights,
                100.0 * (predictedWeights - measuredWeights) / measuredWeights);
        System.out.printf(
                "  workspace+kv   measured %10d  predicted %10d  %+6.2f%%%n",
                measuredWorkspace,
                predictedWorkspaceAndKv,
                100.0 * (predictedWorkspaceAndKv - measuredWorkspace) / measuredWorkspace);

        assertWithin("weights", measuredWeights, predictedWeights, 1.0);
        // Both bands are far wider than the measured error (-0.01% and +0.02%), and deliberately
        // so: they are there to catch a structural mistake — a class omitted, a dtype assumed, a
        // context length ignored — not to pin arithmetic that will drift with a family addition.
        // The workspace band is the looser of the two because it is derived from the transformer's
        // dimensions rather than from a field table, so a family that adds a buffer under-counts.
        assertWithin("workspace+kv", measuredWorkspace, predictedWorkspaceAndKv, 5.0);
    }

    private static void assertWithin(
            String what, long measured, long predicted, double tolerancePct) {
        double errorPct = 100.0 * Math.abs(predicted - measured) / measured;
        assertTrue(
                what
                        + ": predicted "
                        + predicted
                        + " B against measured "
                        + measured
                        + " B is "
                        + String.format("%.2f%%", errorPct)
                        + " off, tolerance "
                        + tolerancePct
                        + "%",
                errorPct <= tolerancePct);
    }

    /** Distinct native segments reachable from {@code root}, deduplicated by segment address. */
    private static long distinctSegmentBytes(Object root) {
        java.util.Set<Long> addresses = new java.util.HashSet<>();
        java.util.IdentityHashMap<Object, Boolean> seen = new java.util.IdentityHashMap<>();
        java.util.ArrayDeque<Object> queue = new java.util.ArrayDeque<>();
        queue.add(root);
        long total = 0;
        while (!queue.isEmpty()) {
            Object o = queue.poll();
            if (o == null || seen.putIfAbsent(o, Boolean.TRUE) != null) {
                continue;
            }
            if (o instanceof uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray na) {
                if (addresses.add(na.getSegmentWithHeader().address())) {
                    total += na.getNumBytesOfSegmentWithHeader();
                }
                continue;
            }
            if (o.getClass().isArray()) {
                if (!o.getClass().getComponentType().isPrimitive()) {
                    for (int i = 0; i < java.lang.reflect.Array.getLength(o); i++) {
                        queue.add(java.lang.reflect.Array.get(o, i));
                    }
                }
                continue;
            }
            if (o.getClass().getName().startsWith("java.")) {
                continue;
            }
            for (Class<?> c = o.getClass(); c != null && c != Object.class; c = c.getSuperclass()) {
                for (java.lang.reflect.Field f : c.getDeclaredFields()) {
                    if (java.lang.reflect.Modifier.isStatic(f.getModifiers())) {
                        continue;
                    }
                    try {
                        f.setAccessible(true);
                        queue.add(f.get(o));
                    } catch (ReflectiveOperationException | RuntimeException ignored) {
                        // absent from the total rather than fatal; a systematic omission would show
                        // up as a large negative error against the prediction
                    }
                }
            }
        }
        return total;
    }

    private static MemoryPlan predict(Path path, ExecutionPolicy policy) throws Exception {
        var weights = ModelLoader.weightFootprint(path);
        Model model = ModelLoader.loadModel(path, 512, false, false);
        return TornadoMemoryModel.predict(
                weights, model.configuration(), policy, TornadoDevices.current(), 0);
    }

    private static ExecutionPolicy singleToken() {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.SINGLE_TOKEN)
                .build();
    }

    private static ExecutionPolicy sequentialPrefill() {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(1)
                .build();
    }

    private static ExecutionPolicy batchedPrefill(int batch) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(batch)
                .build();
    }
}
