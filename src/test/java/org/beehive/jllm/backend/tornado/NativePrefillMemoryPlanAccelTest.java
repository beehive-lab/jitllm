package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.backend.tornado.memory.TornadoMemoryModel;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.memory.MemoryComponent;
import org.beehive.jllm.runtime.memory.MemoryPlan;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * What the native batch-prefill path costs, predicted against what it was measured to allocate.
 *
 * <h2>The measurement these assertions are calibrated on</h2>
 *
 * <p>{@code -Dtornado.print.bytecodes} on Qwen3-0.6B FP16, context 600, chunk width 128, RTX 5070
 * Ti, TornadoVM develop 65f06c5d1 — distinct {@code (graph, object)} device allocations, native on
 * against native off:
 *
 * <pre>
 *   family                    native off   native on     delta
 *   batch-prefill graphs        914.5        916.5       +  2.0   the staging quartet
 *   decode layer graphs           0.1        560.1       +560.0   five weights x 28 layers
 *   fallback family            absent          0.0       +  0.0   three carriers, nothing uploaded
 *   total                      1212.7       1774.7       +562.0
 * </pre>
 *
 * <p>The stacked copies do not appear as growth in the prefill family: they displace exactly the
 * five originals it stops reading. Those five reappear in the decode family, which uploads them
 * itself. So the path's real cost is one extra copy of the projection weights plus the staging, and
 * that is the difference this test requires the prediction to show.
 *
 * <h2>Why the fallback family must not move the number</h2>
 *
 * <p>It is a third layer graph family and it allocates three carriers. Every weight, workspace and
 * cache it touches is bound from the primary family's buffers. A model that charged a copy of the
 * weights per family would add hundreds of megabytes for graphs that share one, which is why the
 * measured delta above is asserted rather than a family count.
 */
// @formatter:on
public class NativePrefillMemoryPlanAccelTest {

    private static final long MIB = 1048576L;
    private static final int CONTEXT = 600;
    private static final int BATCH = 128;

    /** Stacked {@code [q|k|v]} and {@code [gate|up]} for 28 layers, from the trace above. */
    private static final long MEASURED_STACKED_MIB = 560;

    /** Four half-precision {@code [head][token][headDim]} tensors at width 128. */
    private static final long MEASURED_STAGING_MIB = 2;

    private static final String STACKED = "stacked projection weights (native prefill)";
    private static final String STAGING = "fused attention staging (native prefill)";

    @Test
    public void theNativePathIsPredictedAndCostsWhatItWasMeasuredToCost() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previous = System.getProperty(NativePrefillSupport.PROPERTY);
        try {
            System.setProperty(NativePrefillSupport.PROPERTY, "true");
            assumeTrue(
                    "this host does not select the native prefill projections, so there is no"
                            + " native memory plan to check",
                    NativePrefillSupport.nativeProjections());
            MemoryPlan on = predict(model, BATCH);

            System.setProperty(NativePrefillSupport.PROPERTY, "false");
            MemoryPlan off = predict(model, BATCH);

            // ── the two components exist exactly where the path does ─────────
            assertTrue(
                    "the native path keeps stacked projection weights and the plan does not"
                            + " mention them: "
                            + names(on),
                    component(on, STACKED).isPresent());
            assertTrue(
                    "the fused attention's staging is missing from the plan: " + names(on),
                    component(on, STAGING).isPresent());
            assertFalse(
                    "the generated-kernel path has no stacked weights, yet the plan charges for"
                            + " them: "
                            + names(off),
                    component(off, STACKED).isPresent());
            assertFalse(
                    "the generated-kernel path has no fused attention staging: " + names(off),
                    component(off, STAGING).isPresent());

            // ── and they are the size the trace measured ─────────────────────
            long stackedMib = component(on, STACKED).orElseThrow().logicalBytes() / MIB;
            long stagingMib = component(on, STAGING).orElseThrow().logicalBytes() / MIB;
            assertEquals(
                    "stacked projection weights, against the measured allocation",
                    MEASURED_STACKED_MIB,
                    stackedMib);
            assertEquals(
                    "fused attention staging, against the measured allocation",
                    MEASURED_STAGING_MIB,
                    stagingMib);

            long deltaMib = (on.logicalBytes() - off.logicalBytes()) / MIB;
            assertEquals(
                    "native on and off must differ by exactly what the native path adds, and by"
                            + " nothing the fallback family contributes",
                    MEASURED_STACKED_MIB + MEASURED_STAGING_MIB,
                    deltaMib);
            assertNotEquals(
                    "the two configurations predicted the same budget, so the plan cannot be"
                            + " describing the native path at all",
                    on.predictedBudgetBytes(),
                    off.predictedBudgetBytes());

            // ── the confidence the model is actually entitled to ─────────────
            assertEquals(
                    "the native path's reservation overhead has not been bisected, so its plan"
                            + " must not claim to be exact — only an exact plan refuses a load",
                    MemoryPlan.Confidence.CONSERVATIVE,
                    on.confidence());
        } finally {
            restore(previous);
        }
    }

    // @formatter:off
    /**
     * A budget below the prediction is reported as not fitting, and — because the native path is
     * conservative — is still not grounds for refusing a load.
     *
     * <p>Predictions only: nothing here allocates on the device, which matters on a shared one. The
     * budget is passed to the model rather than to a driver, so a value small enough to prove the
     * comparison works costs nothing to try.
     */
    // @formatter:on
    @Test
    public void aConstrainedBudgetIsReportedButNotEnforcedOnTheNativePath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previous = System.getProperty(NativePrefillSupport.PROPERTY);
        try {
            System.setProperty(NativePrefillSupport.PROPERTY, "true");
            assumeTrue(
                    "this host does not select the native prefill projections",
                    NativePrefillSupport.nativeProjections());

            MemoryPlan generous = predict(model, BATCH, 64L * 1024 * MIB);
            assertTrue("a 64 GiB budget has to fit", generous.fitsConfiguredBudget());

            long tight = generous.predictedBudgetBytes() / 2;
            MemoryPlan constrained = predict(model, BATCH, tight);
            assertFalse(
                    "a budget of half the prediction must not be reported as fitting",
                    constrained.fitsConfiguredBudget());
            assertEquals(
                    "a conservative plan must stay conservative under a tight budget, so that it"
                            + " is reported to a person rather than used to refuse the load",
                    MemoryPlan.Confidence.CONSERVATIVE,
                    constrained.confidence());
        } finally {
            restore(previous);
        }
    }

    private static Optional<MemoryComponent> component(MemoryPlan plan, String name) {
        return plan.components().stream().filter(c -> c.name().equals(name)).findFirst();
    }

    private static String names(MemoryPlan plan) {
        return plan.components().stream().map(MemoryComponent::name).toList().toString();
    }

    private static MemoryPlan predict(Path model, int batch) throws Exception {
        return predict(model, batch, 0L);
    }

    private static MemoryPlan predict(Path model, int batch, long budgetBytes) throws Exception {
        var weights = ModelLoader.weightFootprint(model);
        Model loaded = ModelLoader.loadModel(model, CONTEXT, false, false);
        ExecutionPolicy policy =
                ExecutionPolicy.builder()
                        .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                        .prefillBatchSize(batch)
                        // The plan follows the policy, not the property; the tests toggle the
                        // property, so carry it across.
                        .nativeLibraries(Boolean.getBoolean(NativePrefillSupport.PROPERTY))
                        .build();
        return TornadoMemoryModel.predict(
                weights, loaded.configuration(), policy, TornadoDevices.current(), budgetBytes);
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(NativePrefillSupport.PROPERTY);
        } else {
            System.setProperty(NativePrefillSupport.PROPERTY, previous);
        }
    }

    /** Unused, but keeps the import honest if the list form is ever needed in a message. */
    @SuppressWarnings("unused")
    private static List<MemoryComponent> all(MemoryPlan plan) {
        return plan.components();
    }
}
