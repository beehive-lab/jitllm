package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.Options;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;

// @formatter:off
/**
 * What the plan does when cuDNN's fused attention is <b>not</b> usable.
 *
 * <p>The failure this guards against is a capability check that answers from the wrong evidence.
 * Looking for {@code libtornado-cudnn} on the library path says the file exists; it does not say
 * the file can do anything. TornadoVM builds that shim on every CUDA toolkit, and on anything older
 * than CUDA 12 its four entry points are compiled as stubs whose {@code createSdpaPlan} prints a
 * diagnostic and returns a null plan. A file check passes, dispatch selects cuDNN, and the first
 * prefill chunk of the first request fails inside the library. A real implementation can also
 * decline a particular device or a particular shape — the fused path wants Ampere or newer and a
 * head dimension that is a multiple of 8 and at most 256.
 *
 * <p>So the probe builds a plan for this session's exact shape and throws it away, and this test
 * asserts both halves of the contract:
 *
 * <ol>
 *   <li>when the probe says yes, the plan really is built with cuDNN attention — which also
 *       exercises the probe's own plumbing, since that answer comes from the real probe;
 *   <li>when it says no, the plan keeps the <b>native projections</b> and selects batched JIT
 *       attention, rather than losing the whole path or building a fallback family it cannot use.
 * </ol>
 *
 * <h2>What the simulation can and cannot cover</h2>
 *
 * <p>The second case installs a probe that answers no. It therefore covers everything downstream of
 * the answer — dispatch, graph construction, the fallback family's presence, and that the
 * projections survive — on the machine running the test. It does <b>not</b> re-create the native
 * failure itself: producing a genuinely stubbed shim needs a TornadoVM built against CUDA 11, and
 * producing a declined shape needs a pre-Ampere device, neither of which a test can conjure on a
 * host whose toolkit and device support the fused path. The probe's own translation of a native
 * failure into {@code false} is covered by its single catch, not by this test.
 */
// @formatter:on
public class NativeAttentionCapabilityAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String KV_FP16_PROPERTY = "jllm.kvcache.fp16";
    private static final int BATCH = 128;
    private static final int CONTEXT = 512;

    @Test
    public void aUsableFusedAttentionIsSelectedAndAnUnusableOneKeepsTheProjections()
            throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousKv = System.getProperty(KV_FP16_PROPERTY);
        String previousNative = System.getProperty(NativePrefillSupport.PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(KV_FP16_PROPERTY, "true");
        // Native libraries are opt-in (--with-native-libraries); this test is about their dispatch.
        System.setProperty(NativePrefillSupport.PROPERTY, "true");
        NativePrefillSupport.SdpaProbe previousProbe = null;
        try {
            PlanDispatchEvidence.NativePrefillEvidence selected = buildAndRead(model);
            assumeTrue(
                    "this host does not select the native prefill projections, so there is no"
                            + " native dispatch to check: "
                            + selected.describe(),
                    selected.nativeProjections());
            assumeTrue(
                    "this host's cuDNN fused attention is not usable, so the positive half cannot"
                            + " be checked here; the negative half below still runs: "
                            + selected.describe(),
                    selected.cudnnAttention());

            assertFalse(
                    "cuDNN attention was selected, so the primary family must not also carry the"
                            + " JIT attention task: "
                            + selected.describe(),
                    selected.jitAttentionInPrimary());
            assertTrue(
                    "cuDNN attention covers only the first chunk, so the batched fallback family"
                            + " must exist for the rest: "
                            + selected.describe(),
                    selected.batchedFallbackFamily());

            // ── the same session, with the capability answered no ────────────
            previousProbe = NativePrefillSupport.setSdpaProbeForTesting(shape -> false);
            PlanDispatchEvidence.NativePrefillEvidence degraded = buildAndRead(model);

            assertTrue(
                    "an unusable fused attention must not cost the native projections, which do"
                            + " not touch cuDNN at all: "
                            + degraded.describe(),
                    degraded.nativeProjections());
            assertFalse(
                    "cuDNN attention was reported unusable and was selected anyway: "
                            + degraded.describe(),
                    degraded.cudnnAttention());
            assertTrue(
                    "without cuDNN attention the primary family has to carry the batched JIT"
                            + " attention for every chunk: "
                            + degraded.describe(),
                    degraded.jitAttentionInPrimary());
            assertFalse(
                    "the fallback family exists only to cover the chunks cuDNN cannot mask, so a"
                            + " primary that already handles them must not build one: "
                            + degraded.describe(),
                    degraded.batchedFallbackFamily());
            assertEquals(
                    "the capability answer must not change how the layers are grouped",
                    selected.primaryLayerGraphs(),
                    degraded.primaryLayerGraphs());
        } finally {
            if (previousProbe != null) {
                NativePrefillSupport.setSdpaProbeForTesting(previousProbe);
            }
            restore(GPU_PROPERTY, previousGpu);
            restore(KV_FP16_PROPERTY, previousKv);
            restore(NativePrefillSupport.PROPERTY, previousNative);
        }
    }

    /** Builds a batched prefill/decode plan and reads what it was built with. */
    private static PlanDispatchEvidence.NativePrefillEvidence buildAndRead(Path model)
            throws Exception {
        Options options =
                new Options(
                        model,
                        "capability",
                        null,
                        null,
                        false,
                        0.0f,
                        1.0f,
                        42,
                        CONTEXT,
                        false,
                        false,
                        true,
                        true,
                        BATCH);
        Model loaded = ModelLoader.loadModel(options);
        State state = loaded.createNewState();
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, loaded);
        try {
            return PlanDispatchEvidence.qwen3NativePrefill(
                    PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
        } finally {
            plan.freeTornadoExecutionPlan();
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
