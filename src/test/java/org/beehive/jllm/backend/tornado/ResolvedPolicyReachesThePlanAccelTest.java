package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.golden.ProgramIdentity;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.policy.ExecutionPolicy.SamplingResidency;
import org.junit.Test;

/**
 * A migration off a {@code static final} is only done if the new value actually selects the kernel.
 * So this builds two plans from two policies in one JVM — which the old constant made impossible —
 * and asserts the device argmax task is present in one and absent in the other.
 */
public class ResolvedPolicyReachesThePlanAccelTest {

    private static final int CONTEXT_LENGTH = 512;
    private static final String ARGMAX_TASK = "logits.argmax_sample";

    @Test
    public void bothResidenciesReachThePlan() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Model loaded = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);

            var host = gridEntries(loaded, SamplingResidency.HOST);
            var device = gridEntries(loaded, SamplingResidency.DEVICE);

            // Entries read "name global=[…] local=[…]", so the task is matched by prefix.
            assertFalse(
                    "a host-sampling plan must not contain the device argmax",
                    contains(host, ARGMAX_TASK));
            assertTrue(
                    "a device-sampling plan must contain it — otherwise the resolved policy"
                            + " never reached the task graph and this migration proved nothing",
                    contains(device, ARGMAX_TASK));
            assertEquals(
                    "and nothing else about the graph changes", host.size() + 1, device.size());
        } finally {
            if (previousGpu == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previousGpu);
            }
        }
    }

    /**
     * Split-KV attention adds a combine task per layer. The partition <b>count</b> is not policy
     * and is not asserted here: it sizes the scratch array, so it belongs to whoever allocates What
     * is asserted is that turning the selection on changes the graph.
     *
     * <p>{@code DeviceCapability.SPLIT_KV_ATTENTION} is deliberately withheld on Metal (see {@link
     * TornadoDevices#capabilitiesOf}: {@code processHeadsFlashAttentionSplitKV} fails to JIT
     * there), so asserting that "on" always adds the combine task would assume every device is
     * NVIDIA-class. The assertion follows the capability instead.
     */
    @Test
    public void bothSplitKvSettingsReachThePlan() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        boolean splitKvSupported =
                TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SPLIT_KV_ATTENTION);
        String previousGpu = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Model loaded = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);

            var off = gridEntries(loaded, ExecutionPolicy.builder().build());
            var on = gridEntries(loaded, ExecutionPolicy.builder().splitKvPartitions(8).build());

            boolean combineOff = off.stream().anyMatch(e -> e.contains(".attention_combine"));
            boolean combineOn = on.stream().anyMatch(e -> e.contains(".attention_combine"));
            assertFalse("without split-KV there is no combine task", combineOff);
            if (splitKvSupported) {
                assertTrue(
                        "with it there is — otherwise the resolved policy never reached the"
                                + " graph",
                        combineOn);
            } else {
                assertFalse(
                        "this device lacks DeviceCapability.SPLIT_KV_ATTENTION; requesting"
                                + " split-KV here must not silently add a combine task the device"
                                + " cannot run",
                        combineOn);
            }
        } finally {
            if (previousGpu == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previousGpu);
            }
        }
    }

    /**
     * Storage, not policy: it types the arrays, so it comes from {@code ModelOptions} and every
     * session on a model shares it. What is checked here is the same thing the policy fields get —
     * that the non-default value changes the graph, not only the default.
     *
     * <p>The FP16 key/value path is NVIDIA-only, so on another device this asserts the honest
     * thing: that the state reports what it actually allocated.
     */
    @Test
    public void bothKeyValueRepresentationsReachTheState() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty("use.tornadovm");
        String previousLegacy = System.getProperty("jllm.kvcache.fp16");
        String previousFp32 = System.getProperty("jllm.kvcache.fp32");
        System.setProperty("use.tornadovm", "true");
        try {
            System.clearProperty("jllm.kvcache.fp16");
            System.clearProperty("jllm.kvcache.fp32");
            Model fp16Model = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);
            State fp16State = fp16Model.createNewState();
            assertTrue(
                    "with no property set the state must hold FP16 arrays, the default — otherwise"
                            + " the default never reached allocation and the description would"
                            + " name a dtype the buffers do not have",
                    fp16State.usesFp16KeyValueCache());

            var entries = gridEntries(fp16Model, ExecutionPolicy.builder().build(), fp16State);
            assertTrue(
                    "and the plan built from it must still be a plan",
                    entries.stream().anyMatch(e -> e.contains("logits.vocab_proj")));

            System.setProperty("jllm.kvcache.fp32", "true");
            Model fp32Model = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);
            assertFalse(
                    "with -Djllm.kvcache.fp32=true the state must hold FP32 key/value arrays only",
                    fp32Model.createNewState().usesFp16KeyValueCache());
        } finally {
            restore("use.tornadovm", previousGpu);
            restore("jllm.kvcache.fp16", previousLegacy);
            restore("jllm.kvcache.fp32", previousFp32);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static boolean contains(java.util.List<String> entries, String task) {
        return entries.stream().anyMatch(e -> e.startsWith(task + " ") || e.equals(task));
    }

    private static java.util.List<String> gridEntries(Model model, SamplingResidency residency) {
        return gridEntries(model, ExecutionPolicy.builder().samplingResidency(residency).build());
    }

    private static java.util.List<String> gridEntries(Model model, ExecutionPolicy policy) {
        return gridEntries(model, policy, model.createNewState());
    }

    private static java.util.List<String> gridEntries(
            Model model, ExecutionPolicy policy, State state) {
        state.resolveExecutionPolicy(policy);
        TornadoVMMasterPlanSingleToken plan =
                new TornadoVMMasterPlanSingleToken(state, model, MetricsSink.disabled());
        try {
            return ProgramIdentity.gridEntries(plan.tornadoVMForwardPlan.getGridScheduler());
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }
}
