package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

// @formatter:off
/**
 * A {@code qwen35} plan built on Metal must run its decode attention on the lane-cooperative
 * 256-wide split-KV kernels, over the FP32 cache and over the FP16 one.
 *
 * <p>The kernel it replaces gives each query head one threadgroup and walks every position in
 * 8-position tiles: eight threadgroups for the whole GPU on Qwen3.5-0.8B, and a decode that fell
 * from about 100 to 30 tok/s by depth 680. A build that quietly went back to it would still
 * generate the same text, so this reads the selection and its geometry off the built plan's own
 * grid scheduler ({@link PlanDispatchEvidence#assertQwen35LaneHead256Attention}): a split and a
 * combine in every attention layer, {@code heads * splits} groups of {@code
 * HEAD256_WARPS_PER_GROUP} warps.
 *
 * <p>Uses {@code Qwen3.5-0.8B-Q4_0.gguf} under the fixture directory, by name; skips when it is
 * absent or the backend is not Metal.
 */
// @formatter:on
public class Qwen35MetalDecodeAttentionDispatchAccelTest {

    private static final String MODEL = "Qwen3.5-0.8B-Q4_0.gguf";
    private static final int CONTEXT = 1024;
    private static final int BATCH = 128;

    @Test
    public void fp32CacheDecodeAttentionIsSplitOnMetal() throws Exception {
        check(DataType.F32);
    }

    @Test
    public void fp16CacheDecodeAttentionIsSplitOnMetal() throws Exception {
        check(DataType.F16);
    }

    private static void check(DataType keyValue) throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve(MODEL);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        String[] keys = {"use.tornadovm", "jitllm.withPrefillDecode", "jitllm.prefillBatchSize"};
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", Integer.toString(BATCH));
        try {
            BackendId backend = TornadoDevices.current().backend();
            assumeTrue(
                    "this runs on the " + backend + " backend; the selection is Metal's",
                    BackendId.METAL.equals(backend));
            Model model = ModelLoader.loadModel(file, CONTEXT, true, true);
            var config = (Qwen35Configuration) model.configuration();
            State state =
                    State.withStorageOptions(
                            new StorageOptions(keyValue, false),
                            () -> State.withPrefillBatchSize(BATCH, model::createNewState));
            assertEquals(
                    "the cache precision this case is about",
                    keyValue == DataType.F16,
                    state.usesFp16KeyValueCache());
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            try {
                int splits =
                        LaneAttentionPolicy.head256Splits(
                                Qwen35Configuration.DECODE_ATTENTION_SPLITS);
                List<String> tasks =
                        PlanDispatchEvidence.assertQwen35LaneHead256Attention(
                                PlanDispatchEvidence.gridSchedulerIfAvailable(plan),
                                config.numberOfHeads(),
                                splits,
                                config.numberOfAttentionLayers());
                System.out.println(
                        "[qwen35-metal-attention] "
                                + keyValue
                                + " cache, "
                                + plan.executionInfo().mode()
                                + ": "
                                + tasks.size()
                                + " lane-cooperative split attention tasks, "
                                + splits
                                + " splits");
            } finally {
                plan.freeTornadoExecutionPlan();
            }
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
