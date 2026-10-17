package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
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
 * A {@code qwen35} Q8_0 plan built on Metal must run its batched-prefill projections on the
 * SIMD-group GEMMs, as its Q4_0 projections do.
 *
 * <p>Metal has no int8 tensor-core GEMM and no cuBLAS, so without this every Q8_0 projection fell
 * to the warp-per-output kernel, which reads the whole weight matrix once per prompt row: about 600
 * tok/s at pp512 on Qwen3.5-0.8B against 2,400 for its Q4_0 sibling. A build that went back to it
 * would still generate the same text, so this reads the selection off the built plan: the kernel
 * each layer graph compiles for every projection, and that projection's grid ({@link
 * PlanDispatchEvidence#assertQwen35Q8ProjectionsOnSimdgroup}). Only the 16-output decay and beta
 * projections, which no 64-wide tile fits, may keep the warp kernel.
 *
 * <p>Uses {@code Qwen3.5-0.8B-Q8_0.gguf} under the fixture directory, by name; skips when it is
 * absent or the backend is not Metal. Width 512 is the default chunk; 70 ends inside a tile.
 */
// @formatter:on
public class Qwen35MetalQ8PrefillDispatchAccelTest {

    private static final String MODEL = "Qwen3.5-0.8B-Q8_0.gguf";
    private static final int CONTEXT = 1024;

    @Test
    public void q8ProjectionsRunOnSimdgroupGemmsAtWidth512() throws Exception {
        check(512);
    }

    @Test
    public void q8ProjectionsRunOnSimdgroupGemmsAtWidth70() throws Exception {
        check(70);
    }

    private static void check(int batch) throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve(MODEL);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        String[] keys = {"use.tornadovm", "jitllm.withPrefillDecode", "jitllm.prefillBatchSize"};
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", Integer.toString(batch));
        try {
            BackendId backend = TornadoDevices.current().backend();
            assumeTrue(
                    "this runs on the " + backend + " backend; the selection is Metal's",
                    BackendId.METAL.equals(backend));
            Model model = ModelLoader.loadModel(file, CONTEXT, true, true);
            var config = (Qwen35Configuration) model.configuration();
            State state =
                    State.withStorageOptions(
                            new StorageOptions(DataType.F32, false),
                            () -> State.withPrefillBatchSize(batch, model::createNewState));
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            try {
                assertEquals(
                        "a batched plan",
                        TornadoVMMasterPlanBatchPrefillDecode.class,
                        plan.getClass());
                Map<String, Integer> outputs = new LinkedHashMap<>();
                outputs.put("attn_q_proj", config.queryGateDim());
                outputs.put("attn_k_proj", config.kvDim());
                outputs.put("attn_v_proj", config.kvDim());
                outputs.put("attn_output_proj", config.dim());
                outputs.put("ssm_qkv_proj", config.deltaNetConvDim());
                outputs.put("ssm_gate_proj", config.deltaNetValueDim());
                outputs.put("ssm_out_proj", config.dim());
                outputs.put("ffn_gate_proj", config.hiddenDim());
                outputs.put("ffn_up_proj", config.hiddenDim());
                outputs.put("ffn_down_proj", config.dim());
                int checked =
                        PlanDispatchEvidence.assertQwen35Q8ProjectionsOnSimdgroup(
                                plan,
                                PlanDispatchEvidence.gridSchedulerIfAvailable(plan),
                                batch,
                                outputs,
                                Set.of("attn_output_proj", "ssm_out_proj", "ffn_down_proj"),
                                Set.of("ssm_alpha_proj", "ssm_beta_proj"));
                int attention = config.numberOfAttentionLayers();
                int recurrent = config.numberOfLayers() - attention;
                assertEquals(
                        "projection tasks checked",
                        4 * attention + 3 * recurrent + 3 * config.numberOfLayers(),
                        checked);
                System.out.println(
                        "[qwen35-metal-q8-prefill] width "
                                + batch
                                + ": "
                                + checked
                                + " Q8_0 projections on SIMD-group GEMMs");
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
