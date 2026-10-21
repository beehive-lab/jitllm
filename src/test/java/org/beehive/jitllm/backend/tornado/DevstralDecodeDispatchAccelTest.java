package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.ProgramIdentity;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * Which kernels Devstral's decode layers are built with, read off the plan's own grid scheduler.
 *
 * <p>Where the device runs packed integer dot products, the projections must be the {@code dp4a}
 * ones over the activation quantized to Q8 blocks: the quantize tasks present, and every projection
 * on 64-lane rows sized by Devstral's non-square attention ({@code qDim + 2 kvDim} for Q, K and V
 * together, {@code qDim} for Q alone). Over the FP16 cache on the NVIDIA path, decode attention
 * must be split across partitions and merged by a combine task. On any other device the plan must
 * keep the floating-point projections and one attention workgroup per head.
 *
 * <p>Needs the Devstral fixtures under {@code $JITLLM_TEST_MODELS}; each one absent is skipped.
 */
// @formatter:on
public class DevstralDecodeDispatchAccelTest {

    private static final int CONTEXT_LENGTH = 512;
    private static final int PACKED_LOCAL = 64;

    @Test
    public void q8_0DecodeLayersSelectPackedProjectionsAndSplitAttention() throws Exception {
        withModel(
                "Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf",
                (config, state, entries) -> {
                    int qkvRows = config.qDim() + 2 * config.kvDim();
                    if (PackedActivationTasks.enabled()) {
                        assertGrid(entries, "layer_0.attn_out_quantize", config.qDim(), 32);
                        assertGrid(entries, "layer_0.ffn_down_quantize", config.hiddenDim(), 32);
                        assertGrid(
                                entries,
                                "layer_0.qkv_projection",
                                (long) qkvRows * PACKED_LOCAL,
                                PACKED_LOCAL);
                        assertGrid(
                                entries,
                                "layer_0.ffn_gate_up",
                                (long) config.hiddenDim() * PACKED_LOCAL,
                                PACKED_LOCAL);
                        assertGrid(
                                entries,
                                "layer_0.ffn_down_proj",
                                (long) config.dim() * PACKED_LOCAL,
                                PACKED_LOCAL);
                    } else {
                        assertFalse(has(entries, "layer_0.attn_out_quantize"));
                        assertTrue(has(entries, "layer_0.rms_ffn_gate_up"));
                    }
                    assertAttention(config, state, entries);
                });
    }

    @Test
    public void q4_KDecodeLayersSelectPackedProjectionsAndSplitAttention() throws Exception {
        withModel(
                "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
                (config, state, entries) -> {
                    if (PackedActivationTasks.enabled()) {
                        assertGrid(entries, "layer_0.attn_out_quantize", config.qDim(), 32);
                        assertGrid(entries, "layer_0.ffn_down_quantize", config.hiddenDim(), 32);
                        assertGrid(
                                entries,
                                "layer_0.q_proj",
                                (long) config.qDim() * PACKED_LOCAL,
                                PACKED_LOCAL);
                        assertGrid(
                                entries,
                                "layer_0.v_proj",
                                (long) config.kvDim() * PACKED_LOCAL,
                                PACKED_LOCAL);
                        assertGrid(
                                entries,
                                "layer_0.ffn_down_proj",
                                (long) config.dim() * PACKED_LOCAL,
                                PACKED_LOCAL);
                        // A Q4_K_M file keeps gate and up as Q4_K: one fused launch.
                        assertTrue(has(entries, "layer_0.ffn_gate_up"));
                        assertFalse(has(entries, "layer_0.ffn_silu"));
                    } else {
                        assertFalse(has(entries, "layer_0.attn_out_quantize"));
                        assertTrue(has(entries, "layer_0.ffn_silu"));
                    }
                    assertAttention(config, state, entries);
                });
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    @FunctionalInterface
    private interface Check {
        void accept(DevstralConfiguration config, State state, List<String> entries);
    }

    private static void withModel(String file, Check check) throws Exception {
        Path path = GoldenFixture.modelsRoot().resolve(file);
        if (!Files.isRegularFile(path)) {
            assumeTrue(
                    "environment absent: " + file + " under " + GoldenFixture.modelsRoot(), false);
        }
        String previousGpu = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Model model = ModelLoader.loadModel(path, CONTEXT_LENGTH, true, true);
            State state = model.createNewState();
            state.resolveExecutionPolicy(ExecutionPolicy.builder().build());
            // The plan's graphs and grids only: nothing is compiled or uploaded, so the test needs
            // no device memory for the weights.
            List<String> entries =
                    ProgramIdentity.gridEntries(
                            ForwardPlanFactory.createSingleToken(
                                            model.weights().dataType(), state, model)
                                    .getGridScheduler());
            System.out.println("[DEVSTRAL-DISPATCH] " + file + ":");
            entries.stream()
                    .filter(e -> e.startsWith("layer_0.") || e.startsWith("logits."))
                    .forEach(e -> System.out.println("  " + e));
            check.accept((DevstralConfiguration) model.configuration(), state, entries);
        } finally {
            if (previousGpu == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previousGpu);
            }
        }
    }

    /** Split over the FP16 cache where the lane kernel runs for the head width, else per head. */
    private static void assertAttention(
            DevstralConfiguration config, State state, List<String> entries) {
        boolean split =
                state.usesFp16KeyValueCache()
                        && LaneAttentionPolicy.laneCooperativeAttention(config.headSize())
                        && state.executionPolicy().splitKvPartitions().isEmpty()
                        && !state.executionPolicy().packedHalf2Attention()
                        && org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService
                                        .llamaLayerSchedulerType()
                                == org.beehive.jitllm.backend.tornado.scheduling.SchedulerType
                                        .NVIDIA;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            assertEquals(
                    "layer " + layer + " attention combine",
                    split,
                    has(entries, "layer_" + layer + ".attention_combine"));
        }
    }

    private static boolean has(List<String> entries, String task) {
        return entries.stream().anyMatch(e -> e.startsWith(task + " "));
    }

    private static void assertGrid(List<String> entries, String task, long global, long local) {
        String expected = task + " global=[" + global + ", 1, 1] local=[" + local + ", 1, 1]";
        assertTrue(
                "expected "
                        + expected
                        + " among "
                        + entries.stream().filter(e -> e.startsWith(task)).toList(),
                entries.contains(expected));
    }
}
