package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.beehive.jllm.backend.tornado.kernels.Qwen35MMAKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;

// @formatter:off
/**
 * What a built plan is configured to dispatch, read off that plan's own grid scheduler.
 *
 * <p>Evidence about the plan a test built, not a trace of execution: the scheduler holds one entry
 * per task the plan contains, with the geometry that task is configured to launch on. It cannot say
 * a kernel ran. It can say which branch built the graph, which a system property cannot — the
 * property says what was asked for, and the layer class folds it once at initialization.
 *
 * <p>In this package because each master plan's forward plan is reached through a package-private
 * field. Same observation as {@code ProgramIdentity.gridEntries}.
 */
// @formatter:on
public final class PlanDispatchEvidence {

    private static final Pattern ATTENTION_OUTPUT =
            Pattern.compile("^batchLayer_(\\d+)\\.attn_output_proj$");

    private PlanDispatchEvidence() {}

    /**
     * The grid scheduler of a plan shape this seam knows, or {@code null} for any other.
     *
     * <p>Null rather than a throw: callers that record this for whoever may want it must not fail a
     * family whose plan it cannot reach. A caller that needs the evidence asserts on it.
     */
    public static GridScheduler gridSchedulerIfAvailable(TornadoVMMasterPlan plan) {
        if (plan instanceof TornadoVMMasterPlanBatchPrefillDecode batched) {
            return batched.batchPrefillDecodeForwardPlan.getGridScheduler();
        }
        if (plan instanceof TornadoVMMasterPlanPrefillDecode prefillDecode) {
            return prefillDecode.prefillDecodeForwardPlan.getGridScheduler();
        }
        if (plan instanceof TornadoVMMasterPlanSingleToken singleToken) {
            return singleToken.tornadoVMForwardPlan.getGridScheduler();
        }
        return null;
    }

    // @formatter:off
    /**
     * Asserts that this plan's batched attention-output projection is configured for the tensor
     * cores, in every attention layer.
     *
     * <p>Per layer: the {@code attn_output_fp16} and {@code attn_output_residual} tasks that only
     * the MMA branch adds, and the projection's grid — one warp per {@code BM x BN} tile of a
     * {@code batchSize x dim} output. The grid is asserted because a projection left in the
     * tensor-core task map but given the scalar matrix-vector geometry would carry the right task
     * name.
     *
     * @param scheduler the plan's own scheduler; a missing one fails here
     * @param batchSize the prefill width the plan was built for
     * @param dim the model dimension, which is this projection's output width
     */
    // @formatter:on
    public static void assertQwen35AttentionOutputOnTensorCores(
            GridScheduler scheduler, int batchSize, int dim) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        List<Integer> layers = attentionLayers(scheduler);
        assertTrue(
                "no batched attention-output projection in this plan: "
                        + new TreeSet<>(scheduler.keySet()),
                !layers.isEmpty());

        long expectedGlobal =
                (long) (batchSize / Qwen35MMAKernels.BM)
                        * (dim / Qwen35MMAKernels.BN)
                        * Qwen35MMAKernels.LOCAL;
        for (int layer : layers) {
            String prefix = "batchLayer_" + layer + ".";
            assertTrue(
                    "layer "
                            + layer
                            + " built its attention-output projection on the scalar path: the"
                            + " tensor-core branch's conversion and residual tasks are absent",
                    scheduler.keySet().contains(prefix + "attn_output_fp16")
                            && scheduler.keySet().contains(prefix + "attn_output_residual"));

            WorkerGrid grid = scheduler.get(prefix + "attn_output_proj");
            assertEquals(
                    "layer " + layer + " attn_output_proj global work",
                    expectedGlobal,
                    grid.getGlobalWork()[0]);
            assertEquals(
                    "layer " + layer + " attn_output_proj local work",
                    Qwen35MMAKernels.LOCAL,
                    grid.getLocalWork()[0]);
        }
    }

    // @formatter:off
    /**
     * Asserts that this plan's decode attention is the split-KV pair rather than the per-head
     * kernel: the split phase and the combine that merges its partials, in every attention layer
     * that has one.
     *
     * <p>The task names are what distinguishes the two shapes — the per-head kernel writes the
     * attention result itself and has no combine — so a plan that quietly fell back would fail here
     * instead of being measured as though it had not.
     *
     * @param scheduler the plan's own scheduler; a missing one fails here
     */
    // @formatter:on
    public static void assertQwen35SplitKvAttention(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        List<String> attention = new ArrayList<>();
        List<String> combine = new ArrayList<>();
        for (String task : new TreeSet<>(scheduler.keySet())) {
            if (task.endsWith(".attention")) {
                attention.add(task);
            } else if (task.endsWith(".attention_combine")) {
                combine.add(task);
            }
        }
        assertTrue("this plan has no attention task at all", !attention.isEmpty());
        assertEquals(
                "every attention layer needs its combine: attention "
                        + attention.size()
                        + ", combine "
                        + combine.size(),
                attention.size(),
                combine.size());
    }

    // @formatter:off
    /**
     * Which batch-prefill implementation a Qwen3 FP16 plan was built with, read off its own grid
     * scheduler.
     *
     * <p>The scheduler registers a worker grid for every JIT task the plan contains and for none of
     * its library tasks, because a cuBLAS or cuDNN call has no grid to launch. That asymmetry is
     * the evidence: a native projection is present as the <b>absence</b> of a {@code qkvProj} grid,
     * and cuDNN attention as the presence of its three adapter grids where {@code batch_attention}
     * would otherwise be. A system property cannot say this — it says what was asked for, and the
     * capability probe may have answered no.
     */
    // @formatter:on
    public record NativePrefillEvidence(
            int primaryLayerGraphs,
            boolean nativeProjections,
            boolean cudnnAttention,
            boolean jitAttentionInPrimary,
            boolean batchedFallbackFamily) {

        /** Every native component this plan selected, for a message that names what it built. */
        public String describe() {
            return "primaryLayerGraphs="
                    + primaryLayerGraphs
                    + " nativeProjections="
                    + nativeProjections
                    + " cudnnAttention="
                    + cudnnAttention
                    + " jitAttentionInPrimary="
                    + jitAttentionInPrimary
                    + " batchedFallbackFamily="
                    + batchedFallbackFamily;
        }
    }

    private static final Pattern PRIMARY_GRAPH = Pattern.compile("^batchPrefillLayer_(\\d+)\\..*$");

    /** Reads {@link NativePrefillEvidence} off a built plan's scheduler. */
    public static NativePrefillEvidence qwen3NativePrefill(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        TreeSet<String> keys = new TreeSet<>(scheduler.keySet());
        TreeSet<Integer> primaryGraphs = new TreeSet<>();
        boolean cudnn = false;
        boolean jitAttention = false;
        boolean jitProjection = false;
        boolean fallbackFamily = false;
        for (String task : keys) {
            if (task.startsWith("batchPrefillFallbackLayer_")) {
                fallbackFamily = true;
                continue;
            }
            Matcher matcher = PRIMARY_GRAPH.matcher(task);
            if (!matcher.matches()) {
                continue;
            }
            primaryGraphs.add(Integer.parseInt(matcher.group(1)));
            if (task.endsWith("cudnn_pack_q")) {
                cudnn = true;
            } else if (task.endsWith("batch_attention")) {
                jitAttention = true;
            } else if (task.endsWith("qkvProj")
                    || task.endsWith("gateUpProj")
                    || task.endsWith("woProj")
                    || task.endsWith("w2Proj")) {
                jitProjection = true;
            }
        }
        assertTrue(
                "this plan has no batch-prefill layer graph at all: " + keys,
                !primaryGraphs.isEmpty());
        return new NativePrefillEvidence(
                primaryGraphs.size(), !jitProjection, cudnn, jitAttention, fallbackFamily);
    }

    /** The layers whose batched graph holds an attention-output projection, in index order. */
    private static List<Integer> attentionLayers(GridScheduler scheduler) {
        List<Integer> layers = new ArrayList<>();
        for (String task : new TreeSet<>(scheduler.keySet())) {
            Matcher matcher = ATTENTION_OUTPUT.matcher(task);
            if (matcher.matches()) {
                layers.add(Integer.parseInt(matcher.group(1)));
            }
        }
        return layers;
    }

    /**
     * The batched conversion tasks only the MMA branch adds, for a report that names what it built.
     */
    public static List<String> qwen35MmaBatchedTasks(GridScheduler scheduler) {
        List<String> tasks = new ArrayList<>();
        if (scheduler == null) {
            return tasks;
        }
        for (String task : new TreeSet<>(scheduler.keySet())) {
            if (task.startsWith("batchLayer_") && task.endsWith("_fp16")) {
                tasks.add(task);
            }
        }
        return tasks;
    }
}
