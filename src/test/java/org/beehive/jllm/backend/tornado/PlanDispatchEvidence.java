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
     * The immutable task graphs of a plan shape this seam knows, or {@code null} for any other.
     *
     * <p>Same package-private reach as {@link #gridSchedulerIfAvailable}, for evidence that is
     * about the graphs themselves rather than about what they dispatch.
     */
    public static java.util.List<uk.ac.manchester.tornado.api.ImmutableTaskGraph>
            forwardPlanIfAvailable(TornadoVMMasterPlan plan) {
        if (plan instanceof TornadoVMMasterPlanBatchPrefillDecode batched) {
            return batched.batchPrefillDecodeForwardPlan.getImmutableTaskGraphs();
        }
        if (plan instanceof TornadoVMMasterPlanPrefillDecode prefillDecode) {
            return prefillDecode.prefillDecodeForwardPlan.getImmutableTaskGraphs();
        }
        if (plan instanceof TornadoVMMasterPlanSingleToken singleToken) {
            return singleToken.tornadoVMForwardPlan.getImmutableTaskGraphs();
        }
        return null;
    }

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

    // @formatter:off
    /**
     * How a Qwen3 FP16 plan laid out its <b>decode</b> layer graphs, read off its grid scheduler.
     *
     * <p>Grid keys are {@code graphName.taskName}, and a grouped family names its graph after the
     * first layer in the group and prefixes the later slots' tasks. So the number of distinct
     * {@code layer_<n>} graphs is the number of submissions a token costs, and the highest slot
     * prefix is how many layers share one. A property cannot say this: it says what was asked for,
     * and the family folds the answer once at construction.
     */
    // @formatter:on
    public record DecodeGrouping(int layerGraphs, int layersPerGraph) {}

    private static final Pattern DECODE_GRAPH = Pattern.compile("^layer_(\\d+)\\.(?:l(\\d+)_)?.*$");

    /** Reads {@link DecodeGrouping} off a built plan's scheduler. */
    public static DecodeGrouping qwen3DecodeGrouping(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        TreeSet<Integer> graphs = new TreeSet<>();
        int maxSlot = 0;
        for (String task : new TreeSet<>(scheduler.keySet())) {
            Matcher m = DECODE_GRAPH.matcher(task);
            if (!m.matches()) {
                continue;
            }
            graphs.add(Integer.parseInt(m.group(1)));
            if (m.group(2) != null) {
                maxSlot = Math.max(maxSlot, Integer.parseInt(m.group(2)));
            }
        }
        assertTrue("this plan has no decode layer graph at all", !graphs.isEmpty());
        return new DecodeGrouping(graphs.size(), maxSlot + 1);
    }

    // @formatter:off
    /**
     * How a plan's vocabulary projection is configured to reduce, read off its grid scheduler.
     *
     * <p>{@code logits.vocab_proj} carries the same task name either way, so the name says nothing.
     * The worker grid does: the shuffle-reducing kernel assumes exactly one 32-lane workgroup per
     * output row, while the shared-memory kernel scales the local size by {@code
     * THREAD_SCALE_FOR_LOGITS}. So a local size of 32 is the witness that {@code
     * LogitsFP16Layer.useSimd32Reduction()} answered yes in the process that built the plan.
     *
     * @param localWork the configured local work size of {@code logits.vocab_proj}
     * @param globalWork its configured global work size, which is one workgroup per vocabulary row
     */
    // @formatter:on
    public record VocabularyProjection(long localWork, long globalWork) {

        /** One 32-lane workgroup per row is the shuffle-reducing kernel's contract. */
        public boolean shuffleReduced() {
            return localWork == 32;
        }
    }

    /** Reads {@link VocabularyProjection} off a built plan's scheduler. */
    public static VocabularyProjection qwen3VocabularyProjection(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        WorkerGrid grid = scheduler.get("logits.vocab_proj");
        assertNotNull(
                "this plan has no logits.vocab_proj grid: " + new TreeSet<>(scheduler.keySet()),
                grid);
        return new VocabularyProjection(grid.getLocalWork()[0], grid.getGlobalWork()[0]);
    }

    // @formatter:off
    /**
     * Which reduction each Qwen3 FP16 decode layer installed, read off the plan's grid scheduler by
     * task name.
     *
     * <p>The four matrix-vector kernels a decode layer runs come in two forms that compute the same
     * thing by reducing differently, and they used to share both a task name and a worker grid,
     * which made the selection unobservable. {@code Qwen3FP16FFNLayers.reductionVariant} now
     * suffixes the shuffle-reducing form with {@code _warp}, so the scheduler names which kernel
     * the plan actually contains -- per layer and per kernel, not inferred from something else.
     *
     * @param layers how many distinct decode layers were seen
     * @param shuffleReduced task names ending in {@code _warp}, across all layers
     * @param sharedMemory the same four names without the suffix, across all layers
     */
    // @formatter:on
    public record LayerReduction(int layers, int shuffleReduced, int sharedMemory) {

        /** The four kernels a decode layer selects by reduction strategy. */
        public static final List<String> KERNELS =
                List.of(
                        "attn_rms_qkv_projection",
                        "attn_output_proj",
                        "rms_ffn_gate_up",
                        "ffn_down_proj");

        /** Every layer installed all four shuffle-reducing kernels and none of their twins. */
        public boolean allShuffleReduced() {
            return layers > 0 && sharedMemory == 0 && shuffleReduced == layers * KERNELS.size();
        }

        public String describe() {
            return "layers="
                    + layers
                    + " shuffleReduced="
                    + shuffleReduced
                    + " sharedMemory="
                    + sharedMemory
                    + " (expected "
                    + layers * KERNELS.size()
                    + " and 0)";
        }
    }

    private static final Pattern DECODE_TASK =
            Pattern.compile("^layer_(\\d+)\\.(?:l(\\d+)_)?(.+?)(_warp)?$");

    /** Reads {@link LayerReduction} off a built plan's scheduler. */
    public static LayerReduction qwen3DecodeLayerReduction(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        TreeSet<String> layerSlots = new TreeSet<>();
        int warp = 0;
        int shared = 0;
        for (String task : new TreeSet<>(scheduler.keySet())) {
            Matcher m = DECODE_TASK.matcher(task);
            if (!m.matches() || !LayerReduction.KERNELS.contains(m.group(3))) {
                continue;
            }
            layerSlots.add(m.group(1) + "/" + (m.group(2) == null ? "0" : m.group(2)));
            if (m.group(4) != null) {
                warp++;
            } else {
                shared++;
            }
        }
        assertTrue(
                "this plan has no Qwen3 decode layer tasks at all: "
                        + new TreeSet<>(scheduler.keySet()),
                !layerSlots.isEmpty());
        return new LayerReduction(layerSlots.size(), warp, shared);
    }

    // @formatter:off
    /**
     * Which decode attention kernel a Qwen3 FP16 plan installed, and on what grid.
     *
     * <p>The lane-cooperative kernel and the per-key kernel it replaces do not merely differ in
     * speed: they partition the head across lanes differently, so running either on the other's
     * worker grid computes a wrong answer without failing. The task name carries the selection
     * ({@code attention_lane} against {@code attention}) and the local work size carries the shape,
     * and both are asserted, because a plan with the right name and the wrong grid is the failure
     * that would be silent.
     *
     * @param laneTasks decode layers whose attention task is the lane-cooperative one
     * @param perKeyTasks decode layers still on the per-key kernel
     * @param laneLocalWork the local work size registered for the lane task, or 0 if there is none
     */
    // @formatter:on
    public record DecodeAttention(int laneTasks, int perKeyTasks, long laneLocalWork) {

        public String describe() {
            return "lane="
                    + laneTasks
                    + " perKey="
                    + perKeyTasks
                    + " laneLocalWork="
                    + laneLocalWork;
        }
    }

    private static final Pattern DECODE_ATTENTION =
            Pattern.compile("^layer_(\\d+)\\.(?:l(\\d+)_)?attention(_lane)?$");

    /** Reads {@link DecodeAttention} off a built plan's scheduler. */
    public static DecodeAttention qwen3DecodeAttention(GridScheduler scheduler) {
        assertNotNull(
                "no grid scheduler for the plan this run built, so its dispatch cannot be checked",
                scheduler);
        int lane = 0;
        int perKey = 0;
        long laneLocal = 0;
        for (String task : new TreeSet<>(scheduler.keySet())) {
            Matcher m = DECODE_ATTENTION.matcher(task);
            if (!m.matches()) {
                continue;
            }
            if (m.group(3) != null) {
                lane++;
                laneLocal = scheduler.get(task).getLocalWork()[0];
            } else {
                perKey++;
            }
        }
        assertTrue(
                "this plan has no decode attention task at all: "
                        + new TreeSet<>(scheduler.keySet()),
                lane + perKey > 0);
        return new DecodeAttention(lane, perKey, laneLocal);
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
