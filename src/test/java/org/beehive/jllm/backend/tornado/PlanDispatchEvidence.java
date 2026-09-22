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
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoTaskGraphInterface;
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

    /**
     * Asserts that every batched attention-output projection in {@code scheduler} runs as the
     * dequantize-then-GEMM pair: its {@code _dequant} task present with one lane per weight, and
     * the projection on the FP16 GEMM's two-dimensional grid rather than the direct kernel's.
     */
    public static void assertQwen35AttentionOutputOnDequantGemm(
            GridScheduler scheduler, int batchSize, int dim, int attnDim) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        List<Integer> layers = attentionLayers(scheduler);
        assertTrue("no batched attention-output projection in this plan", !layers.isEmpty());
        for (int layer : layers) {
            String prefix = "batchLayer_" + layer + ".";
            WorkerGrid dequant = scheduler.get(prefix + "attn_output_proj_dequant");
            assertNotNull("layer " + layer + " has no attn_output_proj_dequant task", dequant);
            // The Q4_0 decoder takes one lane per packed byte: half the elements; the int8
            // investigation's decoder a lane per word of four.
            long attnLanes = dequant.getGlobalWork()[0];
            assertTrue(
                    "layer " + layer + " dequantization lanes " + attnLanes,
                    attnLanes == (long) dim * attnDim / 2 || attnLanes == (long) dim * attnDim / 4);
            WorkerGrid gemm = scheduler.get(prefix + "attn_output_proj");
            assertEquals(
                    "layer " + layer + " GEMM rows of work",
                    (batchSize / 128) * 256L,
                    gemm.getGlobalWork()[0]);
            assertEquals(
                    "layer " + layer + " GEMM column tiles", dim / 128L, gemm.getGlobalWork()[1]);
            assertEquals("layer " + layer + " GEMM local", 256L, gemm.getLocalWork()[0]);
        }
    }

    /**
     * Asserts that every batched ssm_out projection in {@code scheduler} runs as the
     * dequantize-then-GEMM pair: its {@code _dequant} task with one lane per weight and the
     * projection on the FP16 GEMM's two-dimensional grid.
     */
    public static void assertQwen35SsmOutOnDequantGemm(
            GridScheduler scheduler, int batchSize, int dim, int valueDim) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        List<Integer> layers = new ArrayList<>();
        for (String task : new TreeSet<>(scheduler.keySet())) {
            if (task.matches("batchLayer_\\d+\\.ssm_out_proj")) {
                layers.add(Integer.parseInt(task.replaceAll("\\D", "")));
            }
        }
        assertTrue("no batched ssm_out projection in this plan", !layers.isEmpty());
        for (int layer : layers) {
            String prefix = "batchLayer_" + layer + ".";
            WorkerGrid dequant = scheduler.get(prefix + "ssm_out_proj_dequant");
            assertNotNull("layer " + layer + " has no ssm_out_proj_dequant task", dequant);
            assertEquals(
                    "layer " + layer + " ssm_out dequantization lanes",
                    (long) dim * valueDim / 2,
                    dequant.getGlobalWork()[0]);
            WorkerGrid gemm = scheduler.get(prefix + "ssm_out_proj");
            assertEquals(
                    "layer " + layer + " ssm_out GEMM rows of work",
                    (batchSize / 128) * 256L,
                    gemm.getGlobalWork()[0]);
            assertEquals(
                    "layer " + layer + " ssm_out GEMM column tiles",
                    dim / 128L,
                    gemm.getGlobalWork()[1]);
        }
    }

    /**
     * Asserts that every batched ffn_down projection in {@code scheduler} — the Q4_1 blocks and the
     * Q4_0 ones alike — runs as the dequantize-then-GEMM pair.
     */
    public static void assertQwen35FfnDownOnDequantGemm(
            GridScheduler scheduler, int batchSize, int dim, int hiddenDim) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        int found = 0;
        for (String task : new TreeSet<>(scheduler.keySet())) {
            if (task.matches("batchLayer_\\d+\\.ffn_down_proj")) {
                WorkerGrid dequant = scheduler.get(task + "_dequant");
                assertNotNull(task + " has no dequantization task", dequant);
                // Both decoders take a lane per packed byte (the int8 investigation's a lane
                // per four); assertQwen35DequantGemmPairs pins the kernel names.
                long lanes = dequant.getGlobalWork()[0];
                assertTrue(
                        task + " dequantization lanes " + lanes,
                        lanes == (long) dim * hiddenDim / 2 || lanes == (long) dim * hiddenDim / 4);
                WorkerGrid gemm = scheduler.get(task);
                assertEquals(
                        task + " GEMM rows of work",
                        (batchSize / 128) * 256L,
                        gemm.getGlobalWork()[0]);
                assertEquals(task + " GEMM column tiles", dim / 128L, gemm.getGlobalWork()[1]);
                found++;
            }
        }
        assertTrue("no batched ffn_down projection in this plan", found > 0);
    }

    /**
     * The kernel method a task of a batched plan was built with, read off the plan's own task
     * graphs: for every graph holding a task named {@code task}, the name of the Java method it
     * compiles. The grid scheduler cannot tell two kernels of the same task name and geometry
     * apart; the task graph can. Reaches the graph behind {@link ImmutableTaskGraph} by reflection,
     * which is what the API offers nothing public for.
     */
    public static java.util.Set<String> batchedTaskKernels(TornadoVMMasterPlan plan, String task) {
        java.util.Set<String> kernels = batchedTaskKernelsIfAny(plan, task);
        assertTrue("no batched layer graph holds a task named " + task, !kernels.isEmpty());
        return kernels;
    }

    /**
     * {@link #batchedTaskKernels} without the presence assertion: empty when no batched layer graph
     * holds the task, which a family other than qwen35 legitimately is.
     */
    public static java.util.Set<String> batchedTaskKernelsIfAny(
            TornadoVMMasterPlan plan, String task) {
        assertTrue(
                "not a batched plan: " + plan.getClass().getSimpleName(),
                plan instanceof TornadoVMMasterPlanBatchPrefillDecode);
        var batched = (TornadoVMMasterPlanBatchPrefillDecode) plan;
        java.util.Set<String> kernels = new TreeSet<>();
        try {
            java.lang.reflect.Field field = ImmutableTaskGraph.class.getDeclaredField("taskGraph");
            field.setAccessible(true);
            java.lang.reflect.Field impl = TaskGraph.class.getDeclaredField("taskGraphImpl");
            impl.setAccessible(true);
            for (ImmutableTaskGraph immutable :
                    batched.batchPrefillDecodeForwardPlan.getImmutableTaskGraphs()) {
                TaskGraph graph = (TaskGraph) field.get(immutable);
                if (!graph.getTaskGraphName().startsWith("batchLayer_")) {
                    continue;
                }
                var found = ((TornadoTaskGraphInterface) impl.get(graph)).getTask(task);
                if (found != null) {
                    kernels.add(found.getTaskName());
                }
            }
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("cannot reach the task graphs behind the plan", e);
        }
        return kernels;
    }

    /**
     * The kernel method each batched layer graph compiles for a task named {@code task}, by graph
     * name; graphs without the task are absent. See {@link #batchedTaskKernels}.
     */
    public static java.util.Map<String, String> batchedTaskKernelsByGraph(
            TornadoVMMasterPlan plan, String task) {
        assertTrue(
                "not a batched plan: " + plan.getClass().getSimpleName(),
                plan instanceof TornadoVMMasterPlanBatchPrefillDecode);
        var batched = (TornadoVMMasterPlanBatchPrefillDecode) plan;
        java.util.Map<String, String> kernels = new java.util.TreeMap<>();
        try {
            java.lang.reflect.Field field = ImmutableTaskGraph.class.getDeclaredField("taskGraph");
            field.setAccessible(true);
            java.lang.reflect.Field impl = TaskGraph.class.getDeclaredField("taskGraphImpl");
            impl.setAccessible(true);
            for (ImmutableTaskGraph immutable :
                    batched.batchPrefillDecodeForwardPlan.getImmutableTaskGraphs()) {
                TaskGraph graph = (TaskGraph) field.get(immutable);
                if (!graph.getTaskGraphName().startsWith("batchLayer_")) {
                    continue;
                }
                var found = ((TornadoTaskGraphInterface) impl.get(graph)).getTask(task);
                if (found != null) {
                    kernels.put(graph.getTaskGraphName(), found.getTaskName());
                }
            }
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("cannot reach the task graphs behind the plan", e);
        }
        return kernels;
    }

    /**
     * Asserts the producer and consumer of every dequantize-then-GEMM pair in a batched plan agree
     * on the scratch's layout, layer by layer: every pair is its format's paired-nibble tiled
     * decoder with the tiled-B GEMM, and no other combination exists. Every {@code *_dequant} task
     * in the plan's own scheduler is examined, so a pair this test does not know of fails rather
     * than passing unexamined. Returns the count of each combination seen, keyed {@code
     * decoder+gemm}.
     */
    public static java.util.Map<String, Integer> assertQwen35DequantGemmPairs(
            TornadoVMMasterPlan plan, GridScheduler scheduler) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        // A pair's GEMM is the tiled-B GEMM with one of its epilogues: a store (gate, k, v), the
        // residual add (attention output, ffn_down, ssm_out) or SwiGLU (up).
        java.util.Set<String> allowed =
                java.util.Set.of(
                        "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledB",
                        "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledBResidual",
                        "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledBSwiGLU",
                        "dequantizeQ4_1ToFP16TiledPairs+gemmMMATiledBResidual",
                        "dequantizeQ5_KToFP16TiledPairs+gemmMMATiledBResidual",
                        // The int8 pair (Q4_0 projections on a tensor-core device).
                        "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaled",
                        "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaledResidual",
                        "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaledSwiGLU");
        java.util.Map<String, Integer> seen = new java.util.TreeMap<>();
        java.util.Set<String> tasks = new TreeSet<>();
        for (String key : scheduler.keySet()) {
            if (key.matches("batchLayer_\\d+\\..*_dequant")) {
                tasks.add(key.substring(key.indexOf('.') + 1));
            }
        }
        assertTrue("no dequantize-then-GEMM pair in this plan", !tasks.isEmpty());
        for (String dequantTask : tasks) {
            String gemmTask = dequantTask.substring(0, dequantTask.length() - "_dequant".length());
            java.util.Map<String, String> decoders = batchedTaskKernelsByGraph(plan, dequantTask);
            java.util.Map<String, String> gemms = batchedTaskKernelsByGraph(plan, gemmTask);
            assertTrue("no graph holds " + dequantTask, !decoders.isEmpty());
            for (var entry : decoders.entrySet()) {
                String gemm = gemms.get(entry.getKey());
                assertNotNull(entry.getKey() + " has " + dequantTask + " but no " + gemmTask, gemm);
                String pair = entry.getValue() + "+" + gemm;
                assertTrue(
                        entry.getKey()
                                + "."
                                + gemmTask
                                + " pairs "
                                + pair
                                + ", not a layout the"
                                + " scratch has a producer and consumer for",
                        allowed.contains(pair));
                seen.merge(pair, 1, Integer::sum);
            }
        }
        return seen;
    }

    /**
     * Asserts that the batched F32 alpha and beta projections of every recurrent layer run as the
     * warp-per-tile kernel: {@code batchedMatVecF32WarpTile} by name off the task graphs, and a
     * grid of {@code ceil(batchSize / 4) * (valueHeads / 4) * 32} lanes in 128-lane blocks.
     */
    public static void assertQwen35AlphaBetaOnWarpMatVec(
            TornadoVMMasterPlan plan, GridScheduler scheduler, int batchSize, int valueHeads) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        for (String task : new String[] {"ssm_alpha_proj", "ssm_beta_proj"}) {
            assertEquals(
                    task + " kernel",
                    java.util.Set.of("batchedMatVecF32WarpTile"),
                    batchedTaskKernels(plan, task));
            int found = 0;
            for (String key : new TreeSet<>(scheduler.keySet())) {
                if (key.matches("batchLayer_\\d+\\." + task)) {
                    WorkerGrid grid = scheduler.get(key);
                    assertEquals("value heads divide into 4-wide tiles", 0, valueHeads % 4);
                    assertEquals(
                            key + " global",
                            (long) ((batchSize + 3) / 4) * (valueHeads / 4) * 32,
                            grid.getGlobalWork()[0]);
                    assertEquals(key + " local", 128L, grid.getLocalWork()[0]);
                    found++;
                }
            }
            assertTrue("no batched " + task + " in this plan", found > 0);
        }
    }

    /**
     * Asserts the batched Q4_0 key and value projections' dispatch: on the pair (decoder task, GEMM
     * grid of {@code batchSize / 128 * 256} by {@code kvDim / 128}, kernels by name) when {@code
     * pair}, else on the direct kernel with its grid and no decoder task.
     */
    public static void assertQwen35KvProjectionDispatch(
            TornadoVMMasterPlan plan,
            GridScheduler scheduler,
            int batchSize,
            int kvDim,
            boolean pair) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        for (String task : new String[] {"attn_k_proj", "attn_v_proj"}) {
            int found = 0;
            for (String key : new TreeSet<>(scheduler.keySet())) {
                if (!key.matches("batchLayer_\\d+\\." + task)) {
                    continue;
                }
                found++;
                WorkerGrid grid = scheduler.get(key);
                WorkerGrid dequant = scheduler.get(key + "_dequant");
                if (pair) {
                    assertNotNull(key + " has no decoder task", dequant);
                    long kvLanes = dequant.getGlobalWork()[0];
                    assertTrue(
                            key + " decoder lanes " + kvLanes,
                            kvLanes == (long) kvDim * 5120 / 2
                                    || kvLanes == (long) kvDim * 5120 / 4);
                    assertEquals(
                            key + " GEMM rows of work",
                            (batchSize / 128) * 256L,
                            grid.getGlobalWork()[0]);
                    assertEquals(key + " GEMM column tiles", kvDim / 128L, grid.getGlobalWork()[1]);
                } else {
                    assertTrue(key + " has a decoder task", dequant == null);
                    assertEquals(key + " direct local", 32L, grid.getLocalWork()[0]);
                }
            }
            assertTrue("no batched " + task + " in this plan", found > 0);
            java.util.Set<String> kernels = batchedTaskKernels(plan, task);
            assertEquals(
                    task + " kernel",
                    pair
                            ? java.util.Set.of(
                                    org.beehive.jllm.backend.tornado.TensorCoreSupport
                                                    .isInt8MmaCapable()
                                            ? "gemmInt8BlockScaled"
                                            : "gemmMMATiledB")
                            : java.util.Set.of("projectionMMAQ4_0Prefetch"),
                    kernels);
        }
    }

    /**
     * Asserts that every batched delta-rule scan in {@code scheduler} is the shared-state form:
     * 32-lane groups, one per (head, 32 columns), rather than the per-lane scan's 128-lane groups.
     */
    public static void assertQwen35BatchDeltaRuleShared(
            GridScheduler scheduler, int valueHeads, int stateDim) {
        assertQwen35BatchDeltaRuleGrid(scheduler, (long) valueHeads * stateDim, 32L);
    }

    /**
     * Asserts that every batched delta-rule scan in {@code scheduler} is the warp-per-column form:
     * a warp per column, four columns to a 128-lane group.
     */
    public static void assertQwen35BatchDeltaRuleWarp(
            GridScheduler scheduler, int valueHeads, int stateDim) {
        assertQwen35BatchDeltaRuleGrid(scheduler, (long) valueHeads * stateDim * 32, 128L);
    }

    private static void assertQwen35BatchDeltaRuleGrid(
            GridScheduler scheduler, long global, long local) {
        assertNotNull("no grid scheduler for the plan this run built", scheduler);
        int found = 0;
        for (String task : new TreeSet<>(scheduler.keySet())) {
            if (task.matches("batchLayer_\\d+\\.ssm_delta_rule")) {
                WorkerGrid grid = scheduler.get(task);
                assertEquals(task + " global work", global, grid.getGlobalWork()[0]);
                assertEquals(task + " local work", local, grid.getLocalWork()[0]);
                found++;
            }
        }
        assertTrue("no batched delta-rule scan in this plan", found > 0);
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
