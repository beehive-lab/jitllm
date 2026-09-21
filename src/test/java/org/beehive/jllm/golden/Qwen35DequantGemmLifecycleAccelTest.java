package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;

// @formatter:off
/**
 * The dequantize-then-GEMM prefill path through a whole session at width 256: two full chunks and a
 * partial tail, several decode steps, a reset, and a different prompt — under CUDA graph replay.
 *
 * <p>The comparison is against the <b>direct quantized path at the same width</b>, selected by a
 * test-only means: the state's dequantization scratch is removed before the plan is built, which is
 * what the dispatch consults, so the direct plan chunks the prompt identically and differs only in
 * the projection kernels. That run happens in a child JVM launched with this one's own JVM
 * arguments and class path, before this JVM builds any plan: device memory a freed plan returns is
 * not given back to the driver until the process exits (see the surefire configuration), so two 27B
 * plans cannot coexist in one process, and the child's exit is what frees its share. Every logits
 * row of the two runs must carry the same raw bits: the pair decodes the halves the direct kernel
 * stages and accumulates each output in the same K order, so the only way the rows can differ is a
 * defect in the pair — a scratch read before it is written, a chunk's inactive rows leaking, a GEMM
 * tile off by a row.
 *
 * <p>Stale state: after the reset, the second prompt's rows must equal those of a <b>fresh</b>
 * state and plan given the same prompt, so nothing the first sequence left in the scratch, the
 * key/value store or the recurrent state can pass. And the two prompts differ, so a plan that
 * ignored its input would fail on the first comparison.
 *
 * <p>The width-128/256 parity tests cover one partial chunk against the host reference with the
 * committed bounds; this test adds the multi-chunk, decode, reset and replay coverage those do not
 * have, at raw-bit strictness against the retained direct path.
 */
// @formatter:on
public class Qwen35DequantGemmLifecycleAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
        System.setProperty("jllm.kvcache.fp16", "true");
        // Graph capture on the first execution, replay on every later chunk and decode step.
        System.setProperty("jllm.cudaGraphs", "true");
    }

    /**
     * The width under test: 256 by default; {@code JLLM_LIFECYCLE_WIDTH} selects another, so a
     * wider width can be validated the same way before it is recommended. The prompts and the
     * context scale with it so prompt A always spans two full chunks and a partial third.
     */
    private static final int WIDTH = width();

    /**
     * On a tensor-core device the Q4_0 pairs are int8, whose arithmetic differs from the direct
     * FP16 path's: prompt A is then compared by replay, and the distance to the direct path is
     * reported rather than asserted.
     */
    private static final boolean INT8 =
            org.beehive.jllm.backend.tornado.TensorCoreSupport.isInt8MmaCapable();

    /** relL2, max |diff| and argmax agreement over the logits rows of two runs. */
    private static void reportRowDistance(String what, List<float[]> a, List<float[]> b) {
        assertEquals(what + ": row counts", a.size(), b.size());
        double num = 0, den = 0, maxAbs = 0;
        int argmaxAgree = 0;
        for (int r = 0; r < a.size(); r++) {
            float[] x = a.get(r);
            float[] y = b.get(r);
            int ax = 0, ay = 0;
            for (int i = 0; i < x.length; i++) {
                double e = (double) x[i] - y[i];
                num += e * e;
                den += (double) y[i] * y[i];
                maxAbs = Math.max(maxAbs, Math.abs(e));
                if (x[i] > x[ax]) {
                    ax = i;
                }
                if (y[i] > y[ay]) {
                    ay = i;
                }
            }
            if (ax == ay) {
                argmaxAgree++;
            }
        }
        System.out.printf(
                java.util.Locale.ROOT,
                "[lifecycle] %s: rows %d relL2 %.4f maxAbs %.4f argmax agreement %d/%d%n",
                what,
                a.size(),
                Math.sqrt(num / den),
                maxAbs,
                argmaxAgree,
                a.size());
    }

    private static final int CONTEXT = 4 * WIDTH;

    private static int width() {
        String env = System.getenv("JLLM_LIFECYCLE_WIDTH");
        return env == null ? 256 : Integer.parseInt(env);
    }

    private static final int DECODE_STEPS = 8;

    /** Long enough for two full 256-token chunks and a partial third. */
    private static final String PROMPT_A =
            repeat(
                            "A matrix multiplication combines two matrices by taking dot products of the"
                                    + " rows of the first with the columns of the second. ",
                            32 * WIDTH / 256)
                    + "Explain what a matrix multiplication is in one paragraph.";

    private static final String PROMPT_B =
            repeat(
                            "The river flows past the old mill, turning the wheel that grinds the grain"
                                    + " the farmers bring each autumn. ",
                            20 * WIDTH / 256)
                    + "Describe the mill in one paragraph.";

    private static String repeat(String s, int n) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            b.append(s);
        }
        return b.toString();
    }

    /** A sequence's rows, plus what was fed: the seed, the ingestion's first index, the prompt. */
    private record Run(
            List<float[]> rows,
            GridScheduler scheduler,
            int seed,
            int firstIndex,
            List<Integer> prompt) {}

    @Test
    public void thePairPathMatchesTheDirectPathAcrossChunksDecodeAndReset() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent", false);
        }
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        assertTrue(TornadoVMMasterPlan.CUDA_GRAPHS);

        String previousPrefill = System.getProperty("jllm.withPrefillDecode");
        String previousBatch = System.getProperty("jllm.prefillBatchSize");
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(WIDTH));
        try {
            // Direct path first, in a child JVM (see the class comment), and before this process
            // loads anything: the weights are copied into anonymous host memory (about 16 GiB per
            // process, plus its caches and scratch), so the two captures must not be resident at
            // once. Sequencing them reduces the peak to one process's; it does not shrink it.
            Run directA = captureDirectInChildJvm(modelPath);

            Model model = ModelLoader.loadModel(modelPath, CONTEXT, true, true);
            List<Integer> promptA = encode(model, PROMPT_A);
            List<Integer> promptB = encode(model, PROMPT_B);
            assertTrue(
                    "prompt A has " + promptA.size() + " tokens; needs > 2 chunks of " + WIDTH,
                    promptA.size() > 2 * WIDTH && promptA.size() % WIDTH != 0);
            assertTrue("prompt B must also span a partial chunk", promptB.size() > WIDTH);

            // Pair path, one state and one plan, three sequences: prompt B on the fresh plan,
            // reset, prompt A (the multi-chunk one, compared with the direct path), reset, prompt
            // B again (compared with its own fresh run, so nothing A left behind can pass).
            State pairState = State.withPrefillBatchSize(WIDTH, model::createNewState);
            // The seed the state was created with, restored on every reset as a session does.
            int initialSeed = pairState.latestToken;
            assertNotNull(
                    "the state did not allocate the dequantization scratch at width " + WIDTH,
                    pairState.workspace.wrapDequantScratchFP16);
            TornadoVMMasterPlan pairPlan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(pairState, model);
            Run freshB;
            Run pairA;
            Run pairB;
            Run pairA2 = null;
            java.util.Set<String> pairKernels;
            java.util.Set<String> alphaBetaKernels;
            java.util.Set<String> scanKernels;
            java.util.Map<String, Integer> dequantPairs;
            try {
                pairKernels = PlanDispatchEvidence.batchedTaskKernels(pairPlan, "attention");
                alphaBetaKernels =
                        PlanDispatchEvidence.batchedTaskKernels(pairPlan, "ssm_alpha_proj");
                alphaBetaKernels.addAll(
                        PlanDispatchEvidence.batchedTaskKernels(pairPlan, "ssm_beta_proj"));
                scanKernels = PlanDispatchEvidence.batchedTaskKernels(pairPlan, "ssm_delta_rule");
                // Every pair's producer and consumer agree on the scratch's layout, the Q4_0
                // tiled pairs interleaved in graph order with the row-major Q4_1 and Q5_K ones.
                dequantPairs =
                        PlanDispatchEvidence.assertQwen35DequantGemmPairs(
                                pairPlan, PlanDispatchEvidence.gridSchedulerIfAvailable(pairPlan));
                freshB = run(model, pairState, pairPlan, promptB);
                reset(pairState, pairPlan, initialSeed);
                pairA = run(model, pairState, pairPlan, promptA);
                reset(pairState, pairPlan, initialSeed);
                pairB = run(model, pairState, pairPlan, promptB);
                if (INT8) {
                    // The int8 research path: prompt A once more after a reset, for the exact
                    // replay comparison the direct FP16 path cannot provide across arithmetics.
                    reset(pairState, pairPlan, initialSeed);
                    pairA2 = run(model, pairState, pairPlan, promptA);
                }
            } finally {
                pairPlan.freeTornadoExecutionPlan();
            }
            var qwen = (org.beehive.jllm.model.qwen35.Qwen35Configuration) model.configuration();
            PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(
                    pairA.scheduler(), WIDTH, qwen.dim(), qwen.attentionOutputInputDim());
            PlanDispatchEvidence.assertQwen35SsmOutOnDequantGemm(
                    pairA.scheduler(), WIDTH, qwen.dim(), qwen.deltaNetValueDim());
            PlanDispatchEvidence.assertQwen35BatchDeltaRuleWarp(
                    pairA.scheduler(), qwen.numberOfValueHeads(), qwen.headValueDim());
            PlanDispatchEvidence.assertQwen35FfnDownOnDequantGemm(
                    pairA.scheduler(), WIDTH, qwen.dim(), qwen.hiddenDim());
            assertEquals(
                    "the batched alpha/beta projections this plan compiled",
                    java.util.Set.of("batchedMatVecF32WarpTile"),
                    alphaBetaKernels);
            assertEquals(
                    "the batched attention kernel this plan compiled",
                    java.util.Set.of("attentionBatchFP16PagedTensorCoreT32"),
                    pairKernels);
            assertEquals(
                    "the batched delta-rule scan this plan compiled",
                    java.util.Set.of("deltaRuleScanWarp"),
                    scanKernels);
            // Three decoders by five epilogues: Q4_0 with the store (gate, k, v), the residual
            // (attention output, Q4_0 ffn_down) and SwiGLU (up); Q4_1 ffn_down and Q5_K ssm_out
            // with the residual. All interleaved through the one scratch in graph order.
            assertEquals(
                    "the pair combinations in this plan: " + dequantPairs,
                    INT8
                            // The int8 investigation replaces the three Q4_0 pairs.
                            ? java.util.Set.of(
                                    "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaled",
                                    "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaledResidual",
                                    "decodeQ4_0ToInt8Tiled+gemmInt8BlockScaledSwiGLU",
                                    "dequantizeQ4_1ToFP16TiledPairs+gemmMMATiledBResidual",
                                    "dequantizeQ5_KToFP16TiledPairs+gemmMMATiledBResidual")
                            : java.util.Set.of(
                                    "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledB",
                                    "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledBResidual",
                                    "dequantizeQ4_0ToFP16TiledPairs+gemmMMATiledBSwiGLU",
                                    "dequantizeQ4_1ToFP16TiledPairs+gemmMMATiledBResidual",
                                    "dequantizeQ5_KToFP16TiledPairs+gemmMMATiledBResidual"),
                    dequantPairs.keySet());
            System.out.println("[lifecycle] dequant pairs " + dequantPairs);

            assertSameInput("pair vs direct, prompt A", pairA, directA);
            if (INT8) {
                // Different arithmetic from the direct FP16 path: report the distance, assert
                // the exact replay instead (same arithmetic, after a reset).
                reportRowDistance(
                        "int8 pair vs direct FP16, prompt A", pairA.rows(), directA.rows());
                assertSameInput("int8 pair replay vs first run, prompt A", pairA2, pairA);
                assertRowsIdentical(
                        "int8 pair replay vs first run, prompt A", pairA2.rows(), pairA.rows());
            } else {
                assertRowsIdentical("pair vs direct, prompt A", pairA.rows(), directA.rows());
            }
            assertSameInput("after reset vs fresh, prompt B", pairB, freshB);
            assertRowsIdentical("after reset vs fresh, prompt B", pairB.rows(), freshB.rows());
            assertTrue(
                    "prompts A and B produced identical first rows, so the input is not reaching"
                            + " the plan",
                    !java.util.Arrays.equals(pairA.rows().get(0), pairB.rows().get(0)));
            System.out.printf(
                    "[LIFECYCLE] width %d: prompt A %d tokens (%d chunks), prompt B %d tokens,"
                            + " %d rows compared each, all raw-bit identical%n",
                    WIDTH,
                    pairA.prompt().size(),
                    (pairA.prompt().size() + WIDTH - 1) / WIDTH,
                    pairB.prompt().size(),
                    pairA.rows().size());
        } finally {
            restore("jllm.withPrefillDecode", previousPrefill);
            restore("jllm.prefillBatchSize", previousBatch);
        }
    }

    /**
     * What a session's reset does (see {@code LegacySessionRuntime.reset}): restore the seed the
     * state was created with, then clear the sequence state on the device through the plan.
     */
    private static void reset(State state, TornadoVMMasterPlan plan, int initialSeed) {
        state.latestToken = initialSeed;
        plan.resetSequenceState();
    }

    /** Two runs meant to be equivalent were fed the same thing: seed, first index and prompt. */
    private static void assertSameInput(String what, Run a, Run b) {
        assertEquals(what + ": seed", a.seed(), b.seed());
        assertEquals(what + ": ingestion first index", a.firstIndex(), b.firstIndex());
        assertEquals(what + ": prompt tokens", a.prompt(), b.prompt());
    }

    private static List<Integer> encode(Model model, String prompt) {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(chatFormat.getBeginOfText());
        }
        tokens.addAll(
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, prompt)));
        tokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        return tokens;
    }

    /** Prefill the prompt, then decode {@link #DECODE_STEPS} greedy tokens, capturing every row. */
    private static Run run(Model model, State state, TornadoVMMasterPlan plan, List<Integer> prompt)
            throws Exception {
        List<float[]> rows = new ArrayList<>();
        Sampler capturing =
                tensor -> {
                    float[] row = new float[tensor.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = tensor.get(i);
                    }
                    rows.add(row);
                    int token = Sampler.TENSOR_ARGMAX.sampleToken(tensor);
                    return token;
                };
        int seed = state.latestToken;
        int firstIndex =
                org.beehive.jllm.inference.PromptIngestion.of(state, prompt, 0).firstIndex();
        // generateTokensGPUPrefillDecode ingests the whole prompt at positions 0..N-1, whatever
        // the first index (a seed the prompt opens with is fed once, as the prompt's own first
        // token), then decodes one row per position while pos < maxTokens: exactly
        // maxTokens - N rows. The context has to hold every one of those positions.
        int budget = prompt.size() + DECODE_STEPS;
        assertTrue("prompt + decode exceed the context of " + CONTEXT, budget <= CONTEXT);
        Set<Integer> stopTokens = Set.of();
        model.generateTokensGPU(state, 0, prompt, stopTokens, budget, capturing, false, null, plan);
        GridScheduler scheduler = PlanDispatchEvidence.gridSchedulerIfAvailable(plan);
        assertEquals("decode rows captured", DECODE_STEPS, rows.size());
        return new Run(rows, scheduler, seed, firstIndex, prompt);
    }

    /** Every row of both runs, raw-bit equal, and the same number of them. */
    private static void assertRowsIdentical(String what, List<float[]> a, List<float[]> b) {
        assertEquals(what + ": row count", a.size(), b.size());
        for (int r = 0; r < a.size(); r++) {
            float[] x = a.get(r);
            float[] y = b.get(r);
            assertEquals(what + ": row " + r + " length", x.length, y.length);
            for (int i = 0; i < x.length; i++) {
                assertTrue(
                        what + ": row " + r + " logit " + i + " not finite", Float.isFinite(x[i]));
                if (Float.floatToRawIntBits(x[i]) != Float.floatToRawIntBits(y[i])) {
                    throw new AssertionError(
                            what
                                    + ": row "
                                    + r
                                    + " logit "
                                    + i
                                    + " differs: "
                                    + x[i]
                                    + " vs "
                                    + y[i]);
                }
            }
        }
    }

    /**
     * Runs {@link #main} in a child JVM with this JVM's arguments and class path: the direct path
     * at {@link #WIDTH}, prompt A, rows written raw to a temporary file.
     */
    private static Run captureDirectInChildJvm(Path modelPath) throws Exception {
        Path out = java.nio.file.Files.createTempFile("qwen35-direct-rows", ".bin");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.addAll(
                java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments());
        command.add("-Djllm.qwen35.tensorCores=true");
        command.add("-Djllm.kvcache.fp16=true");
        command.add("-Djllm.cudaGraphs=true");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(Qwen35DequantGemmLifecycleAccelTest.class.getName());
        command.add(modelPath.toString());
        command.add(out.toString());
        Process child =
                new ProcessBuilder(command)
                        .redirectErrorStream(true)
                        .redirectOutput(ProcessBuilder.Redirect.INHERIT)
                        .start();
        int status = child.waitFor();
        assertEquals("the direct-path capture in the child JVM failed", 0, status);
        Run run = readRun(out);
        java.nio.file.Files.deleteIfExists(out);
        return run;
    }

    /** Child entry: the direct path at {@link #WIDTH} over prompt A; rows to {@code args[1]}. */
    public static void main(String[] args) throws Exception {
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(WIDTH));
        Model model = ModelLoader.loadModel(Path.of(args[0]), CONTEXT, true, true);
        List<Integer> promptA = encode(model, PROMPT_A);
        State directState = State.withPrefillBatchSize(WIDTH, model::createNewState);
        // The test-only selection: no scratch, so the dispatch keeps the direct kernels.
        directState.workspace.wrapDequantScratchFP16 = null;
        TornadoVMMasterPlan directPlan =
                TornadoVMMasterPlan.initializeTornadoVMPlan(directState, model);
        Run directA;
        try {
            directA = run(model, directState, directPlan, promptA);
        } finally {
            directPlan.freeTornadoExecutionPlan();
        }
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                directA.scheduler(), WIDTH, model.configuration().dim());
        writeRun(Path.of(args[1]), directA);
        System.out.printf(
                "[LIFECYCLE] child: direct path at width %d, %d rows written%n",
                WIDTH, directA.rows().size());
        System.exit(0);
    }

    private static void writeRun(Path path, Run run) throws java.io.IOException {
        try (var out =
                new java.io.DataOutputStream(
                        new java.io.BufferedOutputStream(
                                java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(run.seed());
            out.writeInt(run.firstIndex());
            out.writeInt(run.prompt().size());
            for (int token : run.prompt()) {
                out.writeInt(token);
            }
            out.writeInt(run.rows().size());
            for (float[] row : run.rows()) {
                out.writeInt(row.length);
                for (float v : row) {
                    out.writeInt(Float.floatToRawIntBits(v));
                }
            }
        }
    }

    private static Run readRun(Path path) throws java.io.IOException {
        try (var in =
                new java.io.DataInputStream(
                        new java.io.BufferedInputStream(
                                java.nio.file.Files.newInputStream(path)))) {
            int seed = in.readInt();
            int firstIndex = in.readInt();
            int promptCount = in.readInt();
            List<Integer> prompt = new ArrayList<>(promptCount);
            for (int i = 0; i < promptCount; i++) {
                prompt.add(in.readInt());
            }
            int count = in.readInt();
            List<float[]> rows = new ArrayList<>(count);
            for (int r = 0; r < count; r++) {
                float[] row = new float[in.readInt()];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Float.intBitsToFloat(in.readInt());
                }
                rows.add(row);
            }
            return new Run(rows, null, seed, firstIndex, prompt);
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
