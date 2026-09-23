package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.GridScheduler;

/**
 * Runs the pinned fixture and captures one logits row per generated token.
 *
 * <p>The hook is the {@link Sampler}: it receives the logits row for every generated position, so
 * capturing needs no production change. Sampling stays greedy (argmax), which makes the seed
 * irrelevant and the token sequence deterministic.
 *
 * <p>Requires {@code -Djitllm.deviceSample=false} (the default). With on-device sampling the argmax
 * runs on the GPU and only the token id crosses to the host, so there would be no logits row to
 * capture — {@link #assertHostLogitsAvailable()} makes that explicit rather than silently producing
 * empty goldens.
 */
public final class GoldenCapture {

    /** Compared rows: one per generated token. Stated verbatim in the golden metadata. */
    public static final int TOKENS = 64;

    /** The fixed prompt. Any change to this invalidates every committed golden. */
    public static final String PROMPT = "Explain what a matrix multiplication is in one paragraph.";

    public static final int CONTEXT_LENGTH = 512;

    public static final class Result {
        public final List<float[]> rows = new ArrayList<>();
        public final List<Integer> tokenIds = new ArrayList<>();

        /**
         * The grid scheduler of the plan this capture ran on: one entry per task, with the geometry
         * it is configured to launch on. {@code null} for a CPU capture, or for a plan shape {@link
         * org.beehive.jitllm.backend.tornado.PlanDispatchEvidence} cannot reach — callers that need
         * it assert on it. Readable after the execution plan is freed.
         */
        public GridScheduler gridScheduler;

        /** The model dimension of the capture, so a caller can compute an expected grid. */
        public int dim;

        /**
         * For a batched plan, the kernel method names its batched layer graphs compile for the
         * tasks of {@link #RECORDED_BATCHED_TASKS}, by task name — read off the plan while it was
         * alive, since the grid cannot tell two kernels of one geometry apart. Empty otherwise.
         */
        public final java.util.Map<String, java.util.Set<String>> batchedTaskKernels =
                new java.util.TreeMap<>();
    }

    /** The batched tasks whose compiled kernel a capture records. */
    static final String[] RECORDED_BATCHED_TASKS = {
        "attention", "ssm_delta_rule", "ssm_alpha_proj", "ssm_beta_proj"
    };

    private GoldenCapture() {}

    public static void assertHostLogitsAvailable() {
        if (Boolean.getBoolean("jitllm.deviceSample")) {
            throw new IllegalStateException(
                    "jitllm.deviceSample=true keeps the logits row on the device; goldens must run with it false");
        }
    }

    public static Result capture(Path ggufPath, boolean useGpu) throws Exception {
        return capture(ggufPath, useGpu, null);
    }

    /**
     * @param forcedTokens when non-null, the sampler returns these tokens instead of its own argmax
     *     ("teacher forcing").
     *     <p>This is what makes a cross-path comparison meaningful. Greedy decoding is
     *     autoregressive, so the first near-tie that tips differently sends the two paths into
     *     different contexts, and every row after that compares unrelated states. Forcing both
     *     paths along the same token sequence keeps the context identical at every position, so a
     *     difference in logits is a difference in arithmetic rather than a difference in history.
     */
    public static Result capture(Path ggufPath, boolean useGpu, List<Integer> forcedTokens)
            throws Exception {
        return capture(ggufPath, useGpu, forcedTokens, 1);
    }

    /**
     * The same capture, driven through batched prefill when {@code prefillBatchSize} is above one.
     *
     * <p>The execution policy is read from system properties when the {@link State} is built, and
     * the batch width has to be known then too because the prefill workspace is sized from it. So
     * the properties are set around construction and restored after: this harness is the only place
     * that knows the mode is being forced, and a leaked property would silently change every later
     * test in the same JVM.
     *
     * <p>Nothing else changes. The same prompt, the same forced token sequence and the same
     * capturing sampler, so a difference in the rows is a difference in the arithmetic of the
     * batched kernels and not a difference in what was asked of them.
     */
    public static Result capture(
            Path ggufPath, boolean useGpu, List<Integer> forcedTokens, int prefillBatchSize)
            throws Exception {
        return capture(ggufPath, useGpu, forcedTokens, prefillBatchSize, prefillBatchSize > 1);
    }

    /**
     * The same capture with the phase strategy stated rather than inferred from the batch width.
     *
     * <p>Sequential prefill is {@code PREFILL_DECODE} at a batch of one, which the batch width
     * alone cannot express: a width of one is also what {@code STANDARD} uses. A caller that wants
     * prompt ingestion as its own phase says so.
     */
    public static Result capture(
            Path ggufPath,
            boolean useGpu,
            List<Integer> forcedTokens,
            int prefillBatchSize,
            boolean separatePrefillPhase)
            throws Exception {
        assertHostLogitsAvailable();

        String previousPrefill = System.getProperty("jitllm.withPrefillDecode");
        String previousBatch = System.getProperty("jitllm.prefillBatchSize");
        if (separatePrefillPhase) {
            System.setProperty("jitllm.withPrefillDecode", "true");
            System.setProperty("jitllm.prefillBatchSize", String.valueOf(prefillBatchSize));
        }
        try {
            return captureWithCurrentPolicy(ggufPath, useGpu, forcedTokens, prefillBatchSize);
        } finally {
            restore("jitllm.withPrefillDecode", previousPrefill);
            restore("jitllm.prefillBatchSize", previousBatch);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static Result captureWithCurrentPolicy(
            Path ggufPath, boolean useGpu, List<Integer> forcedTokens, int prefillBatchSize)
            throws Exception {
        Model model = ModelLoader.loadModel(ggufPath, CONTEXT_LENGTH, true, useGpu);
        State state =
                prefillBatchSize > 1
                        ? State.withPrefillBatchSize(prefillBatchSize, () -> model.createNewState())
                        : model.createNewState();
        ChatFormat chatFormat = model.chatFormat();

        List<Integer> promptTokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        promptTokens.addAll(
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, PROMPT)));
        promptTokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Result result = new Result();
        result.dim = model.configuration().dim();
        Sampler capturing =
                tensor -> {
                    result.rows.add(toFloatArray(tensor));
                    int token = Sampler.TENSOR_ARGMAX.sampleToken(tensor);
                    result.tokenIds.add(token);
                    int step = result.rows.size() - 1;
                    if (forcedTokens != null && step < forcedTokens.size()) {
                        return forcedTokens.get(step);
                    }
                    return token;
                };

        // No stop tokens: a golden must always compare the same number of rows, so generation is
        // bounded only by TOKENS.
        Set<Integer> stopTokens = Set.of();

        // The budget counts every forward, ingestion included, and ingestion is one shorter for a
        // family whose seed the prompt already carries — see PromptIngestion. Deriving the
        // adjustment from the same source keeps the row count at TOKENS for every family, instead
        // of encoding one family's arithmetic as a constant that quietly rots.
        int skippedSeed =
                org.beehive.jitllm.inference.PromptIngestion.of(state, promptTokens, 0)
                        .firstIndex();
        int budget = promptTokens.size() + TOKENS - skippedSeed;

        TornadoVMMasterPlan plan = null;
        try {
            if (useGpu) {
                // The factory consults the lowering's opt-in itself, so this harness needs no
                // branch of its own — it needed one while the branch lived at each construction
                // site, and that duplication is what let the CLI and the benchmark script miss it.
                // Callers still assert on LoweredPlanSelection.loweredPlanCount(), never on the
                // property: the question is whether the lowering ran, not whether it was asked for.
                plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
                // The dispatch this capture's own plan was built with, for callers that assert on
                // it. Optional here: a plan shape this seam cannot reach records nothing.
                result.gridScheduler =
                        org.beehive.jitllm.backend.tornado.PlanDispatchEvidence
                                .gridSchedulerIfAvailable(plan);
                if (plan
                        instanceof
                        org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode) {
                    // Only the tasks this family's batched graphs hold: another family records
                    // nothing under a qwen35 task name rather than failing the capture.
                    for (String task : RECORDED_BATCHED_TASKS) {
                        java.util.Set<String> kernels =
                                org.beehive.jitllm.backend.tornado.PlanDispatchEvidence
                                        .batchedTaskKernelsIfAny(plan, task);
                        if (!kernels.isEmpty()) {
                            result.batchedTaskKernels.put(task, kernels);
                        }
                    }
                }
                model.generateTokensGPU(
                        state, 0, promptTokens, stopTokens, budget, capturing, false, null, plan);
            } else {
                model.generateTokens(
                        state, 0, promptTokens, stopTokens, budget, capturing, false, null);
            }
        } finally {
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }
        return result;
    }

    private static float[] toFloatArray(org.beehive.jitllm.inference.Logits logits) {
        float[] out = new float[logits.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits.get(i);
        }
        return out;
    }
}
