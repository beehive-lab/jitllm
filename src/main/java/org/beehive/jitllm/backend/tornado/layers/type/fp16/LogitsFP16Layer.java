package org.beehive.jitllm.backend.tornado.layers.type.fp16;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.scheduling.Fp16GemvReductionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsFP16Layer extends AbstractLogitsTaskGraph {

    /** Local workgroup size for the single-workgroup on-device argmax. */
    private static final int SAMPLE_LOCAL = 256;

    /**
     * On-device greedy sampling: append a GPU argmax over the logits and transfer only the sampled
     * token id (1 int) to the host instead of the full vocab logits row. Only valid for greedy
     * decoding — {@code JitllmApp} refuses it for temperature &gt; 0 and non-FP16 models, which
     * still need the full logits host-side.
     *
     * <p><b>Resolved from the session's policy on every read, deliberately not cached in a
     * field</b> (then Metal parity task 8). It was originally a {@code public static final boolean}
     * read from {@code jitllm.deviceSample} at class initialization — replaced with a per-session
     * read, then reintroduced as an instance field, which fell into exactly the
     * constructor-ordering pitfall {@link #useSimd32Reduction} was already written to document and
     * avoid: {@link AbstractLogitsTaskGraph}'s constructor invokes {@link #setupLogitsTaskGraph}
     * through {@code super(.)}, which runs before this subclass's own field initializers do, so a
     * {@code private final boolean} field here still held its default {@code false} at that point.
     * The task graph was therefore built as if device-resident sampling had never been requested —
     * no {@code argmax_sample} task, no write to {@code state.workspace.sampledToken} — while
     * {@link #updateGridScheduler}, called after construction completes, saw the correctly-resolved
     * value and registered a worker grid for a task that did not exist. The result: silent,
     * deterministic wrong output (a constant token, since {@code sampledToken} was never written
     * and stayed at its zero-initialized default) whenever {@code SamplingResidency.DEVICE} was
     * requested — found by {@code
     * DeviceResidentSamplingAccelTest.deviceResidentArgmaxMatchesHostResidentArgmax}, not specific
     * to Metal (the ordering bug is pure Java, present on every backend; Metal is simply where it
     * was first exercised end to end). Calling the (cheap) query directly at each use site, exactly
     * as {@link #useSimd32Reduction} already does, avoids the ordering hazard entirely.
     */
    private boolean deviceSample() {
        return state.executionPolicy().samplingResidency()
                == org.beehive.jitllm.runtime.policy.ExecutionPolicy.SamplingResidency.DEVICE;
    }

    /**
     * Vocabulary-projection reduction strategy: 32-lane subgroup shuffle where verified correct
     * (Metal), shared-memory reduction elsewhere — the same selection {@code
     * LlamaFP16FFNLayers.useSimd32Reduction} already uses for the per-layer QKV/residual/ FFN
     * kernels, on the same {@code DeviceCapability.SUBGROUP_SHUFFLE_32} (Metal parity task 6
     * follow-up: {@code matrixVectorGenericSimd32} is the vocabulary-shaped sibling, isolated and
     * verified separately — see the accompanying review document).
     *
     * <p><b>Not cached in a field, deliberately.</b> {@link AbstractLogitsTaskGraph}'s constructor
     * calls {@link #setupLogitsTaskGraph} — the overridden hook that reads this — through {@code
     * super(.)}, which runs before this subclass's own field initializers do. A {@code private
     * final boolean} field here would still hold its default {@code false} at that point, silently
     * selecting the generic kernel even on Metal (caught by {@code
     * MatrixVectorGenericSimd32AccelTest}'s sibling gate failing to reproduce the selection a full
     * model run showed). Calling the (cheap, cached-inside) query directly at each use site avoids
     * the ordering hazard entirely.
     */
    // @formatter:off
    /**
     * Set by {@link #setupLogitsTaskGraph} when it installs the shuffle-reducing vocabulary kernel,
     * and read by {@link #updateGridScheduler} so that the worker grid follows the kernel that was
     * actually installed instead of asking the same question a second time.
     *
     * <p><b>Why this is not just another {@link #useSimd32Reduction()} call.</b> Two subclasses
     * override {@link #setupLogitsTaskGraph} and build their own {@code vocab_proj} — Granite
     * scales by {@code logitScale}, Gemma 4 soft-caps — and neither overrides {@link
     * #updateGridScheduler}. Deriving the grid from the capability therefore paired a 32-lane grid
     * with their shared-memory kernels, which produced 99% wrong logits at up to 2.4e+08×
     * tolerance, silently, with no error anywhere. Recording what was installed makes the
     * shared-memory grid the default for any subclass that supplies its own kernel, which is both
     * the safe answer and the correct one.
     *
     * <p>Deliberately has no field initializer. {@link AbstractLogitsTaskGraph}'s constructor calls
     * {@link #setupLogitsTaskGraph} through {@code super(.)}; an initializer here would run
     * afterwards and overwrite what that call recorded. Allocation-time {@code false} is the
     * starting value, and the only write is the one inside the branch that installs the kernel.
     */
    // @formatter:on
    private boolean vocabularyProjectionReducesWithShuffle;

    private static boolean useSimd32Reduction() {
        return SchedulerDetectionService.isSubgroupShuffle32Supported()
                // The decode-shaped preference. This class is shared, so the preference reaches
                // every FP16 family whose logits layer is this one; Fp16GemvReductionPolicy lists
                // them and says what the evidence covers.
                || Fp16GemvReductionPolicy.preferShuffleReduction();
    }

    public LogitsFP16Layer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    /**
     * Hook called before any data transfers or tasks. Override to prepend {@code consumeFromDevice}
     * declarations that must precede the bytecode (e.g. KV-cache pass-through in the Phase 4
     * unified plan).
     */
    protected void configureAdditionalConsumes(TaskGraph logits) {}

    /**
     * Hook called after {@code transferToHost}. Override to append {@code persistOnDevice}
     * declarations (e.g. KV-cache pass-through in Phase 4).
     */
    protected void configureAdditionalPersists(TaskGraph logits) {}

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");
        // === Data Setup ===
        configureAdditionalConsumes(logits);
        logits.consumeFromDevice(lastTaskGraphID, state.workspace.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.tempLogits);
        logits.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                // Kernel context
                context,
                // Output buffer
                state.workspace.wrapLogits,
                // Intermediate FP16 buffer
                state.workspace.wrapXbFP16,
                // Weights
                weights.wclsByteArray.asHalfFloatArray(),
                weights.rms_final_weight_as_floatArray.asFloatArray());

        // === Final RMS Normalization ===
        logits.task(
                "rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempLogits, // output: partial sums + final scale factor
                state.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon for numerical stability
                state.localSize); // local workgroup size

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task(
                    "rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempLogits, // in/out: combines partial sums
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        logits.task(
                "rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantizeLogits,
                context,
                state.workspace.wrapXbFP16, // output: normalized (FP16)
                state.workspace.wrapX, // input: hidden state
                weights.rms_final_weight_as_floatArray.asFloatArray(), // RMS weights
                state.workspace.tempLogits); // scale factor from reduction

        // === Vocabulary Projection ===
        // Same task name, same output contract either way; only the reduction kernel (and its
        // matching worker grid, in updateGridScheduler) differs, by device capability.
        if (useSimd32Reduction()) {
            vocabularyProjectionReducesWithShuffle = true;
            logits.task(
                    "vocab_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericSimd32,
                    context,
                    state.workspace.wrapXbFP16,
                    state.workspace.wrapLogits,
                    weights.wclsByteArray.asHalfFloatArray(),
                    config.dim(),
                    config.vocabularySize());
        } else {
            logits.task(
                    "vocab_proj",
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    state.workspace.wrapXbFP16, // input (FP16)
                    state.workspace.wrapLogits, // output
                    weights.wclsByteArray.asHalfFloatArray(), // vocabulary weights
                    config.dim(), // input dimension
                    config.vocabularySize(), // output dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);
        }

        // === Sampling / result transfer ===
        if (deviceSample()) {
            // Greedy argmax on the GPU; only the token id crosses to the host (the full
            // vocab logits row stays device-side — no big D2H copy, no host scan).
            logits.transferToDevice(DataTransferMode.FIRST_EXECUTION, state.workspace.sampledToken);
            logits.task(
                    "argmax_sample",
                    TransformerComputeKernels::argmaxLogits,
                    context,
                    state.workspace.wrapLogits,
                    state.workspace.sampledToken,
                    config.vocabularySize(),
                    SAMPLE_LOCAL);
            logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.sampledToken);
        } else {
            logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        }
        configureAdditionalPersists(logits);
        return logits;
    }

    // @formatter:on

    // @formatter:off
    /**
     * The worker grid that pairs with a vocabulary projection kernel.
     *
     * <p>{@code matrixVectorGenericSimd32} assumes exactly one 32-lane workgroup per output row,
     * the same assumption every other {@code Simd32} kernel makes; the shared-memory kernel scales
     * the local size by {@code THREAD_SCALE_FOR_LOGITS} instead. The two share a task name, so the
     * grid is the only thing that distinguishes them, and pairing it with the wrong kernel is
     * silent: it produced 99.43% wrong logits on Granite F16 with nothing thrown anywhere.
     *
     * <p>Pure and package-visible so the pairing can be asserted without a model fixture; see
     * {@code LogitsVocabularyGridContractTest}.
     *
     * @param shuffleReduced whether the kernel installed is the shuffle-reducing one. {@code false}
     *     is the answer for any subclass that supplies its own {@code vocab_proj} -- Granite scales
     *     by {@code logitScale}, Gemma 4 soft-caps -- because those kernels reduce through shared
     *     memory.
     */
    // @formatter:on
    static WorkerGrid1D vocabularyWorker(int vocabularySize, boolean shuffleReduced) {
        int localSize = shuffleReduced ? 32 : LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid1D worker = new WorkerGrid1D(vocabularySize * localSize);
        worker.setLocalWork(localSize, 1, 1);
        return worker;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), rmsLocalSize());
        var vocabWorker =
                vocabularyWorker(config.vocabularySize(), vocabularyProjectionReducesWithShuffle);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", rmsReduceWorker(logitsRMS));
        tornadoForwardScheduler.addWorkerGrid("logits.rms_apply_fp16", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        if (deviceSample()) {
            var argmaxWorker = new WorkerGrid1D(SAMPLE_LOCAL); // one workgroup
            argmaxWorker.setLocalWork(SAMPLE_LOCAL, 1, 1);
            tornadoForwardScheduler.addWorkerGrid("logits.argmax_sample", argmaxWorker);
        }
        return tornadoForwardScheduler;
    }

    /** Local workgroup size for RMS norm. Qwen2 requires a smaller group (32 vs 256). */
    protected int rmsLocalSize() {
        return weights instanceof Qwen2TornadoWeights ? 32 : 256;
    }
}
