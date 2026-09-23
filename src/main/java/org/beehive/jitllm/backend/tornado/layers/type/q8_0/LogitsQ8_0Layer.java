package org.beehive.jllm.backend.tornado.layers.type.q8_0;

import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsQ8_0Layer extends AbstractLogitsTaskGraph {

    // @formatter:off
    public LogitsQ8_0Layer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:on

    protected void configureAdditionalConsumes(TaskGraph logits) {}

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
                context,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                weights.rms_final_weight_as_floatArray);
        // === Final RMS Normalization ===
        logits.task(
                "rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempLogits, // output: partial sums + final scale factor
                state.workspace.wrapX, // input: hidden state
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task(
                    "rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempLogits,
                    config.dim(),
                    config.rmsNormEps());
        }

        logits.task(
                "mapContextLogits",
                TransformerComputeKernels::reductionOneBlock2WithLogits,
                context,
                state.workspace.wrapX,
                weights.rms_final_weight_as_floatArray.asFloatArray(),
                state.workspace.tempLogits);

        // === Vocabulary Projection ===
        // By the projection's own representation, not the plan's. A mixed file keeps its output
        // projection in whatever the quantizer chose -- Qwen3.5's is Q6_K where its layers are
        // Q4_0 -- and reading one block layout as another produces plausible logits and wrong
        // tokens. A representation with no kernel is refused here rather than converted.
        addVocabularyProjection(logits, weights, config);

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        configureAdditionalPersists(logits);
        return logits;
    }

    // @formatter:off
    /**
     * Whether the vocabulary projection reads a quantized activation and a packed integer dot
     * product.
     *
     * <p>Three facts, none of them a preference: the output projection is {@code Q6_K}, the device
     * lowers {@code dp4a}, and this session's state carries the quantization scratch -- which is a
     * family's choice, so a family without it keeps the kernel it had, as does any other
     * representation and any other device.
     *
     * <p>It also honours the escape hatch the exact-comparison tests use to pin themselves to the
     * floating-point path. Leaving it out is what made six of them fail: their subject is
     * addressing, they compare the device against the host exactly, and this projection is as
     * unable to be exact as the layer ones are.
     *
     * <p>This is the only call site. The activation it quantizes is the final normalized one,
     * quantized in this graph immediately before the projection reads it and read by nothing else,
     * which is why it needs no provenance flag: there is no second consumer to confuse it with.
     */
    // @formatter:on
    protected boolean packedVocabulary(TornadoWeights weights) {
        return !"false".equalsIgnoreCase(System.getProperty("jllm.qwen35.packedIntegerDot", "true"))
                && weights.wclsByteArray.dataType() == org.beehive.jllm.runtime.tensor.DataType.Q6_K
                && state.workspace.wrapXbQuants != null
                && org.beehive.jllm.backend.tornado.device.TornadoDevices.current()
                        .capabilities()
                        .supports(
                                org.beehive.jllm.runtime.backend.DeviceCapability
                                        .PACKED_INTEGER_DOT);
    }

    /** The vocabulary projection task, chosen by what the output projection actually holds. */
    /**
     * The vocabulary projection, by the output tensor's own representation.
     *
     * <p>Visible to subclasses because a family that overrides {@link #setupLogitsTaskGraph} still
     * has to dispatch this the same way. Hardcoding a kernel here reads one block layout as another
     * -- 34-byte Q8_0 blocks over a 144-byte Q4_K super-block tensor walks off the end of the
     * buffer, which surfaces as an illegal address rather than as wrong numbers.
     */
    protected void addVocabularyProjection(
            TaskGraph logits, TornadoWeights weights, Configuration config) {
        int localSize = LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var w = weights.wclsByteArray;
        if (packedVocabulary(weights)) {
            // Emitted here rather than by the caller, because the projection below is the only
            // reader of this triple and a graph that has one without the other reads a buffer
            // nothing wrote. A family that overrides setupLogitsTaskGraph — Gemma 4 does, for its
            // logit soft-cap — would otherwise have to remember to bring the quantize with it, and
            // one that forgot would get plausible logits off uninitialized quants.
            logits.task(
                    "vocab_quantize",
                    org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0
                            ::quantizeActivationQ8Blocks,
                    context,
                    state.workspace.wrapX,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapXbSums);
            logits.task(
                    "vocab_proj",
                    org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K
                            ::matrixVectorGenericQ6_KDP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapXbSums,
                    state.workspace.wrapLogits,
                    w.asByteArray(),
                    config.dim(),
                    config.vocabularySize(),
                    localSize);
            return;
        }
        switch (w.dataType()) {
            case Q8_0 ->
                    logits.task(
                            "vocab_proj",
                            TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            case Q4_0 ->
                    logits.task(
                            "vocab_proj",
                            org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0
                                    ::matrixVectorGenericQ4_0,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            case Q4_1 ->
                    logits.task(
                            "vocab_proj",
                            org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1
                                    ::matrixVectorGenericQ4_1,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            case Q4_K ->
                    logits.task(
                            "vocab_proj",
                            org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K
                                    ::matrixVectorGenericQ4_K,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            case Q5_K ->
                    logits.task(
                            "vocab_proj",
                            org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K
                                    ::matrixVectorGenericQ5_K,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            case Q6_K ->
                    logits.task(
                            "vocab_proj",
                            org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K
                                    ::matrixVectorGenericQ6_K,
                            context,
                            state.workspace.wrapX,
                            state.workspace.wrapLogits,
                            w.asByteArray(),
                            config.dim(),
                            config.vocabularySize(),
                            localSize);
            default ->
                    throw new UnsupportedOperationException(
                            "the vocabulary projection is "
                                    + w.dataType()
                                    + ", for which there is no device kernel. It is not converted"
                                    + " to Q8_0 to get one: that would double what it occupies and"
                                    + " hide a missing kernel behind a memory cost.");
        }
    }

    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), rmsLocalSize());
        var vocabSizeRowMajor =
                config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", rmsReduceWorker(logitsRMS));
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", logitsRMS);
        if (weights instanceof TornadoWeights tornadoWeights && packedVocabulary(tornadoWeights)) {
            tornadoForwardScheduler.addWorkerGrid(
                    "logits.vocab_quantize",
                    org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory.genericWorker(
                            config.dim(), 32));
        }
        return tornadoForwardScheduler;
    }

    /** Local workgroup size for RMS norm. Qwen2 requires a smaller group (32 vs 256). */
    protected int rmsLocalSize() {
        return weights instanceof Qwen2TornadoWeights ? 32 : 256;
    }
}
