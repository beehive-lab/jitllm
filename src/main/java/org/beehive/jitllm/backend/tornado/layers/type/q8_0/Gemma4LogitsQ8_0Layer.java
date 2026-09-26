package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.Gemma4Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Gemma4-specific Q8_0 logits layer.
 *
 * <p>Identical to {@link LogitsQ8_0Layer} except for one addition: Gemma4 applies a final logit
 * soft-cap, {@code logits = softcap * tanh(logits / softcap)}, after the vocabulary projection (see
 * {@code gemma4.final_logit_softcapping} and {@link
 * org.beehive.jitllm.backend.cpu.InferenceCore#forwardJavaGemma4}).
 */
public class Gemma4LogitsQ8_0Layer extends LogitsQ8_0Layer {

    private static final String SOFTCAP_TASK = "logit_softcap";

    public Gemma4LogitsQ8_0Layer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    private float softcap() {
        return ((Gemma4Configuration) config).finalLogitSoftcapping();
    }

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
                state.workspace.tempLogits,
                state.workspace.wrapX,
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
        // By the output tensor's own representation, not by this class's name. A Q4_0 Gemma 4 file
        // ties the output projection to a Q4_K token_embd, and reading that with Q8_0 block
        // arithmetic runs off the end of the buffer.
        addVocabularyProjection(logits, weights, config);

        // === Final logit soft-capping (Gemma4-specific) ===
        if (softcap() != 0.0f) {
            logits.task(
                    SOFTCAP_TASK,
                    Gemma4Kernels::applyLogitSoftcap,
                    context,
                    state.workspace.wrapLogits,
                    softcap(),
                    config.vocabularySize());
        }

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        configureAdditionalPersists(logits);
        return logits;
    }

    // @formatter:on

    // @formatter:off
    /**
     * A Q8_0 vocabulary projection takes the warp-per-row kernel; everything else the inherited
     * dispatch.
     */
    // @formatter:on
    @Override
    protected void addVocabularyProjection(
            TaskGraph logits, TornadoWeights weights, Configuration config) {
        if (!warpVocabulary(weights)) {
            super.addVocabularyProjection(logits, weights, config);
            return;
        }
        logits.task(
                "vocab_proj",
                Gemma4Kernels::matrixVectorQ8_0Warp,
                context,
                state.workspace.wrapX,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                config.dim(),
                config.vocabularySize());
    }

    // @formatter:off
    /**
     * A Q8_0 output projection of a model whose own projections are Q8_0. The Q4_0 file's Q4_K
     * {@code token_embd} is materialized as Q8_0 on the device as well, but this kernel was
     * evaluated on the Q8_0 file only; that one keeps the shared-memory kernel.
     */
    // @formatter:on
    private boolean warpVocabulary(TornadoWeights weights) {
        return weights.dataType() == org.beehive.jitllm.runtime.tensor.DataType.Q8_0
                && weights.wclsByteArray.dataType()
                        == org.beehive.jitllm.runtime.tensor.DataType.Q8_0
                && config.dim() % 32 == 0
                && schedulerType != SchedulerType.NON_NVIDIA
                // An NVIDIA device through OpenCL is on the NVIDIA path as well, and that backend
                // cannot compile the kernel's shuffle; see Gemma4Q8_0FFNLayers.warpProjection.
                && SchedulerDetectionService.isShuffleReducedFp16GemvSupported();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var scheduler = super.updateGridScheduler(tornadoForwardScheduler);
        if (weights instanceof TornadoWeights tw && warpVocabulary(tw)) {
            int groups =
                    (config.vocabularySize() + Gemma4Kernels.WARP_ROWS_PER_GROUP - 1)
                            / Gemma4Kernels.WARP_ROWS_PER_GROUP;
            var worker = new uk.ac.manchester.tornado.api.WorkerGrid1D(groups * 256);
            worker.setLocalWork(256, 1, 1);
            scheduler.addWorkerGrid("logits.vocab_proj", worker);
        }
        if (softcap() != 0.0f) {
            scheduler.addWorkerGrid(
                    "logits." + SOFTCAP_TASK,
                    WorkerGridFactory.genericWorker(
                            config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC));
        }
        return scheduler;
    }
}
