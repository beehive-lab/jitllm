package org.beehive.jllm.backend.tornado.layers.type.fp16;

import org.beehive.jllm.backend.tornado.kernels.Gemma4Kernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.gemma4.Gemma4Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Gemma4-specific FP16 logits layer.
 *
 * <p>Identical to {@link LogitsFP16Layer} except for one addition: Gemma4 applies a final logit
 * soft-cap, {@code logits = softcap * tanh(logits / softcap)}, after the vocabulary projection (see
 * {@code gemma4.final_logit_softcapping} and {@link
 * org.beehive.jllm.backend.cpu.InferenceCore#forwardJavaGemma4}).
 */
public class Gemma4LogitsFP16Layer extends LogitsFP16Layer {

    private static final String SOFTCAP_TASK = "logit_softcap";

    public Gemma4LogitsFP16Layer(
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
        logits.consumeFromDevice(lastTaskGraphID, state.workspace.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.tempLogits);
        logits.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                state.workspace.wrapLogits,
                state.workspace.wrapXbFP16,
                weights.wclsByteArray.asHalfFloatArray(),
                weights.rms_final_weight_as_floatArray.asFloatArray());

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
                "rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantizeLogits,
                context,
                state.workspace.wrapXbFP16,
                state.workspace.wrapX,
                weights.rms_final_weight_as_floatArray.asFloatArray(),
                state.workspace.tempLogits);

        // === Vocabulary Projection ===
        logits.task(
                "vocab_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                state.workspace.wrapXbFP16,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asHalfFloatArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

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

        // === Transfer Results to Host ===
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        return logits;
    }

    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var scheduler = super.updateGridScheduler(tornadoForwardScheduler);
        if (softcap() != 0.0f) {
            scheduler.addWorkerGrid(
                    "logits." + SOFTCAP_TASK,
                    WorkerGridFactory.genericWorker(
                            config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC));
        }
        return scheduler;
    }
}
