package org.beehive.jllm.backend.tornado.layers.type.fp16;

import org.beehive.jllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Granite-specific FP16 logits layer. Identical to LogitsFP16Layer except vocab_proj uses a scaled
 * kernel (logitScale).
 */
public class LogitsGraniteFP16Layer extends LogitsFP16Layer {

    public LogitsGraniteFP16Layer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        GraniteConfiguration graniteCfg = (GraniteConfiguration) config;
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

        // === Vocabulary Projection (Granite: scaled by logitScale) ===
        logits.task(
                "vocab_proj",
                GraniteKernels::matrixVectorGenericWithGraniteScale,
                context,
                state.workspace.wrapXbFP16,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asHalfFloatArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS,
                graniteCfg.logitScale());

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        return logits;
    }
    // @formatter:on
}
