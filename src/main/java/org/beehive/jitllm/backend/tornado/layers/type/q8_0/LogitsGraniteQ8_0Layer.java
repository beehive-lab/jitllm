package org.beehive.jllm.backend.tornado.layers.type.q8_0;

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
 * Granite-specific Q8_0 logits layer. Identical to LogitsQ8_0Layer except vocab_proj uses a scaled
 * kernel (logitScale).
 */
public class LogitsGraniteQ8_0Layer extends LogitsQ8_0Layer {

    public LogitsGraniteQ8_0Layer(
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

        // === Vocabulary Projection (Granite: scaled by logitScale) ===
        logits.task(
                "vocab_proj",
                GraniteKernels::matrixVectorGenericQ8ByteWithGraniteScale,
                context,
                state.workspace.wrapX,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS,
                graniteCfg.logitScale());

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        return logits;
    }
    // @formatter:on
}
