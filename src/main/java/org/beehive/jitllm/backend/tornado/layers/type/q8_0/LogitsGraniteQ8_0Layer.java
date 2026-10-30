package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.GraniteBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.scheduling.LogitsWorkgroupPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Granite-specific Q8_0 logits layer. Identical to LogitsQ8_0Layer except vocab_proj uses a scaled
 * kernel (logitScale).
 */
public class LogitsGraniteQ8_0Layer extends LogitsQ8_0Layer {

    public LogitsGraniteQ8_0Layer(
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        GraniteConfiguration graniteCfg = (GraniteConfiguration) config;
        var logits = new TaskGraph(GraphNames.LOGITS);

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
        if (weights.wclsByteArray.dataType() == DataType.Q8_0) {
            logits.task(
                    "vocab_proj",
                    GraniteKernels::matrixVectorGenericQ8ByteWithGraniteScale,
                    context,
                    state.workspace.wrapX,
                    state.workspace.wrapLogits,
                    weights.wclsByteArray.asByteArray(),
                    config.dim(),
                    config.vocabularySize(),
                    // Must match the workgroup LogitsQ8_0Layer.updateGridScheduler launches.
                    LogitsWorkgroupPolicy.q8Threads(),
                    graniteCfg.logitScale());
        } else {
            // A projection kept in its own representation: the logit multiplier on its input,
            // which is the same product, then the projection by its type.
            logits.task(
                    "logits_scale",
                    GraniteBatchPrefillKernels::batchedScale,
                    context,
                    state.workspace.wrapX,
                    graniteCfg.logitScale(),
                    config.dim());
            addVocabularyProjection(logits, weights, config);
        }

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        return logits;
    }

    // @formatter:on

    /** The base grids, and the multiplier's where the projection is not Q8_0. */
    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        super.updateGridScheduler(tornadoForwardScheduler);
        if (weights instanceof TornadoWeights t && t.wclsByteArray.dataType() != DataType.Q8_0) {
            tornadoForwardScheduler.addWorkerGrid(
                    "logits.logits_scale", WorkerGridFactory.genericWorker(config.dim(), 128));
        }
        return tornadoForwardScheduler;
    }
}
