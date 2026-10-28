package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.GraniteBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Granite-specific activation: applies an embedding scale factor during the FP32 conversion.
 * Overrides only the TaskGraph builder; all other behaviour is inherited from Activation.
 */
public class ActivationGranite extends Activation {

    // Granite is a special case where activation X is scaled by embedding scale float value that
    // inside model.
    public ActivationGranite(
            String taskGraphHandle, State state, Weights weights, GraniteConfiguration config) {
        super(taskGraphHandle, state, weights, config);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupActivationTaskGraph(String handle) {
        GraniteConfiguration cfg = (GraniteConfiguration) config;
        DataType embedding =
                weights instanceof TornadoWeights t ? t.getTokenEmbeddingTable().dataType() : null;
        if (keptEmbedding(embedding)) {
            // The row decoded by its own representation, then the embedding multiplier.
            return new TaskGraph(handle)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                    .task(
                            "updateX",
                            embeddingRowDecode(embedding),
                            context,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX)
                    .task(
                            "scaleX",
                            GraniteBatchPrefillKernels::batchedScale,
                            context,
                            state.workspace.wrapX,
                            cfg.embeddingScale(),
                            config.dim())
                    .persistOnDevice(state.workspace.wrapX);
        }
        return switch (config.quantization()) {
            case "FP16" ->
                    new TaskGraph(handle)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                            .task(
                                    "updateX",
                                    GraniteKernels::convertFP16toFP32withGraniteScale,
                                    context,
                                    (HalfFloatArray) state.workspace.embeddingX,
                                    state.workspace.wrapX,
                                    cfg.embeddingScale())
                            .persistOnDevice(state.workspace.wrapX);
            case "Q8_0" ->
                    new TaskGraph(handle)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                            .task(
                                    "updateX",
                                    GraniteKernels::convertQ8_0toFP32withGraniteScale,
                                    context,
                                    (ByteArray) state.workspace.embeddingX,
                                    state.workspace.wrapX,
                                    cfg.embeddingScale())
                            .persistOnDevice(state.workspace.wrapX);
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        };
    }

    // @formatter:on

    /** The decode's grid, and the multiplier's where the row is decoded by its own type. */
    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        super.updateGridScheduler(scheduler);
        if (weights instanceof TornadoWeights t
                && keptEmbedding(t.getTokenEmbeddingTable().dataType())) {
            scheduler.addWorkerGrid(
                    getTaskGraph().getTaskGraphName() + ".scaleX",
                    WorkerGridFactory.genericWorker(config.dim(), 128));
        }
        return scheduler;
    }
}
