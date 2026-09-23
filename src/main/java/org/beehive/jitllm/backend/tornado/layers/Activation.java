package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Activation extends AbstractLayer implements ActivationTaskGraph {
    private final TaskGraph activationTaskGraph;

    public Activation(String name, State state, Weights weights, Configuration config) {
        super(name, state, weights, config);
        this.activationTaskGraph = setupActivationTaskGraph(name);
    }

    // @formatter:off
    /**
     * The embedding row, converted to the FP32 activation the first layer reads.
     *
     * <p>Dispatches on the <b>embedding tensor's own</b> representation, not on the model-wide
     * quantization string. They agree for a uniform F16 or Q8_0 file and disagree for every mixed
     * one: Qwen3.5 reports {@code Q8_0} model-wide while holding its token embeddings as retained
     * Q4_0, and staging 18-byte blocks to be read as 34-byte ones produces a plausible activation
     * and wrong output. {@code TornadoForwardPass} already stages by the tensor's own type for the
     * same reason; this is the other half of that.
     */
    protected TaskGraph setupActivationTaskGraph(String name) {
        org.beehive.jitllm.runtime.tensor.DataType embedding =
                weights instanceof org.beehive.jitllm.inference.weights.tornado.TornadoWeights t
                        ? t.getTokenEmbeddingTable().dataType()
                        : null;
        if (embedding == org.beehive.jitllm.runtime.tensor.DataType.Q4_0) {
            return new TaskGraph(name)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                    .task(
                            "updateX",
                            TransformerComputeKernels::convertQ4_0toFP32,
                            context,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX)
                    .persistOnDevice(state.workspace.wrapX);
        }
        return switch (config.quantization()) {
            case "FP16" ->
                    new TaskGraph(name)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                            .task(
                                    "updateX",
                                    TransformerComputeKernels::convertFP16toFP32,
                                    context,
                                    (HalfFloatArray) state.workspace.embeddingX,
                                    state.workspace.wrapX)
                            .persistOnDevice(state.workspace.wrapX);
            case "Q8_0" ->
                    new TaskGraph(name)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                            .task(
                                    "updateX",
                                    TransformerComputeKernels::convertQ8_0toFP32,
                                    context,
                                    (ByteArray) state.workspace.embeddingX,
                                    state.workspace.wrapX)
                            .persistOnDevice(state.workspace.wrapX);
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        };
    }

    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid worker = WorkerGridFactory.genericWorker(config.dim(), 128);
        scheduler.addWorkerGrid(activationTaskGraph.getTaskGraphName() + ".updateX", worker);
        return scheduler;
    }

    public TaskGraph getTaskGraph() {
        return activationTaskGraph;
    }

    public ImmutableTaskGraph getImmutableTaskGraph() {
        return activationTaskGraph.snapshot();
    }
}
