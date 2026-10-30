package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Activation extends AbstractLayer implements ActivationTaskGraph {

    /**
     * Whether {@code embedding} is a quantized representation other than Q8_0, which a load keeps
     * only as the file stores it: its rows are staged in that representation and decoded by {@link
     * #embeddingRowDecode}.
     */
    public static boolean keptEmbedding(DataType embedding) {
        return embedding == DataType.Q4_0
                || embedding == DataType.Q4_1
                || embedding == DataType.Q4_K
                || embedding == DataType.Q5_K
                || embedding == DataType.Q6_K;
    }

    /** The device decode of one staged embedding row of a {@link #keptEmbedding} type into FP32. */
    public static TornadoFunctions.Task3<KernelContext, ByteArray, FloatArray> embeddingRowDecode(
            DataType embedding) {
        return switch (embedding) {
            case Q4_0 -> TransformerComputeKernels::convertQ4_0toFP32;
            case Q4_1 -> TransformerComputeKernels::convertQ4_1toFP32;
            case Q4_K -> TransformerComputeKernels::convertQ4_KtoFP32;
            case Q5_K -> TransformerComputeKernels::convertQ5_KtoFP32;
            case Q6_K -> TransformerComputeKernels::convertQ6_KtoFP32;
            default ->
                    throw new IllegalArgumentException(
                            "no device row decode for a " + embedding + " embedding");
        };
    }

    private final TaskGraph activationTaskGraph;

    public Activation(String name, State state, Weights weights, Configuration config) {
        super(state, weights, config);
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
        DataType embedding =
                weights instanceof TornadoWeights t ? t.getTokenEmbeddingTable().dataType() : null;
        if (keptEmbedding(embedding)) {
            return new TaskGraph(name)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                    .task(
                            "updateX",
                            embeddingRowDecode(embedding),
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
