package org.beehive.jitllm.backend.tornado.plan.components.activation;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batch prefill activation for a Q8_0 or Q4_0 embedding, decoded on the device.
 *
 * <p>{@link BatchPrefillActivation} takes the batch already decoded to FP32 by the host, element by
 * element in Java. Until the JIT compiles that loop the first chunks of a prompt spend a quarter of
 * a second each in it. Here the host only copies each token's raw row into {@code
 * embeddingQ8Batch}, one bulk copy per row, and {@code convertQ8_0toFP32} decodes the whole chunk:
 * consecutive Q8_0 rows of a width divisible by 32 are themselves one Q8_0 vector.
 */
public class BatchPrefillQ8DeviceActivation implements ActivationTaskGraph {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int Q4_0_BLOCK_BYTES = 18;

    private final ImmutableTaskGraph itg;
    private final int batchSize;
    private final int dim;

    public BatchPrefillQ8DeviceActivation(State state, Configuration config, int batchSize) {
        this(state, config, batchSize, DataType.Q8_0);
    }

    /**
     * The embedding rows in {@code embedding}'s representation: Q8_0, or Q4_0, whose consecutive
     * rows are likewise one Q4_0 vector.
     */
    public BatchPrefillQ8DeviceActivation(
            State state, Configuration config, int batchSize, DataType embedding) {
        if (embedding != DataType.Q8_0 && embedding != DataType.Q4_0) {
            throw new IllegalArgumentException(
                    "the device-decoded batch activation reads Q8_0 or Q4_0 rows, not "
                            + embedding);
        }
        if (config.dim() % Q8_0_BLOCK_SIZE != 0) {
            throw new IllegalArgumentException(
                    "a Q8_0 batch decoded as one vector needs a width divisible by 32, not "
                            + config.dim());
        }
        this.batchSize = batchSize;
        this.dim = config.dim();
        boolean q8 = embedding == DataType.Q8_0;
        int rowBytes = dim / Q8_0_BLOCK_SIZE * (q8 ? Q8_0_BLOCK_BYTES : Q4_0_BLOCK_BYTES);
        if (state.workspace.embeddingQ8Batch == null
                || state.workspace.embeddingQ8Batch.getSize() < batchSize * rowBytes) {
            state.workspace.embeddingQ8Batch = TornadoWorkspaces.bytes(batchSize * rowBytes);
        }
        KernelContext context = new KernelContext();
        this.itg =
                new TaskGraph("prefillActivation")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                context,
                                state.workspace.wrapXBatch)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingQ8Batch)
                        .task(
                                "updateX",
                                q8
                                        ? TransformerComputeKernels::convertQ8_0toFP32
                                        : TransformerComputeKernels::convertQ4_0toFP32,
                                context,
                                state.workspace.embeddingQ8Batch,
                                state.workspace.wrapXBatch)
                        .persistOnDevice(state.workspace.wrapXBatch)
                        .snapshot();
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        scheduler.addWorkerGrid(
                "prefillActivation.updateX", WorkerGridFactory.genericWorker(batchSize * dim, 128));
        return scheduler;
    }
}
