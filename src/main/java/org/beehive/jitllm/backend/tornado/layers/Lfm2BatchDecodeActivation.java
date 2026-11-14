package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The decode activation of the batched {@code lfm2moe} plan: one token's embedding, and the
 * hand-over of what the prompt chunks left on the device.
 *
 * <p>It relays the key/value store, the block table and the convolution windows from the last batch
 * layer to decode layer 0, as {@code Qwen35BatchDecodeActivation} relays its recurrent state: the
 * windows are the prompt's history, and a second copy would start decode from zeros.
 */
// @formatter:on
public class Lfm2BatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    public Lfm2BatchDecodeActivation(
            Lfm2State state,
            Lfm2TornadoWeights weights,
            Lfm2Configuration config,
            String lastBatchLayerId) {
        this.dim = config.dim();
        this.itg = build(new KernelContext(), state, weights, lastBatchLayerId).snapshot();
    }

    private TaskGraph build(
            KernelContext context,
            Lfm2State state,
            Lfm2TornadoWeights weights,
            String lastBatchLayerId) {
        var ws = state.workspace;
        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        TornadoWorkspaces.stagingHolds(ws, embedding, dim);
        TaskGraph graph =
                new TaskGraph(GraphNames.DECODE_ACTIVATION)
                        .consumeFromDevice(lastBatchLayerId, keyStore(state), valueStore(state))
                        .consumeFromDevice(lastBatchLayerId, ws.wrapBlockTable, ws.wrapConvState)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.embeddingX);
        if (Activation.stagedAsBytes(embedding, ws.embeddingX)) {
            graph.task(
                    "updateX",
                    Activation.embeddingRowDecode(embedding),
                    context,
                    (ByteArray) ws.embeddingX,
                    ws.wrapX);
        } else if (embedding == DataType.Q8_0) {
            graph.task(
                    "updateX",
                    TransformerComputeKernels::convertQ8_0toFP32,
                    context,
                    (ByteArray) ws.embeddingX,
                    ws.wrapX);
        } else {
            graph.task(
                    "updateX",
                    TransformerComputeKernels::convertFP16toFP32,
                    context,
                    (HalfFloatArray) ws.embeddingX,
                    ws.wrapX);
        }
        return graph.persistOnDevice(
                ws.wrapX, keyStore(state), valueStore(state), ws.wrapBlockTable, ws.wrapConvState);
    }

    private static Object keyStore(Lfm2State state) {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private static Object valueStore(Lfm2State state) {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        scheduler.addWorkerGrid(
                "decodeActivation.updateX", WorkerGridFactory.genericWorker(dim, 128));
        return scheduler;
    }
}
