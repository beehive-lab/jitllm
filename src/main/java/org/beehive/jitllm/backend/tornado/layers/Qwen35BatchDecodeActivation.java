package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
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
 * The decode activation of the batched plan: one token's embedding, and the hand-over of everything
 * batch prefill left on the device.
 *
 * <p>It decodes the embedding row in the embedding tensor's own type, as the shared {@code
 * BatchDecodeActivation} does: a Q4_0 file from unsloth keeps its token embeddings Q6_K.
 *
 * <p><b>It relays the recurrent state as well as the key/value store.</b> A recurrence is the
 * sequence's whole history: the convolution windows and delta-net matrices the prefill chunks
 * advanced are what decode continues from, so they travel the same chain the caches do — last batch
 * layer, this graph, decode layer 0. The consume/persist pairs look inert in a bytecode trace and
 * are not: they are what makes those three graphs resolve to one live buffer per array.
 */
// @formatter:on
public class Qwen35BatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    public Qwen35BatchDecodeActivation(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            String lastBatchLayerId) {
        this.dim = config.dim();
        this.itg = build(new KernelContext(), state, weights, lastBatchLayerId).snapshot();
    }

    private TaskGraph build(
            KernelContext context,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            String lastBatchLayerId) {
        TaskGraph graph =
                new TaskGraph(GraphNames.DECODE_ACTIVATION)
                        .consumeFromDevice(lastBatchLayerId, keyStore(state), valueStore(state))
                        .consumeFromDevice(lastBatchLayerId, state.workspace.wrapBlockTable)
                        .consumeFromDevice(
                                lastBatchLayerId,
                                state.workspace.wrapConvState,
                                state.workspace.wrapDeltaState)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX);

        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        TornadoWorkspaces.stagingHolds(state.workspace, embedding, dim);
        if (Activation.stagedAsBytes(embedding, state.workspace.embeddingX)) {
            graph.task(
                    "updateX",
                    Activation.embeddingRowDecode(embedding),
                    context,
                    (ByteArray) state.workspace.embeddingX,
                    state.workspace.wrapX);
        } else if (embedding == DataType.Q8_0) {
            graph.task(
                    "updateX",
                    TransformerComputeKernels::convertQ8_0toFP32,
                    context,
                    (ByteArray) state.workspace.embeddingX,
                    state.workspace.wrapX);
        } else {
            graph.task(
                    "updateX",
                    TransformerComputeKernels::convertFP16toFP32,
                    context,
                    (HalfFloatArray) state.workspace.embeddingX,
                    state.workspace.wrapX);
        }

        graph.persistOnDevice(state.workspace.wrapBlockTable);
        graph.persistOnDevice(state.workspace.wrapConvState, state.workspace.wrapDeltaState);
        return graph.persistOnDevice(state.workspace.wrapX, keyStore(state), valueStore(state));
    }

    // @formatter:off
    /**
     * The key/value store this session actually holds, in whichever precision it is in.
     *
     * <p>This graph is the only link between the batch-prefill layers and the decode layers, and
     * both of those bind the store through the same choice. Binding the FP32 carriers here
     * unconditionally — which is what this did — left a half-precision session's prefilled history
     * on one pair of arrays and decode reading another, so the model answered as though the prompt
     * had never been read. The FP32 arrays exist either way, which is why it did not fail loudly.
     */
    // @formatter:on
    private static Object keyStore(Qwen35State state) {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private static Object valueStore(Qwen35State state) {
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
