package org.beehive.jllm.backend.tornado.layers;

import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

// @formatter:off
/**
 * The decode activation of the batched plan: one token's embedding, and the hand-over of everything
 * batch prefill left on the device.
 *
 * <p>Two things separate it from the shared {@code BatchDecodeActivation}.
 *
 * <p><b>It converts a Q4_0 embedding row.</b> The shared graph knows F16 and Q8_0, and this
 * family's token embeddings are retained Q4_0 — an 18-byte block against a 34-byte one, which would
 * be read as a plausible activation and wrong output.
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
                new TaskGraph("decodeActivation")
                        .consumeFromDevice(lastBatchLayerId, keyStore(state), valueStore(state))
                        .consumeFromDevice(lastBatchLayerId, state.workspace.wrapBlockTable)
                        .consumeFromDevice(
                                lastBatchLayerId,
                                state.workspace.wrapConvState,
                                state.workspace.wrapDeltaState)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX);

        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        switch (embedding) {
            case Q4_0 ->
                    graph.task(
                            "updateX",
                            TransformerComputeKernels::convertQ4_0toFP32,
                            context,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX);
            case Q8_0 ->
                    graph.task(
                            "updateX",
                            TransformerComputeKernels::convertQ8_0toFP32,
                            context,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX);
            default ->
                    throw new UnsupportedOperationException(
                            "qwen35 batch decode stages a "
                                    + embedding
                                    + " embedding row, for which this graph has no conversion. It"
                                    + " is not read as another representation to get one.");
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
