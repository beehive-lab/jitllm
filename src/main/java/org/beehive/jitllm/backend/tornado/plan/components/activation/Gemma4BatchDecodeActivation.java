package org.beehive.jitllm.backend.tornado.plan.components.activation;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
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
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * This family's decode activation graph with the key/value pass-through ("decodeActivation").
 *
 * <p>The counterpart of {@link BatchDecodeActivation} for a family whose cache is flat rather than
 * paged: there is no block table to relay, because there are no blocks — this family addresses one
 * contiguous buffer through a per-layer base offset, and twenty of its thirty-five layers address
 * an earlier layer's slot rather than one of their own.
 *
 * <p><b>The pass-through is host-side state aliasing, not device work.</b> This graph's only task
 * is the embedding conversion; the caches are arguments to no task in it. What the consume/persist
 * pairs do is make TornadoVM point this graph's buffer state for them at the producing graph's, so
 * <em>last batch-prefill layer → decodeActivation → decode layer 0</em> resolves to one live
 * buffer. They look inert in a bytecode trace and are not.
 */
// @formatter:on
public class Gemma4BatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    public Gemma4BatchDecodeActivation(
            State state, Weights weights, Configuration config, String lastBatchLayerId) {
        this.dim = config.dim();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state, weights, lastBatchLayerId).snapshot();
    }

    // @formatter:off
    /**
     * The conversion comes from the embedding tensor's own representation, as {@link
     * org.beehive.jitllm.backend.tornado.layers.Activation} takes it, and not from the model's. A
     * mixed file holds them apart — staging 18-byte blocks to be read as 34-byte ones is a
     * plausible activation and wrong output.
     */
    // @formatter:on
    private TaskGraph buildGraph(
            KernelContext ctx, State state, Weights weights, String lastBatchLayerId) {
        DataType embedding =
                weights instanceof TornadoWeights t ? t.getTokenEmbeddingTable().dataType() : null;
        TaskGraph tg =
                new TaskGraph("decodeActivation")
                        .consumeFromDevice(
                                lastBatchLayerId,
                                state.workspace.wrapKeyCache,
                                state.workspace.wrapValueCache)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX);
        switch (embedding) {
            case Q4_0 ->
                    tg.task(
                            "updateX",
                            TransformerComputeKernels::convertQ4_0toFP32,
                            ctx,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX);
            case F16 ->
                    tg.task(
                            "updateX",
                            TransformerComputeKernels::convertFP16toFP32,
                            ctx,
                            (HalfFloatArray) state.workspace.embeddingX,
                            state.workspace.wrapX);
            case Q8_0 ->
                    tg.task(
                            "updateX",
                            TransformerComputeKernels::convertQ8_0toFP32,
                            ctx,
                            (ByteArray) state.workspace.embeddingX,
                            state.workspace.wrapX);
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 batched decode has no embedding conversion for " + embedding);
        }
        return tg.persistOnDevice(
                state.workspace.wrapX,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache);
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
