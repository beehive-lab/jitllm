package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Decode activation graph with KV-cache pass-through ("decodeActivation").
 *
 * <p>Used in the 2N+3 batch-prefill/decode plan. Consumes {@code wrapKeyCache}/{@code
 * wrapValueCache} from the last batch-prefill layer, converts the single-token embedding to FP32,
 * then re-persists the KV cache so that decode layer 0 can consume it.
 *
 * <p><b>The pass-through is host-side state aliasing, not device work, and it needs TornadoVM >=
 * 6.0.0.</b> This graph's only task is {@code updateX(ctx, embeddingX, wrapX)}; the KV cache and
 * block table are arguments to no task in it, so it emits no bytecode for them at all — under
 * {@code --print-bytecodes} this graph shows exactly one {@code PERSIST}, for {@code wrapX}. What
 * the consume/persist pairs below actually do is make TornadoVM point this graph's {@code
 * XPUDeviceBufferState} for those buffers at the producing graph's, so the chain <em>last
 * batch-prefill layer → decodeActivation → decode layer 0</em> resolves to one live buffer.
 *
 * <p>That only works where {@code consumeFromDevice(producerName,.)} resolves by <em>name</em>.
 * TornadoVM 5.2.0 ignored the name and aliased from whichever graph ran previously, which in this
 * plan is this graph itself from the second decode token onward — so decode layer 0 imported this
 * graph's null buffer and died in {@code executeAlloc} with {@code NullPointerException:
 * XPUDeviceBufferState.getXPUBuffer() is null}. Fixed upstream in {@code 8603a9fa2d} (PR #996),
 * released in 6.0.0, which is the pinned floor for batched prefill. Do not "simplify" these calls
 * away: they look inert in a bytecode trace and are not.
 */
/**
 * {@link org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation} with
 * Granite's embedding multiplier applied as the embedding is converted.
 */
public class GraniteBatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    private final float embeddingScale;

    public GraniteBatchDecodeActivation(
            State state, GraniteConfiguration config, String lastBatchLayerId, boolean isQ8) {
        this.dim = config.dim();
        this.embeddingScale = config.embeddingScale();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state, lastBatchLayerId, isQ8, false).snapshot();
    }

    // @formatter:off
    private TaskGraph buildGraph(
            KernelContext ctx,
            State state,
            String lastBatchLayerId,
            boolean isQ8,
            boolean q4Embeddings) {
        boolean fp16KV = state.usesFp16KeyValueCache();
        Object keyCache = fp16KV ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                fp16KV ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
        TaskGraph tg =
                new TaskGraph("decodeActivation")
                        .consumeFromDevice(lastBatchLayerId, keyCache, valueCache)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX);
        // The block table travels the same chain as the caches it addresses: the batch-prefill
        // graphs allocate it, this graph passes it through, decode layer 0 consumes it.
        tg.consumeFromDevice(lastBatchLayerId, state.workspace.wrapBlockTable);
        if (isQ8) {
            tg.task(
                    "updateX",
                    GraniteKernels::convertQ8_0toFP32withGraniteScale,
                    ctx,
                    (ByteArray) state.workspace.embeddingX,
                    state.workspace.wrapX,
                    embeddingScale);
        } else {
            tg.task(
                    "updateX",
                    GraniteKernels::convertFP16toFP32withGraniteScale,
                    ctx,
                    (HalfFloatArray) state.workspace.embeddingX,
                    state.workspace.wrapX,
                    embeddingScale);
        }
        tg.persistOnDevice(state.workspace.wrapBlockTable);
        return tg.persistOnDevice(state.workspace.wrapX, keyCache, valueCache);
    }

    // @formatter:on

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
