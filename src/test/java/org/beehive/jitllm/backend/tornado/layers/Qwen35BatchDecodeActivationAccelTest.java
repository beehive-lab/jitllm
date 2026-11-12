package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Qwen35BatchDecodeActivationAccelTest {

    @Test
    public void f16EmbeddingBesideQuantizedWeightsIsRefreshedOnEveryExecution() throws Exception {
        Qwen35Configuration config =
                new Qwen35Configuration(
                        "Q8_0", 256, 512, 2, 0, 4, 2, 32, 32, 2, 4, 32, 2, 4, 128, 32, 32, 32, 32,
                        1e-6f, 1e7f);
        Qwen35State state = Qwen35State.withDeviceArrays(true, () -> new Qwen35State(config, -1));
        assertTrue(((ByteArray) state.workspace.embeddingX).getSize() < config.dim() * 2);

        // Only the embedding is read by this graph; the quantized projection tensors are unused.
        Qwen35TornadoWeights weights = embeddingOnlyWeights(config);
        Qwen35BatchDecodeActivation activation =
                new Qwen35BatchDecodeActivation(state, weights, config, "prefill");

        Object key =
                state.usesFp16KeyValueCache()
                        ? state.workspace.wrapKeyCacheFP16
                        : state.workspace.wrapKeyCache;
        Object value =
                state.usesFp16KeyValueCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;
        Object[] carried = {
            state.workspace.wrapX,
            state.workspace.wrapBlockTable,
            state.workspace.wrapConvState,
            state.workspace.wrapDeltaState,
            key,
            value
        };
        // Supply the device state that the real activation consumes from the last prefill layer.
        TaskGraph prefill =
                new TaskGraph("prefill")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, carried)
                        .task(
                                "carry",
                                TransformerBatchPrefillKernels::batchPassthrough,
                                new KernelContext(),
                                state.workspace.wrapX)
                        .persistOnDevice(carried);
        GridScheduler grids = activation.updateGridScheduler(new GridScheduler());
        grids.addWorkerGrid("prefill.carry", WorkerGridFactory.createSingleWorker());

        ByteArray staging = (ByteArray) state.workspace.embeddingX;
        assertEquals(config.dim() * 2, staging.getSize());
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(prefill.snapshot(), activation.getImmutableTaskGraph())) {
            plan.withGridScheduler(grids);
            for (int step = 1; step <= 2; step++) {
                for (int i = 0; i < config.dim(); i++) {
                    short half = Float.floatToFloat16(step + i / 16.0f);
                    staging.set(2 * i, (byte) half);
                    staging.set(2 * i + 1, (byte) (half >>> 8));
                }
                plan.execute().transferToHost(state.workspace.wrapX);
                System.out.printf(
                        "embedding execution %d: expected %.1f, actual %.1f%n",
                        step, (float) step, state.workspace.wrapX.get(0));
                for (int i = 0; i < config.dim(); i++) {
                    assertEquals(
                            "execution " + step + ", embedding element " + i,
                            step + i / 16.0f,
                            state.workspace.wrapX.get(i),
                            0.0f);
                }
            }
        }
    }

    private static Qwen35TornadoWeights embeddingOnlyWeights(Qwen35Configuration config) {
        return new Qwen35TornadoWeights(
                config.numberOfBlocks(),
                new FP16TornadoTensor(new HalfFloatArray(config.dim()), Shape.of(config.dim())),
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                DataType.Q8_0);
    }
}
