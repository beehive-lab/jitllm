package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import static org.junit.Assert.assertEquals;

import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.Qwen3PrefillRegions;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.ops.Implementations;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.DecodedProjections;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The tensor-core prefill of Qwen3, planned from its operations: the memory it predicts equals the
 * formula the layer class used before, and each layer splits into the tasks it has always run.
 */
public class Qwen3ProjectionNeedsTest {

    private static final Target TENSOR_CORES =
            new Target(
                    ExecutionPolicy.builder()
                            .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                            .prefillBatchSize(512)
                            .decodedProjections(DecodedProjections.BACKEND)
                            .build(),
                    DeviceCapabilities.of(DeviceCapability.TENSOR_CORE_MMA));

    private static Qwen3Configuration qwen3(int dim, int hidden, int layers, int heads) {
        return new Qwen3Configuration(
                "Q8_0",
                dim,
                hidden,
                layers,
                heads,
                8,
                128,
                128,
                151936,
                40960,
                4096,
                true,
                1e-6f,
                1_000_000f);
    }

    private static List<Qwen3PrefillRegions.Layer> regions(Qwen3Configuration config, int batch) {
        int padded = (batch + 127) & ~127;
        return Qwen3PrefillRegions.of(config, batch, padded, (role, layer) -> DataType.Q4_K);
    }

    /** The old {@code Qwen3Q8_0LayersBatchPrefillMMA.workspace} formula, without attention. */
    private static BatchPrefillWorkspace before(Qwen3Configuration config, int batchSize) {
        int padded = (batchSize + 127) & ~127;
        long dim = config.dim();
        long qDim = (long) config.numberOfHeadsKey() * config.numberOfHeads();
        long kvDim = (long) config.numberOfHeadsValue() * config.numberOfKeyValueHeads();
        long weights =
                config.numberOfLayers() * dim * (2L * qDim + 2L * kvDim + 3L * config.hiddenDim());
        long widest =
                Math.max(
                        Math.max((qDim + 2L * kvDim) * dim, 2L * config.hiddenDim() * dim),
                        Math.max(dim * qDim, dim * config.hiddenDim()));
        int calls = 4 * config.numberOfLayers();
        return new BatchPrefillWorkspace(
                weights,
                FP16Projections.residentBytes(weights)
                        + FP16Projections.workspaceBytes(
                                true, widest, padded, config.dim(), calls, false),
                FP16Projections.workspaceBytes(false, widest, padded, config.dim(), calls, false));
    }

    @Test
    public void theNeedsAreTheOldFormula() {
        for (Qwen3Configuration config :
                List.of(
                        qwen3(1024, 3072, 28, 16),
                        qwen3(2048, 6144, 28, 16),
                        qwen3(4096, 12288, 36, 32))) {
            for (int batch : new int[] {128, 300, 512}) {
                assertEquals(
                        before(config, batch),
                        WorkPlan.needs(
                                Qwen3PrefillRegions.all(regions(config, batch)),
                                Implementations.tensorCorePrefill(),
                                TENSOR_CORES));
            }
        }
    }

    @Test
    public void eachLayerSplitsIntoTheTasksItHasAlwaysRun() {
        Qwen3PrefillRegions.Layer layer = regions(qwen3(1024, 3072, 28, 16), 512).get(0);
        assertEquals(
                List.of("RMS norm (none)", "q|k|v (none)"),
                WorkPlan.describe(
                        layer.beforeAttention(),
                        Implementations.tensorCorePrefill(),
                        TENSOR_CORES));
        assertEquals(
                List.of(
                        "projection (none)",
                        "residual add + RMS norm (none)",
                        "gate|up + SwiGLU (none)",
                        "down + residual add (none)"),
                WorkPlan.describe(
                        layer.afterAttention(), Implementations.tensorCorePrefill(), TENSOR_CORES));
    }
}
