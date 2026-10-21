package org.beehive.jitllm.backend.tornado.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.model.mistral.MistralConfiguration;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceId;
import org.beehive.jitllm.runtime.memory.KeyValueReservation;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.memory.WeightFootprint;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.DecodedProjections;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

// @formatter:off
/**
 * What the MLX batched prefill adds to a Metal plan ({@link MlxPrefill#workspace}), against what it
 * was measured to allocate, and the refusal of a load whose buffers alone exceed the budget.
 *
 * <h2>The measurement these sizes are checked against</h2>
 *
 * <p>{@code --print-bytecodes} on Qwen3-0.6B, context 4096, prefill width 128, FP16 key/value
 * cache, Apple M4 Pro, TornadoVM develop 3d2d9695 — distinct {@code (graph, object)} device
 * allocations, MLX on ({@code --with-native-libraries}) against the JIT kernels:
 *
 * <pre>
 *   weights  representation       JIT        MLX     delta   of which in the decode family
 *   F16      stacked QKV/gate-up  1591.2   2171.2   +580.0   560.1 (wq wk wv w1 w3, 28 layers)
 *   Q8_0     resident FP16        1057.8   1918.1   +860.3   446.4 (every projection)
 *   Q8_0     affine               1057.8   1550.6   +492.8   446.4 (every projection)
 * </pre>
 *
 * <p>The decode family's growth is the model weights the prefill graphs stopped binding, and it is
 * exactly what the prefill family shed: the per-layer weights stay counted once. What remains is
 * the MLX form of the projections plus the shared staging and the contiguous key/value buffers,
 * which is what the workspace has to predict — never less than measured, and not by much more.
 */
// @formatter:on
public class MlxPrefillMemoryTest {

    private static final double MIB = 1048576.0;

    /** Qwen3-0.6B at context 4096. */
    private static final Qwen3Configuration QWEN3_0_6B =
            new Qwen3Configuration(
                    "F16",
                    1024,
                    3072,
                    28,
                    16,
                    8,
                    128,
                    128,
                    151936,
                    40960,
                    4096,
                    true,
                    1e-6f,
                    1_000_000f);

    /** Mistral-7B-v0.3 at the context a pp2048 benchmark sizes. */
    private static final MistralConfiguration MISTRAL_7B =
            new MistralConfiguration(
                    "Q8_0", 4096, 14336, 32, 32, 8, 32768, 2064, false, 1e-5f, 1_000_000f);

    private static final Device METAL =
            new Device() {
                @Override
                public DeviceId id() {
                    return DeviceId.of(BackendId.METAL, "Apple M4 Pro");
                }

                @Override
                public DeviceCapabilities capabilities() {
                    return DeviceCapabilities.of();
                }

                @Override
                public String displayName() {
                    return "test Metal device";
                }
            };

    private static double mib(long bytes) {
        return bytes / MIB;
    }

    /** At least what was measured, and within the headers and rounding a plan does not model. */
    private static void assertPredicts(String what, double measuredMib, long predictedBytes) {
        double predicted = mib(predictedBytes);
        assertTrue(
                what + ": predicted " + predicted + " MiB under the measured " + measuredMib,
                predicted >= measuredMib);
        assertTrue(
                what + ": predicted " + predicted + " MiB, more than 4 MiB over " + measuredMib,
                predicted <= measuredMib + 4);
    }

    @Test
    public void projectionWeightsAreEveryLayersFourProjections() {
        // 28 x (1024 x (2048 + 2 x 1024) + 2048 x 1024 + 3 x 1024 x 3072)
        assertEquals(440_401_920L, MlxPrefill.projectionWeights(QWEN3_0_6B));
    }

    @Test
    public void f16StackedCopiesAndStagingMatchTheMeasurement() {
        BatchPrefillWorkspace ws =
                MlxPrefill.workspace(DataType.F16, QWEN3_0_6B, 128, false, false);
        assertFalse("F16 decodes nothing, so residency is not a choice", ws.decodesProjections());
        assertPredicts("Qwen3-0.6B F16 MLX workspace", 580.0, ws.residentBytes());
    }

    @Test
    public void quantizedFormsMatchTheMeasurement() {
        BatchPrefillWorkspace ws =
                MlxPrefill.workspace(DataType.Q8_0, QWEN3_0_6B, 128, false, false);
        assertTrue(ws.decodesProjections());
        assertEquals(MlxPrefill.projectionWeights(QWEN3_0_6B), ws.decodedWeights());
        assertPredicts("Qwen3-0.6B Q8_0 resident FP16", 860.3, ws.bytes(true));
        assertPredicts("Qwen3-0.6B Q8_0 affine", 492.8, ws.bytes(false));
    }

    @Test
    public void affineBytesArePackedValuesPlusAScaleAndBiasPerGroup() {
        assertEquals(1024 + 32 * 4, MlxPrefill.affineBytes(DataType.Q8_0, 1024));
        assertEquals(512 + 32 * 4, MlxPrefill.affineBytes(DataType.Q4_0, 1024));
    }

    @Test
    public void aFusedProjectionIsBoundNotCopied() {
        long split =
                MlxPrefill.workspace(DataType.F16, QWEN3_0_6B, 128, false, false).residentBytes();
        long fused =
                MlxPrefill.workspace(DataType.F16, QWEN3_0_6B, 128, true, true).residentBytes();
        assertEquals(
                "the stacked copies, 560 MiB, are all the difference",
                560.0,
                mib(split - fused),
                0.01);
    }

    // ── the regression: Mistral-7B Q8_0 with MLX at pp2048 b2048 under 14 GB ─────

    private static final long BUDGET_14_GB = 14L << 30;

    /** Mistral-7B-v0.3 Q8_0: per-layer projections and norms, and the two vocabulary tables. */
    private static final WeightFootprint MISTRAL_7B_Q8_0 =
            new WeightFootprint(
                    MlxPrefill.projectionWeights(MISTRAL_7B) * 34 / 32 + 32L * 2 * 4096 * 4,
                    32 * 9,
                    2L * 32768 * 4096 * 34 / 32,
                    3);

    private static ExecutionPolicy mlxPolicy() {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(2048)
                .nativeLibraries(true)
                .decodedProjections(DecodedProjections.BACKEND)
                .build();
    }

    private static MemoryPlan predict(Optional<BatchPrefillWorkspace> workspace) {
        return TornadoMemoryModel.predict(
                MISTRAL_7B_Q8_0,
                MISTRAL_7B,
                mlxPolicy(),
                METAL,
                BUDGET_14_GB,
                new KeyValueReservation(1, false, KeyValueReservation.BLOCK_SIZE_TOKENS, true),
                workspace);
    }

    @Test
    public void mistral7bQ8MlxIsRefusedBeforeAllocationWhereTheJitKernelsFit() {
        BatchPrefillWorkspace mlx =
                MlxPrefill.workspace(DataType.Q8_0, MISTRAL_7B, 2048, false, false);
        // Its FP16 copies (13 GiB) exceed the backend's resident allowance, so the residency
        // resolves to the affine repack — the form the failing run used.
        assertFalse(
                TornadoMemoryModel.residentDecodedProjections(
                        mlxPolicy(), mlx, 0, BUDGET_14_GB, 1));

        MemoryPlan withMlx = predict(Optional.of(mlx));
        MemoryPlan jit = predict(Optional.empty());

        assertEquals(
                "Metal plans are not bisected, so they stay conservative",
                MemoryPlan.Confidence.CONSERVATIVE,
                withMlx.confidence());
        assertTrue(
                "the weights, the affine repack and the cache alone exceed 14 GiB: "
                        + withMlx.describe(),
                withMlx.exceedsConfiguredBudgetAtLowerBound());
        assertFalse(
                "the JIT kernels' plan fits, as the run without native libraries did: "
                        + jit.describe(),
                jit.exceedsConfiguredBudgetAtLowerBound());
        System.out.println("Mistral-7B Q8_0, MLX:\n" + withMlx.describe());
        System.out.println("Mistral-7B Q8_0, JIT kernels:\n" + jit.describe());
    }

    @Test
    public void aLowerBoundIsNeverClaimedWithoutABudgetOrForAnUnknownTopology() {
        MemoryPlan plan =
                predict(
                        Optional.of(
                                MlxPrefill.workspace(
                                        DataType.Q8_0, MISTRAL_7B, 2048, false, false)));
        MemoryPlan noBudget =
                new MemoryPlan(plan.components(), 0, plan.confidence(), plan.assumptions());
        MemoryPlan unknown =
                new MemoryPlan(
                        plan.components(),
                        plan.configuredBudgetBytes(),
                        MemoryPlan.Confidence.UNSUPPORTED,
                        plan.assumptions());
        assertFalse(noBudget.exceedsConfiguredBudgetAtLowerBound());
        assertFalse(unknown.exceedsConfiguredBudgetAtLowerBound());
    }

    @Test
    public void theLowerBoundIgnoresDuplication() {
        // A budget between the logical bytes and the duplicated prediction: over at the upper
        // bound, not at the lower, so a conservative plan must not refuse it.
        MemoryPlan plan = predict(Optional.empty());
        long between = plan.logicalBytes() + 1;
        MemoryPlan tight =
                new MemoryPlan(plan.components(), between, plan.confidence(), plan.assumptions());
        assertFalse(tight.exceedsConfiguredBudgetAtLowerBound());
    }
}
