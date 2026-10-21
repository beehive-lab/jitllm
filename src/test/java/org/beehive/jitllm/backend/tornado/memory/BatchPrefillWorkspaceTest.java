package org.beehive.jitllm.backend.tornado.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceId;
import org.beehive.jitllm.runtime.memory.KeyValueReservation;
import org.beehive.jitllm.runtime.memory.MemoryComponent;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.memory.WeightFootprint;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.DecodedProjections;
import org.junit.Test;

/**
 * The batched-prefill workspace of the tensor-core layers, as the prediction charges it: sized by
 * the layer class, one per session, and resident decoded projections only where the policy allows
 * them and a known budget has room for one copy per session.
 */
public class BatchPrefillWorkspaceTest {

    /** Llama-3.2-1B's shape. */
    private static final LlamaConfiguration LLAMA_1B =
            new LlamaConfiguration("Q8_0", 2048, 8192, 16, 32, 8, 128256, 512, 1e-5f, 500000f);

    /** Its projection weights: 16 x 2048 x (2 x 2048 + 2 x 512 + 3 x 8192). */
    private static final long LLAMA_1B_PROJECTIONS = 16L * 2048 * (2 * 2048 + 2 * 512 + 3 * 8192);

    private static final WeightFootprint FOOTPRINT =
            new WeightFootprint(1_100L << 20, 16 * 9, 280L << 20, 3);

    private static final Device CUDA =
            new Device() {
                @Override
                public DeviceId id() {
                    return DeviceId.of(BackendId.CUDA, "0");
                }

                @Override
                public DeviceCapabilities capabilities() {
                    return DeviceCapabilities.of();
                }

                @Override
                public String displayName() {
                    return "test device";
                }
            };

    private static ExecutionPolicy batched(DecodedProjections decoded) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(128)
                .decodedProjections(decoded)
                .build();
    }

    private static BatchPrefillWorkspace llamaQ8(ExecutionPolicy policy) {
        return LlamaQ8_0LayersBatchPrefillMMA.workspace(LLAMA_1B, 128, policy, false);
    }

    private static MemoryPlan predict(
            ExecutionPolicy policy, long budget, int sessions, BatchPrefillWorkspace workspace) {
        return TornadoMemoryModel.predict(
                FOOTPRINT,
                LLAMA_1B,
                policy,
                CUDA,
                budget,
                new KeyValueReservation(
                        sessions, false, KeyValueReservation.BLOCK_SIZE_TOKENS, true),
                Optional.ofNullable(workspace));
    }

    private static Optional<MemoryComponent> workspaceComponent(MemoryPlan plan) {
        return plan.components().stream()
                .filter(c -> c.name().startsWith("batched prefill workspace"))
                .findFirst();
    }

    @Test
    public void theAllowanceDecidesWhenThePolicyLeavesItToTheBackend() {
        ExecutionPolicy backend = batched(DecodedProjections.BACKEND);
        long atTheAllowance = FP16Projections.RESIDENT_BYTES / 2;
        assertTrue(FP16Projections.resident(backend, atTheAllowance));
        assertFalse(FP16Projections.resident(backend, atTheAllowance + 1));
        assertTrue(
                FP16Projections.resident(batched(DecodedProjections.RESIDENT), Long.MAX_VALUE / 4));
        assertFalse(FP16Projections.resident(batched(DecodedProjections.PER_CHUNK), 1));
    }

    @Test
    public void theLayerClassSizesBothRepresentations() {
        BatchPrefillWorkspace workspace = llamaQ8(batched(DecodedProjections.BACKEND));
        assertEquals(LLAMA_1B_PROJECTIONS, workspace.decodedWeights());
        // Resident: every projection decoded; per chunk: one scratch the widest call wide (the
        // fused gate/up, 2 x 8192 x 2048). Both carry the same split-K partials and flags.
        long widest = 2L * 8192 * 2048;
        assertEquals(
                2 * LLAMA_1B_PROJECTIONS - 2 * widest + 2,
                workspace.residentBytes() - workspace.perChunkBytes());
        // An F16 model without native libraries decodes nothing: only split-K partials.
        BatchPrefillWorkspace f16 =
                LlamaFP16LayersBatchPrefillMMA.workspace(
                        LLAMA_1B, 128, batched(DecodedProjections.BACKEND), false);
        assertFalse(f16.decodesProjections());
        assertEquals(4L * 128 * 2048 * Float.BYTES, f16.perChunkBytes());
    }

    @Test
    public void residencyNeedsRoomForOneCopyPerSession() {
        BatchPrefillWorkspace workspace = llamaQ8(batched(DecodedProjections.BACKEND));
        ExecutionPolicy backend = batched(DecodedProjections.BACKEND);
        long others = 4L << 30;
        // Unknown budget: the allowance alone.
        assertTrue(TornadoMemoryModel.residentDecodedProjections(backend, workspace, others, 0, 2));
        long twoCopies = others + 2 * workspace.residentBytes();
        assertTrue(
                TornadoMemoryModel.residentDecodedProjections(
                        backend, workspace, others, twoCopies, 2));
        assertFalse(
                TornadoMemoryModel.residentDecodedProjections(
                        backend, workspace, others, twoCopies - 1, 2));
        // One session fits where two do not.
        assertTrue(
                TornadoMemoryModel.residentDecodedProjections(
                        backend, workspace, others, twoCopies - 1, 1));
        // A pinned policy is kept.
        assertTrue(
                TornadoMemoryModel.residentDecodedProjections(
                        batched(DecodedProjections.RESIDENT), workspace, others, 1, 2));
        assertFalse(
                TornadoMemoryModel.residentDecodedProjections(
                        batched(DecodedProjections.PER_CHUNK), workspace, others, 0, 2));
    }

    @Test
    public void thePlanChargesTheWorkspaceOncePerSession() {
        ExecutionPolicy backend = batched(DecodedProjections.BACKEND);
        BatchPrefillWorkspace workspace = llamaQ8(backend);
        MemoryPlan plan = predict(backend, 0, 3, workspace);
        MemoryComponent charged = workspaceComponent(plan).orElseThrow();
        assertEquals(3, charged.multiplicity());
        assertEquals(workspace.residentBytes(), charged.logicalBytes());
        assertTrue(charged.name(), charged.name().contains("resident"));
        // Sized from allocation arithmetic, not bisected: never enforced as exact.
        assertEquals(MemoryPlan.Confidence.CONSERVATIVE, plan.confidence());
    }

    @Test
    public void aTightBudgetChargesThePerChunkScratch() {
        ExecutionPolicy backend = batched(DecodedProjections.BACKEND);
        BatchPrefillWorkspace workspace = llamaQ8(backend);
        long others = predict(backend, 0, 2, null).predictedBudgetBytes();
        long budget = others + 2 * workspace.residentBytes() - 1;
        MemoryComponent charged =
                workspaceComponent(predict(backend, budget, 2, workspace)).orElseThrow();
        assertEquals(workspace.perChunkBytes(), charged.logicalBytes());
        assertTrue(charged.name(), charged.name().contains("per chunk"));
    }

    @Test
    public void singleTokenPlansCarryNoWorkspace() {
        ExecutionPolicy single = ExecutionPolicy.builder().build();
        MemoryPlan with = predict(single, 0, 1, llamaQ8(batched(DecodedProjections.BACKEND)));
        MemoryPlan without = predict(single, 0, 1, null);
        assertTrue(workspaceComponent(with).isEmpty());
        assertEquals(without.predictedBudgetBytes(), with.predictedBudgetBytes());
        assertEquals(without.confidence(), with.confidence());
    }
}
