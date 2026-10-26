package org.beehive.jitllm.backend.tornado.memory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
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
import org.junit.Test;

/**
 * The scratch buffer packed projections are repacked through ({@code PackedRepack}), as the
 * prediction charges it: absent when nothing is packed, and otherwise one per session, the largest
 * packed projection plus the flag beside it.
 */
public class PackedRepackScratchTest {

    /** Llama-3.2-1B's shape. */
    private static final LlamaConfiguration LLAMA_1B =
            new LlamaConfiguration("Q4_0", 2048, 8192, 16, 32, 8, 128256, 512, 1e-5f, 500000f);

    private static final WeightFootprint FOOTPRINT =
            new WeightFootprint(600L << 20, 16 * 9, 280L << 20, 3);

    /** Its largest projection, {@code ffn_gate}, as Q4_0: 8192 rows of 2048 / 32 blocks. */
    private static final long LARGEST_Q4_0_PROJECTION = 8192L * (2048 / 32) * 18;

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

    private static MemoryPlan predict(int sessions, long scratch) {
        return TornadoMemoryModel.predict(
                FOOTPRINT,
                LLAMA_1B,
                ExecutionPolicy.builder().build(),
                CUDA,
                0,
                new KeyValueReservation(
                        sessions, false, KeyValueReservation.BLOCK_SIZE_TOKENS, false),
                Optional.empty(),
                scratch);
    }

    private static Optional<MemoryComponent> scratchComponent(MemoryPlan plan) {
        return plan.components().stream()
                .filter(c -> c.name().equals("packed-weight repack scratch"))
                .findFirst();
    }

    @Test
    public void aLoadThatPacksNothingHasNoScratch() {
        assertTrue(scratchComponent(predict(1, 0)).isEmpty());
    }

    @Test
    public void theScratchIsTheLargestPackedProjectionPlusItsFlag() {
        MemoryComponent scratch = scratchComponent(predict(1, LARGEST_Q4_0_PROJECTION)).get();

        assertEquals(LARGEST_Q4_0_PROJECTION + Integer.BYTES, scratch.logicalBytes());
        assertEquals(1, scratch.multiplicity());
    }

    @Test
    public void eachSessionRepacksThroughItsOwnScratch() {
        long one = predict(1, 0).predictedBudgetBytes();
        long withOne = predict(1, LARGEST_Q4_0_PROJECTION).predictedBudgetBytes();
        long withThree = predict(3, LARGEST_Q4_0_PROJECTION).predictedBudgetBytes();
        long three = predict(3, 0).predictedBudgetBytes();
        long header = CUDA.nativeArrayHeaderBytes();

        assertEquals(LARGEST_Q4_0_PROJECTION + Integer.BYTES + 2 * header, withOne - one);
        assertEquals(3 * (LARGEST_Q4_0_PROJECTION + Integer.BYTES + 2 * header), withThree - three);
    }
}
