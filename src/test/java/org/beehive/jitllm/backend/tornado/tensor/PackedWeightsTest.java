package org.beehive.jitllm.backend.tornado.tensor;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/** The prefill widths a tiled plan runs at: any but a batched width the int8 GEMM does not tile. */
public class PackedWeightsTest {

    private static ExecutionPolicy batched(int width) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(width)
                .build();
    }

    @Test
    public void onlyAnUntiledBatchedPrefillWidthDoesNotTile() {
        assertTrue(PackedWeights.prefillWidthTiles(ExecutionPolicy.builder().build()));
        assertTrue(PackedWeights.prefillWidthTiles(batched(1)));
        assertTrue(PackedWeights.prefillWidthTiles(batched(384)));
        assertFalse(PackedWeights.prefillWidthTiles(batched(64)));
    }
}
