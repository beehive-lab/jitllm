package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * What the {@code qwen35} MLX projections hold, counted by layer kind: an attention layer every
 * fourth, delta-net layers otherwise, and the dense feed-forward in all of them.
 */
public class Qwen35MlxPrefillTest {

    /** Qwen3.5-0.8B's shape. */
    private static Qwen35Configuration qwen35_0_8b() {
        return new Qwen35Configuration(
                "Q8_0", 1024, 3584, 24, 1, 8, 2, 256, 256, 4, 4, 128, 16, 16, 2048, 64, 248320,
                262144, 4096, 1e-6f, 1e7f);
    }

    @Test
    public void projectionWeightsFollowEachLayersKind() {
        Qwen35Configuration config = qwen35_0_8b();
        // Six attention layers: query/gate 4096, key and value 512 each, output from 2048.
        long attention = 1024L * (4096 + 2 * 512) + 2048L * 1024;
        // Eighteen delta-net layers: convolution input 6144, output gate 2048, beta and alpha 16
        // each, readout from 2048.
        long deltaNet = 1024L * (6144 + 2048 + 2 * 16) + 2048L * 1024;
        long feedForward = 3L * 1024 * 3584;
        assertEquals(6 * attention + 18 * deltaNet + 24 * feedForward, config.projectionWeights());
    }

    @Test
    public void theFp16CopiesAreResidentByDefaultAndTheRepackIsSmaller() {
        Qwen35Configuration config = qwen35_0_8b();
        long projections = config.projectionWeights();
        assertTrue(FP16Projections.residentBytes(projections) <= FP16Projections.RESIDENT_BYTES);
        BatchPrefillWorkspace ws = Qwen35MlxPrefill.workspace(DataType.Q8_0, config, 512);
        assertEquals(projections, ws.decodedWeights());
        assertEquals(
                FP16Projections.residentBytes(projections)
                        - MlxPrefill.affineBytes(DataType.Q8_0, projections),
                ws.residentBytes() - ws.perChunkBytes());
    }
}
