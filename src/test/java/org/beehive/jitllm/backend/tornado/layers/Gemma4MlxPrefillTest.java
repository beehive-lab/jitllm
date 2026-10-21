package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * What Gemma 4's MLX projections hold, counted per layer: two head widths, two feed-forward widths,
 * and layers that share an earlier layer's cache projecting no key or value.
 */
public class Gemma4MlxPrefillTest {

    private static final int DIM = 1536;
    private static final int LAYERS = 35;
    private static final int HEADS = 8;
    private static final int SHARED_KV = 20;

    /** Gemma 4 E2B's shape: every fifth layer global, the last twenty sharing a cache. */
    private static Gemma4Configuration e2b() {
        int[] ff = new int[LAYERS];
        boolean[] swa = new boolean[LAYERS];
        for (int l = 0; l < LAYERS; l++) {
            ff[l] = l < 15 ? 6144 : 12288;
            swa[l] = l % 5 != 4;
        }
        return new Gemma4Configuration(
                "Q8_0", DIM, LAYERS, HEADS, 1, 256, 512, ff, swa, 512, SHARED_KV, 256, 262144,
                131072, 4096, 1e-6f, 1e6f, 1e4f, 30f);
    }

    @Test
    public void projectionWeightsFollowEachLayersShape() {
        Gemma4Configuration config = e2b();
        long expected = 0;
        for (int l = 0; l < LAYERS; l++) {
            long headDim = l % 5 != 4 ? 256 : 512;
            long qDim = HEADS * headDim;
            long qkv = l < LAYERS - SHARED_KV ? qDim + 2 * headDim : qDim;
            long ff = l < 15 ? 6144 : 12288;
            expected += DIM * qkv + qDim * DIM + 3 * DIM * ff;
        }
        assertEquals(expected, config.projectionWeights());
    }

    @Test
    public void theFp16CopiesAreResidentByDefaultAndTheRepackIsSmaller() {
        Gemma4Configuration config = e2b();
        long projections = config.projectionWeights();
        assertTrue(
                "E2B's FP16 copies should stay resident under the default budget",
                FP16Projections.residentBytes(projections) <= FP16Projections.RESIDENT_BYTES);

        BatchPrefillWorkspace ws =
                Gemma4BatchPrefillLayers.mlxWorkspace(DataType.Q8_0, config, 512);
        assertTrue(ws.decodesProjections());
        assertEquals(projections, ws.decodedWeights());
        assertEquals(
                FP16Projections.residentBytes(projections)
                        - MlxPrefill.affineBytes(DataType.Q8_0, projections),
                ws.residentBytes() - ws.perChunkBytes());
        long staging = ws.residentBytes() - FP16Projections.residentBytes(projections);
        // An FP16 buffer of 512 rows per distinct width: [q|k|v] products of 2560 and 5120 (sliding
        // and global layers that own a cache) and 2048 and 4096 (those that share one), attention
        // outputs of 2048 and 4096, gate/up products and GeGLU outputs of both feed-forward widths,
        // and the dim-wide product; and a row factor per row.
        long widths = 2560 + 5120 + 2048 + 4096 + 2048 + 4096 + 3 * (6144 + 12288) + DIM;
        assertEquals(2L * 512 * widths + 4L * 512, staging);
    }
}
