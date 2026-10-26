package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.After;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Which weights a plan holds in tiles: the projections of a type the family reads as tiles, whose
 * shape divides into 128 x 64 tiles, where packing is on.
 *
 * <p>No device: packing is switched on with {@code -Djitllm.q4.packed=true}, which decides in place
 * of the device check, and cleared after each test.
 */
public class TileLayoutTest {

    private static final ExecutionPolicy POLICY = ExecutionPolicy.builder().build();

    @After
    public void clearProperties() {
        System.clearProperty(PackedWeights.Q4_PROPERTY);
        System.clearProperty(PackedWeights.Q8_PROPERTY);
    }

    private static TornadoTensor q4(int rows, int cols) {
        return new Q4_0TornadoTensor(new ByteArray(rows * cols / 32 * 18), Shape.of(cols, rows));
    }

    private static TornadoTensor f32(int elements) {
        return new FP32TornadoTensor(new FloatArray(elements), Shape.of(elements));
    }

    /**
     * One layer: Q4_0 projections that tile, a Q4_1 down projection, and one too narrow to tile.
     */
    private record Layer(
            TornadoTensor wq,
            TornadoTensor wk,
            TornadoTensor wv,
            TornadoTensor wo,
            TornadoTensor gate,
            TornadoTensor down,
            TornadoTensor up,
            TornadoTensor embeddings) {}

    private static Layer layer() {
        return new Layer(
                q4(128, 64),
                q4(128, 64),
                q4(96, 64), // 96 rows do not divide into 128-row tiles
                q4(128, 128),
                q4(256, 64),
                new Q4_1TornadoTensor(new ByteArray(256 * 64 / 32 * 20), Shape.of(256, 64)),
                q4(256, 64),
                q4(512, 64)); // the embedding table is not a projection
    }

    private static LlamaTornadoWeights weights(Layer l) {
        TornadoTensor[] norm = {f32(64)};
        return new LlamaTornadoWeights(
                l.embeddings(),
                norm,
                new TornadoTensor[] {l.wq()},
                new TornadoTensor[] {l.wk()},
                new TornadoTensor[] {l.wv()},
                new TornadoTensor[] {l.wo()},
                norm,
                new TornadoTensor[] {l.gate()},
                new TornadoTensor[] {l.down()},
                new TornadoTensor[] {l.up()},
                f32(64),
                f32(8),
                f32(8),
                l.embeddings(),
                DataType.Q4_0);
    }

    @Test
    public void tilesTheProjectionsTheFamilyReadsAsTilesWhereTheirShapeFits() {
        System.setProperty(PackedWeights.Q4_PROPERTY, "true");
        Layer l = layer();

        TileLayout layout = TileLayout.forPlan(new LlamaPlanProvider(), weights(l), POLICY);

        assertTrue(layout.tiledQ4(l.wq()));
        assertTrue(layout.tiledQ4(l.wk()));
        assertTrue(layout.tiledQ4(l.wo()));
        assertTrue(layout.tiledQ4(l.gate()));
        assertTrue(layout.tiledQ4(l.up()));
        assertFalse("96 rows do not tile", layout.tiled(l.wv()));
        assertFalse("Q4_1 is not a tile type", layout.tiled(l.down()));
        assertFalse("not a projection", layout.tiled(l.embeddings()));
        assertFalse(layout.tiledQ8(l.wq()));
    }

    @Test
    public void tilesNothingWherePackingIsOff() {
        System.setProperty(PackedWeights.Q4_PROPERTY, "false");

        assertSame(
                TileLayout.NONE,
                TileLayout.forPlan(new LlamaPlanProvider(), weights(layer()), POLICY));
    }

    @Test
    public void tilesNothingOfATypeTheFamilyDoesNotReadAsTiles() {
        System.setProperty(PackedWeights.Q8_PROPERTY, "true");
        TornadoTensor q8 =
                new Q8_0TornadoTensor(new ByteArray(128 * 64 / 32 * 34), Shape.of(64, 128));
        Layer l = layer();
        LlamaTornadoWeights withQ8 =
                new LlamaTornadoWeights(
                        l.embeddings(),
                        new TornadoTensor[] {f32(64)},
                        new TornadoTensor[] {q8},
                        new TornadoTensor[] {l.wk()},
                        new TornadoTensor[] {l.wv()},
                        new TornadoTensor[] {l.wo()},
                        new TornadoTensor[] {f32(64)},
                        new TornadoTensor[] {l.gate()},
                        new TornadoTensor[] {l.down()},
                        new TornadoTensor[] {l.up()},
                        f32(64),
                        f32(8),
                        f32(8),
                        l.embeddings(),
                        DataType.Q8_0);

        TileLayout layout = TileLayout.forPlan(new LlamaPlanProvider(), withQ8, POLICY);

        assertFalse("Llama reads only Q4_0 as tiles", layout.tiled(q8));
    }

    @Test
    public void aFamilyThatReadsNoTilesUnderAPolicyTilesNothing() {
        System.setProperty(PackedWeights.Q4_PROPERTY, "true");
        ExecutionPolicy nativeLibraries = ExecutionPolicy.builder().nativeLibraries(true).build();

        assertEquals(
                "Gemma 4's cuBLAS projections read no tiles",
                Set.of(),
                TileLayout.tiledTypes(new Gemma4PlanProvider(), nativeLibraries));
        assertEquals(
                Set.of(DataType.Q4_0), TileLayout.tiledTypes(new Gemma4PlanProvider(), POLICY));
    }

    @Test
    public void aLayoutComparesTensorsByIdentity() {
        TornadoTensor tiled = q4(128, 64);
        TornadoTensor sameShape = q4(128, 64);

        TileLayout layout = TileLayout.of(List.of(tiled));

        assertTrue(layout.tiled(tiled));
        assertFalse(layout.tiled(sameShape));
        assertFalse(layout.tiled(null));
        assertSame(TileLayout.NONE, TileLayout.of(List.of()));
    }
}
