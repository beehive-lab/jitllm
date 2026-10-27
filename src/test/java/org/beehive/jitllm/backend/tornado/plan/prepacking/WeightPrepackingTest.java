package org.beehive.jitllm.backend.tornado.plan.prepacking;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.Gemma4PlanProvider;
import org.beehive.jitllm.backend.tornado.plan.LlamaPlanProvider;
import org.beehive.jitllm.backend.tornado.tensor.FP32TornadoTensor;
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
 * Which weights a plan prepacks: the projections of a type the family reads prepacked, whose shape
 * fits, where prepacking is on.
 *
 * <p>No device: prepacking is switched on with {@code -Djitllm.q4_0.prepack=true}, which decides in
 * place of the device check, and cleared after each test.
 */
public class WeightPrepackingTest {

    private static final ExecutionPolicy POLICY = ExecutionPolicy.builder().build();

    @After
    public void clearProperties() {
        System.clearProperty(WeightPrepacking.Q4_PROPERTY);
        System.clearProperty(WeightPrepacking.Q8_PROPERTY);
    }

    private static TornadoTensor q4(int rows, int cols) {
        return new Q4_0TornadoTensor(new ByteArray(rows * cols / 32 * 18), Shape.of(cols, rows));
    }

    private static TornadoTensor f32(int elements) {
        return new FP32TornadoTensor(new FloatArray(elements), Shape.of(elements));
    }

    /** One layer: Q4_0 projections that fit, a Q4_1 down projection, and one too narrow to fit. */
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
                q4(96, 64), // 96 rows are not a multiple of 128
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
    public void prepacksTheProjectionsTheFamilyReadsPrepackedWhereTheirShapeFits() {
        System.setProperty(WeightPrepacking.Q4_PROPERTY, "true");
        Layer l = layer();

        WeightPrepacking prepacking =
                WeightPrepacking.forPlan(new LlamaPlanProvider(), weights(l), POLICY);

        assertTrue(prepacking.prepackedQ4(l.wq()));
        assertTrue(prepacking.prepackedQ4(l.wk()));
        assertTrue(prepacking.prepackedQ4(l.wo()));
        assertTrue(prepacking.prepackedQ4(l.gate()));
        assertTrue(prepacking.prepackedQ4(l.up()));
        assertFalse("96 rows do not fit", prepacking.prepacked(l.wv()));
        assertFalse("Llama does not read Q4_1 prepacked", prepacking.prepacked(l.down()));
        assertFalse("not a projection", prepacking.prepacked(l.embeddings()));
        assertFalse(prepacking.prepackedQ8(l.wq()));
    }

    @Test
    public void prepacksNothingWherePrepackingIsOff() {
        System.setProperty(WeightPrepacking.Q4_PROPERTY, "false");

        assertSame(
                WeightPrepacking.NONE,
                WeightPrepacking.forPlan(new LlamaPlanProvider(), weights(layer()), POLICY));
    }

    @Test
    public void prepacksNothingOfATypeTheFamilyDoesNotReadPrepacked() {
        System.setProperty(WeightPrepacking.Q8_PROPERTY, "true");
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

        WeightPrepacking prepacking =
                WeightPrepacking.forPlan(new LlamaPlanProvider(), withQ8, POLICY);

        assertFalse("Llama reads only Q4_0 prepacked", prepacking.prepacked(q8));
    }

    @Test
    public void aFamilyThatReadsNothingPrepackedUnderAPolicyPrepacksNothing() {
        System.setProperty(WeightPrepacking.Q4_PROPERTY, "true");
        ExecutionPolicy nativeLibraries = ExecutionPolicy.builder().nativeLibraries(true).build();

        assertEquals(
                "Gemma 4's cuBLAS projections read nothing prepacked",
                Set.of(),
                WeightPrepacking.prepackedTypes(new Gemma4PlanProvider(), nativeLibraries));
        assertEquals(
                Set.of(DataType.Q4_0),
                WeightPrepacking.prepackedTypes(new Gemma4PlanProvider(), POLICY));
    }

    private static ExecutionPolicy batched(int width) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(width)
                .build();
    }

    @Test
    public void onlyABatchedPrefillWidthTheInt8GemmDoesNotCoverDoesNotFit() {
        assertTrue(WeightPrepacking.prefillWidthFits(POLICY));
        assertTrue(WeightPrepacking.prefillWidthFits(batched(1)));
        assertTrue(WeightPrepacking.prefillWidthFits(batched(384)));
        assertFalse(WeightPrepacking.prefillWidthFits(batched(64)));
    }

    @Test
    public void aPrepackingComparesTensorsByIdentity() {
        TornadoTensor prepacked = q4(128, 64);
        TornadoTensor sameShape = q4(128, 64);

        WeightPrepacking prepacking = WeightPrepacking.of(List.of(prepacked));

        assertTrue(prepacking.prepacked(prepacked));
        assertFalse(prepacking.prepacked(sameShape));
        assertFalse(prepacking.prepacked(null));
        assertSame(WeightPrepacking.NONE, WeightPrepacking.of(List.of()));
    }
}
