package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Set;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.memory.WeightFootprint;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The memory preflight predicts a retained representation at its own size, not at Q8_0's.
 *
 * <p>This is the half of Q4_0 device residency that is not about arithmetic. The kernels being
 * right makes a model correct; the footprint being right makes it <b>loadable</b> — a preflight
 * that predicts every 4-bit weight at 8.5 bits per weight refuses a configuration that would have
 * run, and a refusal is not something the caller can overrule.
 *
 * <p>Reads descriptors only, so it is fast and touches no tensor data. It skips with a named reason
 * when the fixture is absent, per the Class B rule: a missing fixture must never pass.
 */
public class RetainedWeightFootprintTest {

    @Test
    public void aRetainedQ4_0FileIsPredictedAtItsOwnSize() throws Exception {
        Path model = GoldenFixture.locate(GoldenFixture.Fixture.LLAMA_3_2_1B_Q4_0);
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.LLAMA_3_2_1B_Q4_0),
                model != null);

        WeightFootprint materialized = ModelLoader.weightFootprint(model, Set.of());
        WeightFootprint retained = ModelLoader.weightFootprint(model, Set.of(DataType.Q4_0));

        // Q4_0 is 18 bytes per 32 weights, Q8_0 is 34: the per-layer weights should shrink by
        // almost exactly 34/18, since every blk.* weight in this file is Q4_0 or F32.
        double ratio = materialized.perLayerBytes() / (double) retained.perLayerBytes();
        assertTrue(
                "expected roughly the 34/18 block-size ratio, got " + ratio,
                ratio > 1.7 && ratio < 1.9);

        // token_embd is Q6_K, which no kernel here reads, so the global weights are unchanged.
        assertEquals(
                "a representation nothing retains must be predicted the same either way",
                materialized.globalBytes(),
                retained.globalBytes());

        // And a type the family does not retain changes nothing.
        WeightFootprint irrelevant = ModelLoader.weightFootprint(model, Set.of(DataType.Q4_K));
        assertEquals(materialized.perLayerBytes(), irrelevant.perLayerBytes());
    }
}
