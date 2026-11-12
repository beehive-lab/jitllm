package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Set;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The memory preflight sizes the prepack scratch from the file's descriptors: the largest
 * two-dimensional per-layer tensor of a prepacked representation whose shape fits, which is what
 * {@code WeightPrepacker} allocates the scratch for.
 *
 * <p>Reads descriptors only. Skips with a named reason when the fixture is absent.
 */
public class PrepackedProjectionFootprintTest {

    @Test
    public void theLargestPrepackedProjectionOfAQ8_0File() throws Exception {
        Path model = GoldenFixture.locate(GoldenFixture.Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_0_6B_Q8_0), model != null);

        // Qwen3-0.6B: embedding 1024, feed-forward 3072. Its largest projections are ffn_gate and
        // ffn_up (3072 rows of 1024) and ffn_down (1024 rows of 3072), all of which fit; as Q8_0,
        // 34 bytes per 32 weights.
        long largest = 3072L * 1024 / 32 * 34;
        assertEquals(largest, ModelLoader.prepackedProjectionBytes(model, Set.of(DataType.Q8_0)));

        // Nothing is prepacked when the family prepacks nothing, or only a representation the file
        // lacks.
        assertEquals(0, ModelLoader.prepackedProjectionBytes(model, Set.of()));
        assertEquals(0, ModelLoader.prepackedProjectionBytes(model, Set.of(DataType.Q4_0)));
    }
}
