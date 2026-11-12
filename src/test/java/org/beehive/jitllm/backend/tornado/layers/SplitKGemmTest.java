package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import uk.ac.manchester.tornado.api.KernelContext;

/**
 * The narrow family's split-K GEMMs split each shape as the primary split it, while their own rows
 * bound what a split may write.
 *
 * <p>Example: a projection into a model 2,048 wide over 2,048 inputs has 64 output tiles at 512
 * rows and splits in two; at 128 rows it has 16, and a plan that narrow splits it in four.
 */
public class SplitKGemmTest {

    private static final int DIM = 2048;

    @Test
    public void aNarrowedGemmSplitsAsItsPrimary() {
        SplitKGemm primary = new SplitKGemm(new KernelContext(), 512, DIM, false);
        assertEquals(2, primary.slices(512, DIM, DIM));
        SplitKGemm narrow = primary.narrowed(128, DIM);
        assertEquals(4, new SplitKGemm(new KernelContext(), 128, DIM, false).slices(128, DIM, DIM));
        assertEquals(2, narrow.slices(128, DIM, DIM));
    }

    @Test
    public void aNarrowedGemmKeepsWholeAShapeItsPrimaryKeptWhole() {
        SplitKGemm primary = new SplitKGemm(new KernelContext(), 512, 4 * DIM, false);
        // 256 tiles at 512 rows: whole; 64 at 128 rows, which a plan that narrow splits in two.
        assertEquals(1, primary.slices(512, 4 * DIM, DIM));
        SplitKGemm narrow = primary.narrowed(128, 4 * DIM);
        assertEquals(1, narrow.slices(128, 4 * DIM, DIM));
        // A shape the primary never multiplied stays whole too.
        assertEquals(1, narrow.slices(128, DIM / 2, DIM));
    }

    @Test
    public void aNarrowedGemmSplitsOnlyWithinItsOwnRows() {
        SplitKGemm primary = new SplitKGemm(new KernelContext(), 512, DIM, false);
        assertEquals(2, primary.slices(512, DIM, DIM));
        SplitKGemm narrow = primary.narrowed(128, DIM);
        // 256 rows need twice the partial products 128 rows may write.
        assertEquals(1, narrow.slices(256, DIM, DIM));
        // Rows that are not whole 128-row tiles never split.
        assertEquals(1, narrow.slices(120, DIM, DIM));
    }
}
