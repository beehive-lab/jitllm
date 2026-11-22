package org.beehive.jitllm.backend.tornado.ops.nativelib.apple;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.junit.Test;

/**
 * MLX's library tasks run on M4-class GPUs and newer; TornadoVM refuses older ones at the first
 * dispatch, so jitllm has to know before it picks the MLX prefill.
 */
public class MlxPrefillDeviceTest {

    @Test
    public void m1ToM3AreTooOldForMlx() {
        for (int generation = 1;
                generation < MlxPrefill.MIN_APPLE_SILICON_GENERATION;
                generation++) {
            Optional<String> reason = MlxPrefill.deviceTooOld(generation);
            assertTrue("M" + generation, reason.isPresent());
            assertTrue(reason.get(), reason.get().contains("M" + generation));
        }
    }

    @Test
    public void m4AndNewerRunMlx() {
        assertEquals(Optional.empty(), MlxPrefill.deviceTooOld(4));
        assertEquals(Optional.empty(), MlxPrefill.deviceTooOld(5));
    }

    /** An unreadable generation keeps the old behaviour: the dispatch decides. */
    @Test
    public void anUnknownGenerationIsNotRefused() {
        assertEquals(Optional.empty(), MlxPrefill.deviceTooOld(-1));
    }
}
