package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Map;
import org.junit.Test;

/**
 * Q4_0 files on the GPU path, Class A — no model file and no device.
 *
 * <p>A Q4_0 file could not be loaded at all because {@code getModelQuantization} had no case for
 * {@code general.file_type} 2. Its tensors now stay Q4_0 on the device, read by Q4_0 kernels.
 */
public class Q4_0GpuLoadTest {

    /**
     * {@code general.file_type} 2 is Q4_0. Before this change it fell through to {@code
     * UnsupportedOperationException}, so a Q4_0 file could not be loaded on either path — including
     * the CPU one, which has decoded Q4_0 all along.
     */
    @Test
    public void aQ4_0FileTypeIsRecognized() {
        assertEquals(
                "Q8_0", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 2)));
        assertEquals(
                "FP16", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 1)));
        assertEquals(
                "Q8_0", AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 7)));
    }

    @Test
    public void anUnknownFileTypeStillFailsLoudly() {
        try {
            AbstractModelLoader.getModelQuantization(Map.of("general.file_type", 99));
            org.junit.Assert.fail("an unknown quantization must not be guessed at");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("99"));
        }
    }
}
