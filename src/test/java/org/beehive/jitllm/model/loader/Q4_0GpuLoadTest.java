/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
