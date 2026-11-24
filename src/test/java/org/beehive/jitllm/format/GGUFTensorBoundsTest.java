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
package org.beehive.jitllm.format;

import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.WRITE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.beehive.jitllm.format.GGUF.GGUFTensorInfo;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceId;
import org.junit.Test;

/** Mapping a tensor for a device never writes the model file. */
public class GGUFTensorBoundsTest {

    private static final Device DEVICE =
            new Device() {
                @Override
                public DeviceId id() {
                    return DeviceId.of(BackendId.METAL, "0");
                }

                @Override
                public DeviceCapabilities capabilities() {
                    return DeviceCapabilities.of();
                }

                @Override
                public String displayName() {
                    return "test";
                }

                @Override
                public long nativeArrayHeaderBytes() {
                    return 16;
                }
            };

    @Test
    public void aTypeWithNoLayoutIsRefusedAndTheFileKeepsItsSize() throws IOException {
        assertRefused(
                new GGUFTensorInfo("blk.0.attn_q.weight", new int[] {256, 4}, GGMLType.IQ2_XXS, 0));
    }

    @Test
    public void aTensorPastTheEndIsRefusedAndTheFileKeepsItsSize() throws IOException {
        // 8 Q8_0 blocks of 34 bytes, in a file with 64 bytes of tensor data.
        assertRefused(new GGUFTensorInfo("blk.0.attn_q.weight", new int[] {256}, GGMLType.Q8_0, 0));
    }

    private static void assertRefused(GGUFTensorInfo info) throws IOException {
        Path file = Files.createTempFile("bounds", ".gguf");
        try {
            Files.write(file, new byte[128]);
            try (FileChannel channel = FileChannel.open(file, READ, WRITE)) {
                Exception e =
                        assertThrows(
                                Exception.class,
                                () ->
                                        GGUF.loadTensorsTornado(
                                                channel, 64, Map.of(info.name(), info), DEVICE));
                assertTrue(e.getMessage(), e.getMessage().contains(info.name()));
            }
            assertEquals(128, Files.size(file));
        } finally {
            Files.delete(file);
        }
    }
}
