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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

/** A file is parsed once while it is unchanged, and again once it changes. */
public class GGUFParsedOnceTest {

    /** A GGUF file with no tensors and one string key. */
    private static byte[] gguf(String architecture) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer header = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(0x46554747).putInt(3).putLong(0).putLong(1);
        out.writeBytes(header.array());
        for (String s : new String[] {"general.architecture", null, architecture}) {
            if (s == null) {
                out.writeBytes(
                        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(8).array());
                continue;
            }
            byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
            out.writeBytes(
                    ByteBuffer.allocate(8)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .putLong(bytes.length)
                            .array());
            out.writeBytes(bytes);
        }
        return out.toByteArray();
    }

    @Test
    public void anUnchangedFileIsParsedOnceAndAChangedOneAgain() throws IOException {
        Path file = Files.createTempFile("parsed-once", ".gguf");
        try {
            Files.write(file, gguf("llama"));
            GGUF first = GGUF.loadGGUFMetadata(file);
            GGUF second = GGUF.loadGGUFMetadata(file);
            assertSame(first, second);

            Files.write(file, gguf("qwen35"));
            GGUF third = GGUF.loadGGUFMetadata(file);
            assertNotSame(first, third);
            assertEquals("qwen35", third.getMetadata().get("general.architecture"));
            third.getFileChannel().close();
        } finally {
            Files.delete(file);
        }
    }
}
