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
