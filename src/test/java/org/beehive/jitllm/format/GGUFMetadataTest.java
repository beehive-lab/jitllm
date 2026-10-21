package org.beehive.jitllm.format;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.Test;

/** Reads a small GGUF file written by hand, byte for byte as the format describes it. */
public class GGUFMetadataTest {

    private static final int STRING = 8;
    private static final int ARRAY = 9;
    private static final int UINT32 = 4;
    private static final int FLOAT32 = 6;

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    private void u32(int value) {
        out.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    private void u64(long value) {
        out.writeBytes(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array());
    }

    private void string(String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        u64(bytes.length);
        out.writeBytes(bytes);
    }

    @Test
    public void metadataTensorInfosAndTheDataOffsetAreRead() throws IOException {
        // Longer than the reader's 1 MiB buffer, so it is read in pieces.
        String longToken = "x".repeat(3 << 20) + "end";
        String[] tokens = {"<s>", "héllo", longToken, "</s>"};

        u32(0x46554747); // magic
        u32(3); // version
        u64(2); // tensor count
        u64(4); // metadata key-value count
        string("general.architecture");
        u32(STRING);
        string("llama");
        string("llama.block_count");
        u32(UINT32);
        u32(28);
        string("tokenizer.ggml.tokens");
        u32(ARRAY);
        u32(STRING);
        u64(tokens.length);
        for (String token : tokens) {
            string(token);
        }
        string("tokenizer.ggml.scores");
        u32(ARRAY);
        u32(FLOAT32);
        u64(3);
        for (float score : new float[] {-1.5f, 0f, 2.25f}) {
            u32(Float.floatToRawIntBits(score));
        }
        string("token_embd.weight");
        u32(2); // dimensions
        u64(8);
        u64(3);
        u32(0); // F32
        u64(0); // offset
        string("output_norm.weight");
        u32(1);
        u64(8);
        u32(0);
        u64(96);
        int headerBytes = out.size();
        out.writeBytes(new byte[1024]); // padding and tensor data

        Path file = Files.createTempFile("metadata", ".gguf");
        try {
            Files.write(file, out.toByteArray());
            GGUF gguf = GGUF.loadGGUFMetadata(file);
            gguf.getFileChannel().close();

            assertEquals("llama", gguf.getMetadata().get("general.architecture"));
            assertEquals(28, gguf.getMetadata().get("llama.block_count"));
            assertArrayEquals(tokens, (String[]) gguf.getMetadata().get("tokenizer.ggml.tokens"));
            assertArrayEquals(
                    new float[] {-1.5f, 0f, 2.25f},
                    (float[]) gguf.getMetadata().get("tokenizer.ggml.scores"),
                    0f);
            GGUF.GGUFTensorInfo embedding = gguf.getTensorInfos().get("token_embd.weight");
            assertEquals("[8, 3]", Arrays.toString(embedding.dimensions()));
            assertEquals(GGMLType.F32, embedding.ggmlType());
            assertEquals(96, gguf.getTensorInfos().get("output_norm.weight").offset());
            // The data starts at the first 32-byte boundary after the tensor infos.
            assertEquals((headerBytes + 31) / 32 * 32, gguf.getTensorDataOffset());
        } finally {
            Files.delete(file);
        }
    }
}
