package org.beehive.jitllm.golden;

import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.WRITE;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Weights read in place from the mapped file ({@link UnifiedMemoryWeights}) give the logits of
 * weights copied to the device, bit for bit: the same JIT kernels read the same bytes. A prompt
 * that crosses several 128-token chunks, then a decode that reads what prefill wrote.
 */
public class UnifiedMemoryWeightsAccelTest {

    private static final int CONTEXT = 1024;
    private static final int PROMPT_TOKENS = 300;
    private static final int DECODE_STEPS = 16;

    /** A tensor mapped from a file is read in place; an array on the heap is uploaded. */
    @Test
    public void onlyMappedWeightsAreReadInPlace() throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = Files.createTempFile("weights", ".bin");
        try {
            Files.write(file, new byte[1 << 16]);
            ByteArray mapped;
            try (FileChannel channel = FileChannel.open(file, READ, WRITE)) {
                mapped =
                        ByteArray.fromSegmentShallow(
                                channel.map(
                                        FileChannel.MapMode.PRIVATE, 0, 1 << 16, Arena.ofAuto()));
            }
            FloatArray heap = new FloatArray(1024);
            TornadoWorkspace ws = new TornadoWorkspace();
            ws.unifiedMemory = true;
            Object[] others =
                    UnifiedMemoryWeights.consume(new TaskGraph("weights"), ws, mapped, heap);
            assertArrayEquals(new Object[] {heap}, others);
            ws.unifiedMemory = false;
            others = UnifiedMemoryWeights.consume(new TaskGraph("copies"), ws, mapped, heap);
            assertArrayEquals(new Object[] {mapped, heap}, others);
        } finally {
            Files.delete(file);
        }
    }

    @Test
    public void qwen3Q8_0SingleToken() throws Exception {
        check(Fixture.QWEN3_0_6B_Q8_0, 1);
    }

    @Test
    public void qwen3Q8_0Batched() throws Exception {
        check(Fixture.QWEN3_0_6B_Q8_0, 128);
    }

    @Test
    public void qwen3F16Batched() throws Exception {
        check(Fixture.QWEN3_0_6B_F16, 128);
    }

    @Test
    public void llamaQ8_0Batched() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0, 128);
    }

    private static void check(Fixture fixture, int batch) throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        String[] keys = {
            "use.tornadovm",
            "jitllm.withPrefillDecode",
            "jitllm.prefillBatchSize",
            "jitllm.nativeLibraries",
            UnifiedMemoryWeights.PROPERTY
        };
        String[] previous = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            previous[i] = System.getProperty(keys[i]);
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", Boolean.toString(batch > 1));
        System.setProperty("jitllm.prefillBatchSize", Integer.toString(batch));
        // In place, MLX is not used (UnifiedMemoryWeights); the copies run the same JIT kernels.
        System.setProperty("jitllm.nativeLibraries", "false");
        try {
            Model model = KvPrecisionHarness.load(file, CONTEXT, true);
            List<Integer> prompt = KvPrecisionHarness.longPrompt(model, PROMPT_TOKENS);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "false");
            var copied =
                    KvPrecisionHarness.run(
                            model, true, DataType.F16, batch, prompt, DECODE_STEPS, null);
            System.setProperty(UnifiedMemoryWeights.PROPERTY, "true");
            var inPlace =
                    KvPrecisionHarness.run(
                            model, true, DataType.F16, batch, prompt, DECODE_STEPS, null);
            assertEquals(copied.executionMode(), inPlace.executionMode());
            assertArrayEquals(fixture + " tokens", copied.tokens(), inPlace.tokens());
            for (int r = 0; r < copied.rows().size(); r++) {
                assertArrayEquals(
                        fixture + " logits row " + r,
                        copied.rows().get(r),
                        inPlace.rows().get(r),
                        0f);
            }
        } finally {
            for (int i = 0; i < keys.length; i++) {
                if (previous[i] == null) {
                    System.clearProperty(keys[i]);
                } else {
                    System.setProperty(keys[i], previous[i]);
                }
            }
        }
    }
}
