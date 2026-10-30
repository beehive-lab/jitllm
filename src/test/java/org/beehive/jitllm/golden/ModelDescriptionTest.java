package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/**
 * What the memory preflight reads of a file it has not loaded: {@link ModelLoader#describe} gives
 * the identity and configuration a full load gives, from the metadata alone.
 *
 * <p>"Alone" is checked, not assumed: the description is taken again from a copy of the file cut
 * off where the tensor data starts, which a full load cannot read.
 *
 * <p>Skips with a named reason when a fixture is absent.
 */
public class ModelDescriptionTest {

    private static final int CONTEXT = 512;

    @Rule public final TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void qwen3IsDescribedAsItLoads() throws Exception {
        describedAsItLoads(GoldenFixture.Fixture.QWEN3_0_6B_Q8_0);
    }

    @Test
    public void llamaIsDescribedAsItLoads() throws Exception {
        describedAsItLoads(GoldenFixture.Fixture.LLAMA_3_2_1B_Q8_0);
    }

    /**
     * A family whose file declares another name than the identity its model and plan provider
     * carry: {@code qwen2moe} against {@code qwen2-moe}. The preflight looks the plan provider up
     * by the described identity.
     */
    @Test
    public void qwen2MoEIsDescribedAsItLoads() throws Exception {
        describedAsItLoads(GoldenFixture.Fixture.QWEN1_5_MOE_A2_7B_Q4_0);
    }

    private void describedAsItLoads(GoldenFixture.Fixture fixture) throws Exception {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue(GoldenFixture.absentMessage(fixture), file != null);

        Model model = ModelLoader.loadModel(file, CONTEXT, false, false);
        ModelLoader.Description description = ModelLoader.describe(file, CONTEXT);

        assertEquals(model.architectureId(), description.architecture());
        assertEquals(model.configuration(), description.configuration());
        assertEquals(
                model.supportsSharedKvStorage(),
                description.configuration().supportsSharedKvStorage());

        Path headerOnly = headerOnlyCopy(file);
        assertEquals(description, ModelLoader.describe(headerOnly, CONTEXT));
        assertThrows(
                "a full load reads the tensor data the copy does not have",
                Exception.class,
                () -> ModelLoader.loadModel(headerOnly, CONTEXT, false, false));
    }

    /** The file up to where its tensor data starts: every descriptor, and none of the data. */
    private Path headerOnlyCopy(Path file) throws Exception {
        long dataOffset = GGUF.loadGGUFMetadata(file).getTensorDataOffset();
        Path copy = folder.newFile(file.getFileName().toString()).toPath();
        try (FileChannel in = FileChannel.open(file, StandardOpenOption.READ);
                FileChannel out = FileChannel.open(copy, StandardOpenOption.WRITE)) {
            long copied = 0;
            while (copied < dataOffset) {
                copied += in.transferTo(copied, dataOffset - copied, out);
            }
        }
        return copy;
    }
}
