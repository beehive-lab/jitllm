package org.beehive.jitllm.tokenizer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * The byte-level BPE tokenizers, id for id against llama.cpp on the same text.
 *
 * <p>These vocabularies are trained on text that a pre-tokenizer regex first splits into chunks —
 * on the raw text — and only then maps to the byte-level alphabet ({@code Ġ} for a space) and
 * merges. Mapping first and splitting after turns every space into a letter, so the regex's
 * whitespace rules never fire: {@code " ="} came out as {@code " "}, {@code "="} instead of one
 * token, and an indented {@code " int"} as a four-space token and a bare {@code "int"}. The text
 * still decodes the same, which is why nothing caught it, but the model reads a sequence it was not
 * trained on, and a code-heavy prompt took about 6% more tokens.
 *
 * <p>{@code sample.txt} is dense in what the split decides: indentation, spaces before punctuation
 * and operators, runs of spaces and a tab, digits, contractions, blank lines and non-Latin text.
 * The expected ids are recorded by {@code record_llama_cpp_ids.sh}.
 */
public class ByteLevelPreTokenizationAccelTest {

    private static final Path ROOT = Path.of("src/test/resources/tokenizer/byte-level");

    @Test
    public void llama32TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.LLAMA_3_2_1B_Q8_0, "llama-3.2");
    }

    @Test
    public void qwen25TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN2_5_0_5B_Q8_0, "qwen2.5");
    }

    @Test
    public void qwen3TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN3_0_6B_Q8_0, "qwen3");
    }

    @Test
    public void qwen38TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN3_8_27B_Q4_0, "qwen3.8");
    }

    @Test
    public void granite32TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.GRANITE_3_2_2B_Q8_0, "granite-3.2");
    }

    private static void assertTokenizesAsLlamaCpp(Fixture fixture, String name) throws IOException {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent", false);
        }
        // The vocabulary and merges only: no weights
        Model model = ModelLoader.loadModel(modelPath, 512, false, false);
        String text = Files.readString(ROOT.resolve("sample.txt"));

        List<Integer> expected = ids(ROOT.resolve(name + ".llama-cpp-ids.txt"));
        List<Integer> actual = model.tokenizer().encodeAsList(text);

        assertEquals(name + ": the ids llama.cpp produces", expected, actual);
        assertEquals(name + ": decodes to the text", text, model.tokenizer().decode(actual));
    }

    private static List<Integer> ids(Path file) throws IOException {
        String ids = Files.readString(file).strip();
        return Arrays.stream(ids.substring(1, ids.length() - 1).split(","))
                .map(String::strip)
                .map(Integer::valueOf)
                .toList();
    }
}
