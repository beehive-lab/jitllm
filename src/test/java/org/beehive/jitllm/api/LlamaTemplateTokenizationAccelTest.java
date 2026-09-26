package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * The Llama prompts the engine encodes, token for token against llama.cpp's tokenization of the
 * same conversations rendered through the model's own chat template.
 *
 * <p>{@link ToolTemplateConformanceTest} compares text over a synthetic vocabulary; this compares
 * ids over the real one, which is where a header's two newlines must be the single {@code "\n\n"}
 * token (271) rather than two {@code "\n"} tokens (198, 198). The expected ids are {@code
 * chat-templates/<family>/<scenario>.llama-cpp-ids.txt}, recorded by {@code tokenize_llama_cpp.sh}
 * from the rendered {@code <scenario>.txt}. Llama 3.1 and 3.2 share one vocabulary, so the 3.2 1B
 * fixture serves both.
 */
public class LlamaTemplateTokenizationAccelTest {

    private static final Path ROOT = Path.of("src/test/resources/chat-templates");

    @Test
    public void llama32ToolConversationsTokenizeAsLlamaCppTokenizesTheTemplate() throws Exception {
        assertMatchesLlamaCpp(
                "llama-3.2", List.of("system_and_tool", "no_system_two_tools", "call_and_result"));
    }

    @Test
    public void llama31PlainChatTokenizesAsLlamaCppTokenizesTheTemplate() throws Exception {
        assertMatchesLlamaCpp(
                "llama-3.1", List.of("chat_user_only", "chat_system_and_user", "chat_multi_turn"));
    }

    private static void assertMatchesLlamaCpp(String family, List<String> scenarios)
            throws IOException {
        Model model = loadOrSkip();
        ConversationEncoder encoder = new ConversationEncoder(model, ThinkingMode.DEFAULT);
        for (String name : scenarios) {
            Map<String, Object> scenario = ToolTemplateConformanceTest.scenario(name);
            List<Integer> encoded =
                    encoder.encode(
                            ToolTemplateConformanceTest.messages(scenario),
                            ToolTemplateConformanceTest.tools(scenario));
            assertEquals(family + " / " + name, expected(family, name), encoded);
        }
    }

    private static List<Integer> expected(String family, String name) throws IOException {
        String ids =
                Files.readString(ROOT.resolve(family).resolve(name + ".llama-cpp-ids.txt")).strip();
        return Arrays.stream(ids.substring(1, ids.length() - 1).split(","))
                .map(String::strip)
                .map(Integer::valueOf)
                .toList();
    }

    private static Model loadOrSkip() {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent", false);
        }
        try {
            return ModelLoader.loadModel(modelPath, 512, true, false);
        } catch (Exception e) {
            throw new AssertionError("the fixture failed to load", e);
        }
    }
}
