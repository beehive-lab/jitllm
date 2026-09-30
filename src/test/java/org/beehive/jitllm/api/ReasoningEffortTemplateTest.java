package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Qwen35ChatFormat;
import org.beehive.jitllm.tokenizer.Qwen35Tokenizer;
import org.beehive.jitllm.tokenizer.TestVocabularies;
import org.junit.Test;

/**
 * Qwen3.8's reasoning effort, through the facade's encoder, against the family's real template.
 *
 * <p>{@code chat-templates/qwen3.8/reasoning/<scenario>.<variant>.txt} is the template embedded in
 * the Qwen3.8 GGUF rendered by {@code render_expected.py} for each variant: the template's default
 * ({@code reasoning_effort} undefined, which it resolves to {@code xhigh}), each explicit effort,
 * and thinking disabled. The instruction the template writes at the top of the system turn — or in
 * a system turn of its own when the conversation has none, or ahead of the tools block — must be
 * the engine's too.
 *
 * <p>The format's two stated deviations are normalised: with thinking on, the template's generation
 * prompt opens the thinking block ({@code <think>\n}), which the engine lets the model write
 * because the integrations recognise the reasoning by it; and the template writes an empty {@code
 * <think>} block in front of every replayed assistant turn, which the engine does not.
 */
public class ReasoningEffortTemplateTest {

    private static final Path ROOT = Path.of("src/test/resources/chat-templates/qwen3.8/reasoning");

    private static final List<String> SCENARIOS =
            List.of(
                    "chat_user_only",
                    "chat_system_and_user",
                    "chat_multi_turn",
                    "system_and_tool",
                    "no_system_two_tools");

    @Test
    public void theTemplateDefaultIsXhigh() throws IOException {
        assertMatchesTemplate("default", ThinkingMode.DEFAULT, ReasoningEffort.DEFAULT);
    }

    @Test
    public void thinkingEnabledTakesTheTemplateDefaultToo() throws IOException {
        assertMatchesTemplate("default", ThinkingMode.ENABLED, ReasoningEffort.DEFAULT);
    }

    @Test
    public void xhigh() throws IOException {
        assertMatchesTemplate("xhigh", ThinkingMode.DEFAULT, ReasoningEffort.XHIGH);
    }

    @Test
    public void mediumWritesNoInstruction() throws IOException {
        assertMatchesTemplate("medium", ThinkingMode.DEFAULT, ReasoningEffort.MEDIUM);
    }

    @Test
    public void low() throws IOException {
        assertMatchesTemplate("low", ThinkingMode.DEFAULT, ReasoningEffort.LOW);
    }

    @Test
    public void withThinkingDisabledThereIsNoInstruction() throws IOException {
        assertMatchesTemplate("thinking-disabled", ThinkingMode.DISABLED, ReasoningEffort.DEFAULT);
    }

    @Test
    public void onlyQwen38ReportsTheControl() {
        assertEquals(
                new ModelCapabilities(true, true, true), DelegatingModel.capabilitiesOf(qwen38()));
        assertFalse(
                DelegatingModel.capabilitiesOf(
                                new org.beehive.jitllm.model.format.Qwen3ChatFormat(
                                        TestVocabularies.qwen(TestVocabularies.QWEN_MARKERS),
                                        QWEN_CHAT_TOKENS))
                        .reasoningEffortControl());
    }

    private static void assertMatchesTemplate(
            String variant, ThinkingMode thinking, ReasoningEffort effort) throws IOException {
        Qwen35Tokenizer tokenizer = TestVocabularies.qwen35(TestVocabularies.QWEN_MARKERS);
        Qwen35ChatFormat format = new Qwen35ChatFormat(tokenizer, QWEN_CHAT_TOKENS);
        ConversationEncoder encoder =
                new ConversationEncoder(
                        new Gemma4ToolConversationTest.FormatOnlyModel(format, false),
                        thinking,
                        effort);
        for (String name : SCENARIOS) {
            Map<String, Object> scenario = ToolTemplateConformanceTest.scenario(name);
            List<Integer> tokens =
                    encoder.encode(
                            ToolTemplateConformanceTest.messages(scenario),
                            ToolTemplateConformanceTest.tools(scenario));
            String expected =
                    Files.readString(ROOT.resolve(name + "." + variant + ".txt"))
                            // an empty block with text after it: a replayed assistant turn
                            .replaceAll("<think>\n\n</think>\n\n(?=.)", "");
            if (thinking != ThinkingMode.DISABLED) {
                assertTrue(
                        name + ": the template opens the thinking block",
                        expected.endsWith("<think>\n"));
                expected = expected.substring(0, expected.length() - "<think>\n".length());
            }
            assertEquals(variant + " / " + name, expected, tokenizer.decode(tokens));
        }
    }

    private static final ChatFormat.ChatTokens QWEN_CHAT_TOKENS =
            new ChatFormat.ChatTokens("<|im_start|>", "<|im_end|>", "", "<|endoftext|>", "");

    private static Qwen35ChatFormat qwen38() {
        return new Qwen35ChatFormat(
                TestVocabularies.qwen35(TestVocabularies.QWEN_MARKERS), QWEN_CHAT_TOKENS);
    }
}
