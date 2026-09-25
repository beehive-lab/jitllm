package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.GraniteChatFormat;
import org.beehive.jitllm.model.format.LlamaChatFormat;
import org.beehive.jitllm.model.format.Qwen3ChatFormat;
import org.beehive.jitllm.server.Json;
import org.beehive.jitllm.tokenizer.LlamaTokenizer;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.TestVocabularies;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.junit.Test;

/**
 * Tool conversations through the facade's encoder, against each family's real chat template.
 *
 * <p>{@code src/test/resources/chat-templates/<family>/template.jinja} is the template embedded in
 * that family's GGUF; {@code <scenario>.txt} next to it is that template rendered by {@code
 * render_expected.py} for the conversations in {@code scenarios.json}. The encoder is fed the same
 * conversations, and its tokens — over a synthetic vocabulary that decodes to exactly what was
 * encoded — must spell the template's prompt.
 */
public class ToolTemplateConformanceTest {

    private static final Path ROOT = Path.of("src/test/resources/chat-templates");

    @Test
    public void granite32MatchesItsTemplate() throws IOException {
        Tokenizer tokenizer = TestVocabularies.granite(TestVocabularies.GRANITE_3_2_MARKERS);
        GraniteChatFormat format = new GraniteChatFormat(tokenizer, template("granite-3.2"));
        assertEquals(GraniteChatFormat.ToolDialect.GRANITE_3_2, format.toolDialect());
        assertConforms("granite-3.2", tokenizer, format, 1);
    }

    @Test
    public void granite4MatchesItsTemplate() throws IOException {
        Tokenizer tokenizer = TestVocabularies.granite(TestVocabularies.GRANITE_4_MARKERS);
        GraniteChatFormat format = new GraniteChatFormat(tokenizer, template("granite-4.0"));
        assertEquals(GraniteChatFormat.ToolDialect.GRANITE_4, format.toolDialect());
        assertConforms("granite-4.0", tokenizer, format, 1);
    }

    /** A template that is neither dialect, or none at all, means no tool calling. */
    @Test
    public void graniteWithoutARecognisedTemplateHasNoTools() throws IOException {
        Tokenizer tokenizer = TestVocabularies.granite(TestVocabularies.GRANITE_4_MARKERS);
        assertTrue(!new GraniteChatFormat(tokenizer).supportsToolCalling());
        assertTrue(!new GraniteChatFormat(tokenizer, "{{ messages }}").supportsToolCalling());
        // the 4.0 template over a vocabulary without its markers
        Tokenizer bare = TestVocabularies.granite(List.of());
        assertTrue(!new GraniteChatFormat(bare, template("granite-4.0")).supportsToolCalling());
        assertTrue(!new GraniteChatFormat(bare, template("granite-3.2")).supportsToolCalling());
    }

    /**
     * Markers from the template are single tokens, as llama.cpp and the reference tokenizer read
     * them.
     */
    @Test
    public void granite4TemplateMarkersAreSingleTokens() throws IOException {
        Tokenizer tokenizer = TestVocabularies.granite(TestVocabularies.GRANITE_4_MARKERS);
        GraniteChatFormat format = new GraniteChatFormat(tokenizer, template("granite-4.0"));
        List<Integer> tokens =
                encoder(format)
                        .encode(
                                messages(scenario("call_and_result")),
                                tools(scenario("call_and_result")));
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        // the system text's <tools></tools>, <tools>, </tools>
        assertEquals(2, java.util.Collections.frequency(tokens, special.get("<tools>")));
        // the instructions' two, and the replayed call's
        assertEquals(3, java.util.Collections.frequency(tokens, special.get("<tool_call>")));
        assertEquals(1, java.util.Collections.frequency(tokens, special.get("<tool_response>")));
    }

    /**
     * Qwen 3: tools after the system text, calls as JSON in {@code <tool_call>}, a run of results
     * in one user turn.
     */
    @Test
    public void qwen3MatchesItsTemplate() throws IOException {
        Qwen3Tokenizer tokenizer = TestVocabularies.qwen(TestVocabularies.QWEN_MARKERS);
        Qwen3ChatFormat format =
                new Qwen3ChatFormat(tokenizer, QWEN_CHAT_TOKENS)
                        .withChatTemplate(template("qwen3"));
        assertConforms("qwen3", tokenizer, format, false, 0, text -> text);
    }

    /** Qwen 2.5: as Qwen 3, plus the template's default system message when there is none. */
    @Test
    public void qwen25MatchesItsTemplate() throws IOException {
        Qwen3Tokenizer tokenizer = TestVocabularies.qwen(TestVocabularies.QWEN_MARKERS);
        Qwen3ChatFormat format =
                new Qwen3ChatFormat(tokenizer, QWEN_CHAT_TOKENS)
                        .withChatTemplate(template("qwen2.5"));
        assertConforms("qwen2.5", tokenizer, format, false, 0, text -> text);
    }

    /** The tool markers the Qwen templates spell are the vocabulary's single tokens. */
    @Test
    public void qwenTemplateMarkersAreSingleTokens() throws IOException {
        Qwen3Tokenizer tokenizer = TestVocabularies.qwen(TestVocabularies.QWEN_MARKERS);
        Qwen3ChatFormat format =
                new Qwen3ChatFormat(tokenizer, QWEN_CHAT_TOKENS)
                        .withChatTemplate(template("qwen3"));
        List<Integer> tokens =
                encoder(format, false)
                        .encode(
                                messages(scenario("call_and_result")),
                                tools(scenario("call_and_result")));
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        // the instructions' two, and the replayed call's
        assertEquals(3, java.util.Collections.frequency(tokens, special.get("<tool_call>")));
        assertEquals(1, java.util.Collections.frequency(tokens, special.get("<tool_response>")));
    }

    /**
     * Llama 3.2 (and 3.1, whose official template has the same custom-tool branch): the environment
     * lines in the system turn, the tools as {@code tojson(indent=4)} in the first user message, a
     * call as {@code {"name", "parameters"}}, a result in the {@code ipython} role through {@code
     * tojson}.
     *
     * <p>One pre-existing difference is normalised, and is not about tools: this engine writes a
     * single newline after {@code <|end_header_id|>} for every Llama turn, where the template
     * writes two. The template rejects two calls in one turn, so that scenario is not rendered.
     */
    @Test
    public void llama32MatchesItsTemplateButForTheHeaderNewline() throws IOException {
        LlamaTokenizer tokenizer = TestVocabularies.llama();
        assertConforms(
                "llama-3.2",
                tokenizer,
                new LlamaChatFormat(tokenizer),
                true,
                0,
                text -> text.replace("<|end_header_id|>\n\n", "<|end_header_id|>\n"));
    }

    // ---- harness --------------------------------------------------------------------------------

    private static final ChatFormat.ChatTokens QWEN_CHAT_TOKENS =
            new ChatFormat.ChatTokens("<|im_start|>", "<|im_end|>", "", "<|endoftext|>", "");

    private static void assertConforms(
            String family, Tokenizer tokenizer, ChatFormat format, int leadingTokensToSkip)
            throws IOException {
        assertConforms(family, tokenizer, format, true, leadingTokensToSkip, text -> text);
    }

    /**
     * @param beginOfText whether the family's model opens with the begin-of-text token
     * @param leadingTokensToSkip engine tokens ahead of what the template renders (Granite's
     *     begin-of-text, which its template does not write — pre-existing and not about tools)
     * @param normalise applied to the template's text before the comparison, for a stated
     *     pre-existing difference
     */
    private static void assertConforms(
            String family,
            Tokenizer tokenizer,
            ChatFormat format,
            boolean beginOfText,
            int leadingTokensToSkip,
            java.util.function.UnaryOperator<String> normalise)
            throws IOException {
        assertTrue(family + " reports tool calling", format.supportsToolCalling());
        ConversationEncoder encoder = encoder(format, beginOfText);
        int compared = 0;
        for (String name : scenarioNames()) {
            Path expectedFile = ROOT.resolve(family).resolve(name + ".txt");
            if (!Files.exists(expectedFile)) {
                continue; // a scenario the template rejects
            }
            Map<String, Object> scenario = scenario(name);
            List<Integer> tokens = encoder.encode(messages(scenario), tools(scenario));
            String decoded = tokenizer.decode(tokens.subList(leadingTokensToSkip, tokens.size()));
            String expected = normalise.apply(Files.readString(expectedFile));
            assertEquals(family + " / " + name, expected, decoded);
            compared++;
        }
        assertTrue(family + ": nothing compared", compared >= 3);
    }

    private static ConversationEncoder encoder(ChatFormat format) {
        return encoder(format, true);
    }

    private static ConversationEncoder encoder(ChatFormat format, boolean beginOfText) {
        return new ConversationEncoder(
                new Gemma4ToolConversationTest.FormatOnlyModel(format, beginOfText),
                ThinkingMode.DEFAULT);
    }

    private static String template(String family) throws IOException {
        return Files.readString(ROOT.resolve(family).resolve("template.jinja"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> data() throws IOException {
        return Json.parseObject(Files.readString(ROOT.resolve("scenarios.json")));
    }

    @SuppressWarnings("unchecked")
    private static List<String> scenarioNames() throws IOException {
        return new ArrayList<>(((Map<String, Object>) data().get("scenarios")).keySet());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> scenario(String name) throws IOException {
        Map<String, Object> scenario =
                (Map<String, Object>) ((Map<String, Object>) data().get("scenarios")).get(name);
        return scenario;
    }

    @SuppressWarnings("unchecked")
    private static List<ToolSpec> tools(Map<String, Object> scenario) throws IOException {
        Map<String, Object> specs = (Map<String, Object>) data().get("tools");
        List<ToolSpec> tools = new ArrayList<>();
        for (Object name : (List<Object>) scenario.get("tools")) {
            Map<String, Object> spec = (Map<String, Object>) specs.get(name);
            tools.add(
                    new ToolSpec(
                            (String) name,
                            (String) spec.get("description"),
                            Json.write(spec.get("parameters"))));
        }
        return tools;
    }

    @SuppressWarnings("unchecked")
    static List<ChatMessage> messages(Map<String, Object> scenario) {
        List<ChatMessage> messages = new ArrayList<>();
        for (Object item : (List<Object>) scenario.get("messages")) {
            Map<String, Object> m = (Map<String, Object>) item;
            String role = (String) m.get("role");
            switch (role) {
                case "system" ->
                        messages.add(ChatMessage.of(ChatRole.SYSTEM, (String) m.get("content")));
                case "user" ->
                        messages.add(ChatMessage.of(ChatRole.USER, (String) m.get("content")));
                case "assistant" -> {
                    if (m.containsKey("tool_calls")) {
                        List<ChatContent> calls = new ArrayList<>();
                        for (Object c : (List<Object>) m.get("tool_calls")) {
                            Map<String, Object> call = (Map<String, Object>) c;
                            calls.add(
                                    new ChatContent.ToolCall(
                                            (String) call.get("id"),
                                            (String) call.get("name"),
                                            Json.write(call.get("arguments"))));
                        }
                        messages.add(new ChatMessage(ChatRole.ASSISTANT, calls));
                    } else {
                        messages.add(ChatMessage.of(ChatRole.ASSISTANT, (String) m.get("content")));
                    }
                }
                case "tool" ->
                        messages.add(
                                new ChatMessage(
                                        ChatRole.TOOL,
                                        List.of(
                                                new ChatContent.ToolResult(
                                                        (String) m.get("tool_call_id"),
                                                        (String) m.get("name"),
                                                        (String) m.get("content")))));
                default -> throw new IllegalArgumentException(role);
            }
        }
        return messages;
    }
}
