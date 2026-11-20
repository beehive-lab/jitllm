package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.beehive.jitllm.tokenizer.Lfm2Tokenizer;

/**
 * The chat format for the LFM2 family.
 *
 * <p>The turn structure is ChatML, as in the chat template the LFM2.5 GGUF files carry: {@code
 * <|im_start|>role\n…<|im_end|>\n}, so everything about turns is inherited from {@link
 * Qwen3ChatFormat}. Three things are LFM2's own:
 *
 * <ul>
 *   <li><b>A beginning-of-text token.</b> The template opens with {@code bos_token}, {@code
 *       <|startoftext|>}; Qwen has none.
 *   <li><b>The reasoning block is opened by the prompt.</b> LFM2.5's template's generation prompt
 *       is {@code <|im_start|>assistant\n<think>}, so the default turn and a turn that asks for
 *       thinking both write {@code <think>} after the header; LFM2's template has none, and its
 *       default turn writes nothing.
 *   <li><b>Its sampling defaults</b>, from {@code general.sampling.temp} in the file.
 * </ul>
 *
 * <h2>Deviations, stated</h2>
 *
 * <p>The template has no switch that turns reasoning off. Asking for it off writes the closed
 * {@code <think></think>} block Qwen 3's template uses for the same purpose, which the model was
 * not trained against and may ignore.
 *
 * <h2>Tool calling</h2>
 *
 * <p>Both templates put the tools in the system turn as {@code List of tools: [...]}, one {@code
 * tojson} per tool, and the model calls them as a Python-style list between {@code
 * <|tool_call_start|>} and {@code <|tool_call_end|>} ({@link Lfm2ToolCalls}). Results are turns of
 * the {@code tool} role. The two templates differ, and the file's own decides ({@link
 * ToolDialect}):
 *
 * <ul>
 *   <li>{@link ToolDialect#LFM2} (LFM2-8B-A1B): the list between {@code <|tool_list_start|>} and
 *       {@code <|tool_list_end|>}, each result between {@code <|tool_response_start|>} and {@code
 *       <|tool_response_end|>}. Its template replays an assistant message's content only, so an
 *       earlier call is written as content, in the model's output syntax.
 *   <li>{@link ToolDialect#LFM2_5} (LFM2.5-24B-A2B): the bare list and bare results; earlier calls
 *       replayed from {@code tool_calls}, as here.
 * </ul>
 */
public class Lfm2ChatFormat extends Qwen3ChatFormat {

    private final int startOfText;

    /** Whether the file's template opens the assistant's turn with {@code <think>}. */
    private final boolean templateOpensThinking;

    /** Which tool format the file's template writes, if the vocabulary has its markers. */
    private final ToolDialect toolDialect;

    /** The two LFM2 templates' tool formats; see the class note. */
    public enum ToolDialect {
        /** No tool calling: the template writes no tools, or the vocabulary lacks the markers. */
        NONE,
        /** LFM2-8B-A1B's template. */
        LFM2,
        /** LFM2.5-24B-A2B's template. */
        LFM2_5
    }

    /**
     * @param chatTemplate the file's {@code tokenizer.chat_template}, or {@code null}; it decides
     *     whether the default turn opens a reasoning block
     */
    public Lfm2ChatFormat(Lfm2Tokenizer tokenizer, ChatTokens chatTokens, String chatTemplate) {
        super(tokenizer, chatTokens);
        this.startOfText = tokenizer.getSpecialTokens().getOrDefault("<|startoftext|>", -1);
        this.templateOpensThinking = opensThinking(chatTemplate);
        this.toolDialect = toolDialect(chatTemplate, tokenizer.getSpecialTokens());
    }

    /**
     * {@link ToolDialect#LFM2} when the template wraps the tools in {@code <|tool_list_start|>},
     * {@link ToolDialect#LFM2_5} when it writes them bare and renders {@code tool_calls}; each only
     * when the vocabulary holds the markers that dialect writes.
     */
    static ToolDialect toolDialect(String chatTemplate, Map<String, Integer> special) {
        if (chatTemplate == null
                || !chatTemplate.contains("List of tools: ")
                || !special.containsKey(Lfm2ToolCalls.START)
                || !special.containsKey(Lfm2ToolCalls.END)) {
            return ToolDialect.NONE;
        }
        if (chatTemplate.contains("List of tools: <|tool_list_start|>[")) {
            return special.containsKey(LIST_START)
                            && special.containsKey(LIST_END)
                            && special.containsKey(RESPONSE_START)
                            && special.containsKey(RESPONSE_END)
                    ? ToolDialect.LFM2
                    : ToolDialect.NONE;
        }
        return chatTemplate.contains("List of tools: [") && chatTemplate.contains("tool_calls")
                ? ToolDialect.LFM2_5
                : ToolDialect.NONE;
    }

    private static final String LIST_START = "<|tool_list_start|>";
    private static final String LIST_END = "<|tool_list_end|>";
    private static final String RESPONSE_START = "<|tool_response_start|>";
    private static final String RESPONSE_END = "<|tool_response_end|>";

    public ToolDialect toolDialect() {
        return toolDialect;
    }

    /**
     * Whether {@code chatTemplate}'s generation prompt ends in {@code <think>}: the literal {@code
     * <|im_start|>assistant\n<think>}, with the newline as the template spells it (an escape in the
     * Jinja string) or as a character.
     */
    static boolean opensThinking(String chatTemplate) {
        return chatTemplate != null
                && (chatTemplate.contains("<|im_start|>assistant\\n<think>")
                        || chatTemplate.contains("<|im_start|>assistant\n<think>"));
    }

    /** {@code <think>} where the file's template opens the turn with it. */
    @Override
    public List<Integer> defaultGenerationPrimer() {
        int thinkStart = tokenizer.getThinkStartToken();
        return templateOpensThinking && thinkStart != -1 ? List.of(thinkStart) : List.of();
    }

    @Override
    public int getBeginOfText() {
        return startOfText != -1 ? startOfText : super.getBeginOfText();
    }

    /** The template's {@code <think>} when reasoning is asked for; see the class note when not. */
    @Override
    public List<Integer> encodeThinkingControl(boolean enableThinking) {
        int thinkStart = tokenizer.getThinkStartToken();
        int thinkEnd = tokenizer.getThinkEndToken();
        if (thinkStart == -1 || thinkEnd == -1) {
            return super.encodeThinkingControl(enableThinking);
        }
        List<Integer> tokens = new ArrayList<>();
        tokens.add(thinkStart);
        if (!enableThinking) {
            tokens.add(thinkEnd);
        }
        return tokens;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    @Override
    public boolean supportsToolCalling() {
        return toolDialect != ToolDialect.NONE;
    }

    /**
     * {@code <|im_start|>system\n{system}\nList of tools: [{tool}, {tool}]<|im_end|>\n}, the list
     * between the list markers in {@link ToolDialect#LFM2}; without a system message, the list
     * alone.
     */
    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        List<String> tools = new ArrayList<>();
        for (Object tool : ToolJson.parseSequence(toolsJson)) {
            tools.add(ToolJson.dumps(tool));
        }
        boolean markers = toolDialect == ToolDialect.LFM2;
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        String system =
                systemContent == null || systemContent.isEmpty() ? "" : systemContent + "\n";
        tokens.addAll(tokenizer.encodeOrdinaryAsList("system\n" + system + "List of tools: "));
        if (markers) {
            tokens.add(special(LIST_START));
        }
        tokens.addAll(tokenizer.encodeOrdinaryAsList("[" + String.join(", ", tools) + "]"));
        if (markers) {
            tokens.add(special(LIST_END));
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    /**
     * {@code <|im_start|>assistant\n<|tool_call_start|>[call, call]<|tool_call_end|><|im_end|>\n}.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        tokens.add(special(Lfm2ToolCalls.START));
        tokens.addAll(tokenizer.encodeOrdinaryAsList(Lfm2ToolCalls.render(toolCalls)));
        tokens.add(special(Lfm2ToolCalls.END));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    /**
     * One {@code tool} turn per result, as the templates write them; between the response markers
     * in {@link ToolDialect#LFM2}.
     */
    @Override
    public List<Integer> encodeToolResults(List<ToolResult> results) {
        boolean markers = toolDialect == ToolDialect.LFM2;
        List<Integer> tokens = new ArrayList<>();
        for (ToolResult result : results) {
            tokens.add(imStart);
            tokens.addAll(tokenizer.encodeOrdinaryAsList("tool\n"));
            if (markers) {
                tokens.add(special(RESPONSE_START));
            }
            tokens.addAll(tokenizer.encodeOrdinaryAsList(result.content()));
            if (markers) {
                tokens.add(special(RESPONSE_END));
            }
            tokens.addAll(endOfTurn());
        }
        return tokens;
    }

    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return extractAllToolCalls(responseText).stream().findFirst();
    }

    /**
     * Every call in the response's {@code <|tool_call_start|>} blocks; see {@link Lfm2ToolCalls}.
     */
    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return Lfm2ToolCalls.parse(responseText);
    }

    private int special(String spelling) {
        return tokenizer.getSpecialTokens().get(spelling);
    }

    /** {@code general.sampling.temp} in the LFM2.5 files. */
    @Override
    public double defaultTemperature() {
        return 0.1;
    }
}
