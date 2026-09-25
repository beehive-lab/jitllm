package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.tokenizer.GraniteTokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * Chat format for Granite models.
 *
 * <p>Granite uses a different chat template than Llama:
 * <|start_of_role|>system<|end_of_role|>.<|end_of_text|>
 * <|start_of_role|>user<|end_of_role|>.<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>.
 * Each turn ends with {@code <|end_of_text|>} and a newline, as the templates write it.
 *
 * <h2>Tool calling</h2>
 *
 * <p>Granite's releases carry two different tool formats, and which one a file speaks is read from
 * the chat template embedded in it ({@code tokenizer.chat_template}). Anything else — no template,
 * or a Granite release whose template neither dialect matches — has no tool calling, rather than a
 * guess:
 *
 * <ul>
 *   <li>{@link ToolDialect#GRANITE_3_2}: the tools are a separate {@code tools} turn after the
 *       system turn, written as {@code tojson(indent=4)}; without a system message the template's
 *       default one is used, extended with the tool instructions. The model answers with {@code
 *       <|tool_call|>} followed by a JSON list of calls; results are {@code tool} turns.
 *   <li>{@link ToolDialect#GRANITE_4}: the tools go into the system message between {@code
 *       <tools></tools>}, one {@code tojson} line each; calls are {@code
 *       <tool_call>{json}</tool_call>} blocks and a run of results is one {@code user} turn of
 *       {@code <tool_response>} blocks — the Qwen 2.5 shape.
 * </ul>
 */
public class GraniteChatFormat implements ChatFormat {

    /** The tool formats Granite templates use, identified from the embedded template. */
    public enum ToolDialect {
        /** No tool calling. */
        NONE,
        /** Granite 3.2: a {@code tools} turn, {@code <|tool_call|>} + a JSON list. */
        GRANITE_3_2,
        /** Granite 4.0: {@code <tools>} in the system message, {@code <tool_call>} blocks. */
        GRANITE_4
    }

    /** Granite 3.2's default system message when tools are attached and none is given. */
    static final String GRANITE_3_2_TOOLS_SYSTEM =
            "Knowledge Cutoff Date: April 2024.\n"
                    + "You are Granite, developed by IBM. You are a helpful AI assistant with access to"
                    + " the following tools. When a tool is required to answer the user's query,"
                    + " respond with <|tool_call|> followed by a JSON list of tools used. If a tool"
                    + " does not exist in the provided list of tools, notify the user that you do"
                    + " not have the ability to fulfill the request.";

    static final String GRANITE_4_TOOLS_PREFIX =
            "You are a helpful assistant with access to the following tools. You may call one or"
                    + " more tools to assist with the user query.\n\n"
                    + "You are provided with function signatures within <tools></tools> XML tags:\n"
                    + "<tools>";

    static final String GRANITE_4_TOOLS_SUFFIX =
            "\n</tools>\n\n"
                    + "For each tool call, return a json object with function name and arguments"
                    + " within <tool_call></tool_call> XML tags:\n"
                    + "<tool_call>\n"
                    + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                    + "</tool_call>. If a tool does not exist in the provided list of tools, notify"
                    + " the user that you do not have the ability to fulfill the request.";

    private static final String TOOL_CALL_3_2 = "<|tool_call|>";
    private static final List<String> GRANITE_4_MARKERS =
            List.of(
                    "<tool_call>",
                    "</tool_call>",
                    "<tool_response>",
                    "</tool_response>",
                    "<tools>",
                    "</tools>");

    protected final Tokenizer tokenizer;
    protected final int startRole;
    protected final int endRole;
    protected final int endOfText;
    protected final Set<Integer> stopTokens;
    private final ToolDialect toolDialect;
    private final Map<String, Integer> markers;

    public GraniteChatFormat(Tokenizer tokenizer) {
        this(tokenizer, null);
    }

    /**
     * @param chatTemplate the file's {@code tokenizer.chat_template}, or {@code null}; it decides
     *     the tool dialect
     */
    public GraniteChatFormat(Tokenizer tokenizer, String chatTemplate) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        this.startRole = specialTokens.getOrDefault("<|start_of_role|>", -1);
        this.endRole = specialTokens.getOrDefault("<|end_of_role|>", -1);

        // Use tokenizer's EOS token instead of hardcoding
        if (tokenizer instanceof GraniteTokenizer graniteTokenizer) {
            this.endOfText = graniteTokenizer.getEosTokenId();
        } else {
            this.endOfText = specialTokens.getOrDefault("<|end_of_text|>", 0);
        }

        this.stopTokens = Set.of(endOfText);
        this.toolDialect = dialectOf(chatTemplate, specialTokens);
        Map<String, Integer> found = new LinkedHashMap<>();
        for (String marker : allMarkers()) {
            Integer id = specialTokens.get(marker);
            if (id != null) {
                found.put(marker, id);
            }
        }
        this.markers = Map.copyOf(found);
    }

    /**
     * Which tool format a template speaks. Recognised by the constructs each template alone
     * contains, and only when the vocabulary has the markers that format needs.
     */
    static ToolDialect dialectOf(String chatTemplate, Map<String, Integer> specialTokens) {
        if (chatTemplate == null || specialTokens.get("<|start_of_role|>") == null) {
            return ToolDialect.NONE;
        }
        if (chatTemplate.contains("'<|start_of_role|>tools<|end_of_role|>'")
                && chatTemplate.contains("tools | tojson(indent=4)")
                && specialTokens.containsKey(TOOL_CALL_3_2)) {
            return ToolDialect.GRANITE_3_2;
        }
        if (chatTemplate.contains("tools_system_message_prefix")
                && chatTemplate.contains("'<tool_call>\\n{\"name\": \"'")
                && specialTokens.keySet().containsAll(GRANITE_4_MARKERS)) {
            return ToolDialect.GRANITE_4;
        }
        return ToolDialect.NONE;
    }

    /** The tool format this file speaks. */
    public ToolDialect toolDialect() {
        return toolDialect;
    }

    @Override
    public int getBeginOfText() {
        return endOfText; // For Granite, token 0 is both BOS and EOS
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        return header(message.role().name());
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    @Override
    public boolean supportsToolCalling() {
        return toolDialect != ToolDialect.NONE;
    }

    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        // Only for a caller that wants the text; the encoder uses encodeToolSystemMessage.
        return toolDialect == ToolDialect.GRANITE_4
                ? "\n\n" + granite4ToolsMessage(toolsJson)
                : ToolJson.dumps(ToolJson.parseSequence(toolsJson), 4);
    }

    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        List<Integer> tokens = header("system");
        if (toolDialect == ToolDialect.GRANITE_3_2) {
            if (systemContent == null) {
                tokens.addAll(templateText(GRANITE_3_2_TOOLS_SYSTEM));
            } else {
                tokens.addAll(tokenizer.encodeAsList(systemContent.strip()));
            }
            tokens.addAll(endOfTurn());
            tokens.addAll(header("tools"));
            tokens.addAll(
                    tokenizer.encodeAsList(ToolJson.dumps(ToolJson.parseSequence(toolsJson), 4)));
            tokens.addAll(endOfTurn());
            return tokens;
        }
        if (systemContent != null) {
            tokens.addAll(tokenizer.encodeAsList(systemContent.strip() + "\n\n"));
        }
        tokens.addAll(templateText(GRANITE_4_TOOLS_PREFIX));
        for (Object tool : ToolJson.parseSequence(toolsJson)) {
            tokens.addAll(tokenizer.encodeAsList("\n" + ToolJson.dumps(tool)));
        }
        tokens.addAll(templateText(GRANITE_4_TOOLS_SUFFIX));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    private static String granite4ToolsMessage(String toolsJson) {
        StringBuilder sb = new StringBuilder(GRANITE_4_TOOLS_PREFIX);
        for (Object tool : ToolJson.parseSequence(toolsJson)) {
            sb.append('\n').append(ToolJson.dumps(tool));
        }
        return sb.append(GRANITE_4_TOOLS_SUFFIX).toString();
    }

    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        return encodeToolCallAssistantTurn(List.of(toolCall));
    }

    /**
     * Granite 4: the template's {@code <tool_call>} blocks, one per call, newline-separated.
     *
     * <p>Granite 3.2: the template writes only an assistant message's content, so a call has to be
     * the content — written as the model writes it, {@code <|tool_call|>} and a JSON list.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = header("assistant");
        if (toolDialect == ToolDialect.GRANITE_3_2) {
            List<Object> calls = new ArrayList<>();
            for (ToolCallExtract call : toolCalls) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("name", call.name());
                entry.put("arguments", argumentsValue(call.argumentsJson()));
                calls.add(entry);
            }
            tokens.add(marker(TOOL_CALL_3_2));
            tokens.addAll(tokenizer.encodeAsList(ToolJson.dumps(calls)));
        } else {
            for (int i = 0; i < toolCalls.size(); i++) {
                ToolCallExtract call = toolCalls.get(i);
                if (i > 0) {
                    tokens.addAll(tokenizer.encodeAsList("\n"));
                }
                tokens.add(marker("<tool_call>"));
                tokens.addAll(
                        tokenizer.encodeAsList(
                                "\n{\"name\": \""
                                        + call.name()
                                        + "\", \"arguments\": "
                                        + argumentsText(call.argumentsJson())
                                        + "}\n"));
                tokens.add(marker("</tool_call>"));
            }
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        return encodeToolResults(List.of(new ToolResult(toolCallId, toolName, result)));
    }

    /**
     * Granite 3.2: one {@code tool} turn per result. Granite 4: one {@code user} turn for the whole
     * run, each result in {@code \n<tool_response>\n…\n</tool_response>}.
     */
    @Override
    public List<Integer> encodeToolResults(List<ToolResult> results) {
        List<Integer> tokens = new ArrayList<>();
        if (toolDialect == ToolDialect.GRANITE_3_2) {
            for (ToolResult result : results) {
                tokens.addAll(header("tool"));
                tokens.addAll(tokenizer.encodeAsList(result.content()));
                tokens.addAll(endOfTurn());
            }
            return tokens;
        }
        tokens.addAll(header("user"));
        for (ToolResult result : results) {
            tokens.addAll(tokenizer.encodeAsList("\n"));
            tokens.add(marker("<tool_response>"));
            tokens.addAll(tokenizer.encodeAsList("\n" + result.content() + "\n"));
            tokens.add(marker("</tool_response>"));
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        List<ToolCallExtract> all = extractAllToolCalls(responseText);
        return all.isEmpty() ? Optional.empty() : Optional.of(all.get(0));
    }

    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        if (toolDialect == ToolDialect.GRANITE_4) {
            return ToolCallParserUtils.parseAllToolCalls(responseText);
        }
        if (toolDialect == ToolDialect.GRANITE_3_2) {
            return parseGranite32(responseText);
        }
        return List.of();
    }

    /**
     * {@code <|tool_call|>} followed by a JSON list of {@code {"name", "arguments"}} objects.
     *
     * <p>The 2B model, greedy, writes the marker as the text {@code <tool_call>} rather than as the
     * {@code <|tool_call|>} token its template names; the list after it is the same, so that
     * spelling is read too. Nothing else is: without one of the two markers, JSON in a response is
     * an answer, not a call.
     */
    static List<ToolCallExtract> parseGranite32(String responseText) {
        List<ToolCallExtract> calls = new ArrayList<>();
        if (responseText == null) {
            return calls;
        }
        String marker = TOOL_CALL_3_2;
        int at = responseText.indexOf(marker);
        if (at == -1) {
            marker = "<tool_call>";
            at = responseText.indexOf(marker);
        }
        if (at == -1) {
            return calls;
        }
        Object value;
        try {
            // The first JSON value after the marker; whatever follows it is not part of the call.
            ToolJson reader = new ToolJson(responseText.substring(at + marker.length()));
            reader.skipWhitespace();
            value = reader.value();
        } catch (IllegalArgumentException e) {
            return calls; // an unterminated list: not a call a caller should run
        }
        List<?> items = value instanceof List<?> list ? list : List.of(value);
        for (Object item : items) {
            if (!(item instanceof Map<?, ?> map) || !(map.get("name") instanceof String name)) {
                continue;
            }
            Object arguments = map.get("arguments");
            if (arguments instanceof String s) {
                try {
                    arguments = ToolJson.parse(s);
                } catch (IllegalArgumentException e) {
                    continue;
                }
            }
            calls.add(
                    new ToolCallExtract(
                            name,
                            arguments == null || arguments == ToolJson.NULL
                                    ? "{}"
                                    : ToolJson.dumps(arguments)));
        }
        return calls;
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private List<Integer> header(String role) {
        List<Integer> tokens = new ArrayList<>();
        if (startRole >= 0) {
            tokens.add(startRole);
        }
        tokens.addAll(tokenizer.encodeAsList(role));
        if (endRole >= 0) {
            tokens.add(endRole);
        }
        return tokens;
    }

    private List<Integer> endOfTurn() {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(endOfText);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    private int marker(String spelling) {
        Integer id = markers.get(spelling);
        if (id == null) {
            throw new IllegalStateException("the vocabulary has no " + spelling + " token");
        }
        return id;
    }

    private static List<String> allMarkers() {
        List<String> all = new ArrayList<>(GRANITE_4_MARKERS);
        all.add(TOOL_CALL_3_2);
        return all;
    }

    /**
     * Text from the template itself, whose markers are single tokens — as llama.cpp and the
     * reference tokenizer read the rendered prompt.
     */
    private List<Integer> templateText(String text) {
        return MarkerText.encode(text, markers, tokenizer::encodeAsList);
    }

    /** The arguments as a value to embed: parsed when they are JSON, the string otherwise. */
    private static Object argumentsValue(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) {
            return Map.of();
        }
        try {
            return ToolJson.parse(argumentsJson);
        } catch (IllegalArgumentException e) {
            return argumentsJson;
        }
    }

    /**
     * The arguments as the template writes a mapping, {@code tojson}; text that is not JSON as it
     * stands, which is the template's string branch.
     */
    private static String argumentsText(String argumentsJson) {
        Object value = argumentsValue(argumentsJson);
        return value instanceof String s ? s : ToolJson.dumps(value);
    }
}
