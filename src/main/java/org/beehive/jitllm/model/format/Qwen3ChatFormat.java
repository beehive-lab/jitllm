package org.beehive.jitllm.model.format;

import java.util.*;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;

/** Utility tailored for the Chat Markup Language (ChatML) prompt format. */
public class Qwen3ChatFormat implements ChatFormat {

    protected final int beginOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfTextFim;
    protected final int imStart; // beginOfText
    protected final int imEnd; // endOfText
    protected final int fimPrefix;
    protected final int fimSuffix;
    protected final int fimMiddle;
    protected Qwen3Tokenizer tokenizer;
    protected ChatTokens chatTokens;

    public Qwen3ChatFormat(Qwen3Tokenizer tokenizer, ChatTokens chatTokens) {
        this.tokenizer = tokenizer;
        this.chatTokens = chatTokens;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = -1; // Qwen3 has no BOS token; getBeginOfText() falls back to startHeader
        this.startHeader = specialTokens.getOrDefault(chatTokens.tStartHeader(), -1);
        this.endHeader = specialTokens.getOrDefault(chatTokens.tEndHeader(), -1);
        this.endOfTurn = specialTokens.getOrDefault(chatTokens.tEndOfTurn(), -1);
        this.endOfText = specialTokens.getOrDefault(chatTokens.tEndOfText(), -1);
        this.endOfTextFim = specialTokens.getOrDefault(chatTokens.tEndOfTextFim(), -1);

        this.imStart = startHeader;
        this.imEnd = endHeader;

        this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
    }

    public ChatTokens chatTokens() {
        return chatTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (endHeader == -1) {
            // DeepSeek-R1
            String sToken =
                    switch (message.role().name()) {
                        case "system" -> null;
                        case "user" -> "<｜User｜>";
                        case "assistant" -> "<｜Assistant｜>";
                        case "fim_prefix" -> "<|fim_prefix|>";
                        case "fim_middle" -> "<|fim_middle|>";
                        case "fim_suffix" -> "<|fim_suffix|>";
                        default -> null;
                    };
            if (sToken != null) {
                Integer token = tokenizer.getSpecialTokens().get(sToken);
                if (token == null) {
                    throw new IllegalStateException(String.format("Unknown token '%s'", sToken));
                }
                tokens.add(token);
            }
        } else if (Role.FIM_PREFIX.equals(message.role())) {
            // fill-in-the-middle, token fim_prefix.
            tokens.add(fimPrefix);
        } else if (Role.FIM_SUFFIX.equals(message.role())) {
            tokens.add(fimSuffix);
        } else if (Role.FIM_MIDDLE.equals(message.role())) {
            tokens.add(fimMiddle);
        } else {
            // Add the special token directly, don't try to encode it
            tokens.add(imStart);
            // Encode the role name as ordinary text (no special tokens in role names)
            tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.role().name()));
            tokens.addAll(this.tokenizer.encodeOrdinaryAsList("\n"));
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        // Encode message content as ordinary text
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.content().strip()));
        boolean isFim =
                Role.FIM_PREFIX.equals(message.role())
                        || Role.FIM_SUFFIX.equals(message.role())
                        || Role.FIM_MIDDLE.equals(message.role());
        if (imEnd != -1 && !isFim) {
            // Add the end token directly
            tokens.add(imEnd);
            // Standard ChatML: a newline follows <|im_end|>
            tokens.addAll(this.tokenizer.encodeOrdinaryAsList("\n"));
        }
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        if (beginOfText == -1) {
            // deepseek-r1
            return startHeader;
        } else {
            return beginOfText;
        }
    }

    @Override
    public Set<Integer> getStopTokens() {
        if (imEnd == -1 && endOfText == -1) {
            throw new IllegalStateException("No stop token is defined.");
        }

        // Only add valid token IDs (not -1)
        Set<Integer> stopTokens = new HashSet<>();
        if (imEnd != -1) {
            stopTokens.add(imEnd);
        }
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }
        if (endOfTextFim != -1) {
            stopTokens.add(endOfTextFim);
        }

        return stopTokens;
    }

    @Override
    public double defaultTemperature() {
        return 0.8;
    }

    @Override
    public double defaultTopP() {
        return 0.9;
    }

    /**
     * Genuine Qwen3 exposes the {@code enable_thinking} template switch and so supports thinking
     * control. DeepSeek-R1 is routed through this same format (detected by the absence of an {@code
     * <|im_end|>} token) but is a pure reasoning model with no off-switch, so it reports {@code
     * false} and is left to always reason.
     */
    @Override
    public boolean supportsThinking() {
        return imEnd != -1;
    }

    /**
     * {@code </think>}, for every model routed here — DeepSeek-R1 reasons without an off-switch.
     */
    @Override
    public int reasoningEndToken() {
        return tokenizer.getThinkEndToken();
    }

    /**
     * Qwen3 thinking control. When thinking is disabled, primes a pre-closed {@code
     * <think>\n\n</think>\n\n} block right after the assistant header so the model skips its
     * reasoning phase — matching the {@code enable_thinking=false} branch of the official Qwen3
     * chat template. When enabled (or for DeepSeek-R1, which cannot disable thinking), returns
     * nothing and lets the model reason on its own.
     *
     * <p>The {@code <think>}/{@code </think>} markers are emitted as their <em>canonical</em>
     * single token ids (not ordinary BPE sub-pieces): the tokenizer strips them from its special
     * map so reasoning renders as text, but the model only recognises the closed block — and thus
     * actually skips reasoning — when it sees the real control tokens it was trained on.
     */
    @Override
    public List<Integer> encodeThinkingControl(boolean enableThinking) {
        if (enableThinking || !supportsThinking()) {
            return List.of();
        }
        int thinkStart = tokenizer.getThinkStartToken();
        int thinkEnd = tokenizer.getThinkEndToken();
        if (thinkStart == -1 || thinkEnd == -1) {
            // GGUF without dedicated think tokens — fall back to ordinary text encoding.
            return tokenizer.encodeOrdinaryAsList("<think>\n\n</think>\n\n");
        }
        List<Integer> tokens = new ArrayList<>();
        tokens.add(thinkStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("\n\n"));
        tokens.add(thinkEnd);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("\n\n"));
        return tokens;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    /**
     * Qwen 2.5's default system message, which its template puts ahead of the tools when the
     * conversation has none. Used only when the file's own template says so.
     */
    static final String QWEN_2_5_DEFAULT_SYSTEM =
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    private static final List<String> TOOL_MARKERS =
            List.of("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>");

    /**
     * The system text the template uses when tools are attached and the conversation has no system
     * message, or {@code null} when it uses none.
     */
    private String defaultToolSystemMessage;

    /**
     * Declares the file's chat template ({@code tokenizer.chat_template}), so template-specific
     * defaults are read from it rather than assumed. Returns this format.
     */
    public Qwen3ChatFormat withChatTemplate(String chatTemplate) {
        this.defaultToolSystemMessage =
                chatTemplate != null
                                && chatTemplate.contains("'" + QWEN_2_5_DEFAULT_SYSTEM + "'")
                                && chatTemplate.contains("{%- if tools %}")
                        ? QWEN_2_5_DEFAULT_SYSTEM
                        : null;
        return this;
    }

    /**
     * Only a ChatML model whose vocabulary has the {@code <tool_call>} / {@code </tool_call>}
     * tokens its template's tool format is written in (Qwen 2.5, Qwen 3, Qwen 3.5). Not
     * DeepSeek-R1-Distill-Qwen, routed here without {@code <|im_end|>}: its template renders no
     * tool definitions. Not Qwen 1.5 / Qwen 2 (and their MoE releases): their templates have no
     * tools and their vocabularies no {@code <tool_call>}.
     */
    @Override
    public boolean supportsToolCalling() {
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        return imEnd != -1
                && special.containsKey("<tool_call>")
                && special.containsKey("</tool_call>");
    }

    /**
     * The tools block of the system turn, as the Qwen 2.5 and Qwen 3 templates write it: the
     * instructions, and one {@code tojson} line per tool between {@code <tools></tools>}.
     */
    protected String toolsBlock(String toolsJson) {
        return "# Tools\n\n"
                + "You may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n"
                + "<tools>"
                + toolLines(toolsJson)
                + "\n</tools>\n\n"
                + "For each function call, return a json object with function name and arguments "
                + "within <tool_call></tool_call> XML tags:\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>";
    }

    /** {@code "\n" + tool | tojson} for each tool, as the templates' tools loop writes them. */
    protected static String toolLines(String toolsJson) {
        StringBuilder lines = new StringBuilder();
        for (Object tool : ToolJson.parseSequence(toolsJson)) {
            lines.append('\n').append(ToolJson.dumps(tool));
        }
        return lines.toString();
    }

    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return "\n\n" + toolsBlock(toolsJson);
    }

    /**
     * {@code <|im_start|>system\n{system}\n\n{tools block}<|im_end|>\n}; without a system message,
     * the template's default one (Qwen 2.5) or none (Qwen 3).
     */
    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        String system = systemContent != null ? systemContent.strip() : defaultToolSystemMessage;
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("system\n"));
        if (system != null) {
            tokens.addAll(tokenizer.encodeOrdinaryAsList(system + "\n\n"));
        }
        tokens.addAll(templateText(toolsBlock(toolsJson)));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        return encodeToolCallAssistantTurn(List.of(toolCall));
    }

    /**
     * One assistant turn with every call, as the templates replay {@code tool_calls}: {@code
     * <tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>} per call, newline-separated, the
     * arguments through {@code tojson}.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        for (int i = 0; i < toolCalls.size(); i++) {
            ToolCallExtract call = toolCalls.get(i);
            if (i > 0) {
                tokens.addAll(tokenizer.encodeOrdinaryAsList("\n"));
            }
            tokens.addAll(markerTokens("<tool_call>"));
            tokens.addAll(
                    tokenizer.encodeOrdinaryAsList(
                            "\n{\"name\": \""
                                    + call.name()
                                    + "\", \"arguments\": "
                                    + argumentsJson(call.argumentsJson())
                                    + "}\n"));
            tokens.addAll(markerTokens("</tool_call>"));
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        return encodeToolResults(List.of(new ToolResult(toolCallId, toolName, result)));
    }

    /**
     * A run of results as one {@code user} turn, each in {@code \n<tool_response>\n…\n
     * </tool_response>} — the templates' {@code role: tool} branch. Qwen has no tool role.
     */
    @Override
    public List<Integer> encodeToolResults(List<ToolResult> results) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("user"));
        for (ToolResult result : results) {
            tokens.addAll(tokenizer.encodeOrdinaryAsList("\n"));
            tokens.addAll(markerTokens("<tool_response>"));
            tokens.addAll(tokenizer.encodeOrdinaryAsList("\n" + result.content() + "\n"));
            tokens.addAll(markerTokens("</tool_response>"));
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    /**
     * Detects a tool call enclosed in {@code <tool_call>…</tool_call>} tags. Delegates to {@link
     * ToolCallParserUtils#parseToolCallResponse}.
     */
    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return ToolCallParserUtils.parseToolCallResponse(responseText);
    }

    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return ToolCallParserUtils.parseAllToolCalls(responseText);
    }

    /** {@code <|im_end|>\n}. */
    protected List<Integer> endOfTurn() {
        List<Integer> tokens = new ArrayList<>();
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        tokens.addAll(tokenizer.encodeOrdinaryAsList("\n"));
        return tokens;
    }

    /** Template text whose tool markers are encoded as their tokens. */
    protected List<Integer> templateText(String text) {
        Map<String, Integer> markers = new LinkedHashMap<>();
        for (String spelling : TOOL_MARKERS) {
            Integer id = tokenizer.getSpecialTokens().get(spelling);
            if (id != null) {
                markers.put(spelling, id);
            }
        }
        return MarkerText.encode(text, markers, tokenizer::encodeOrdinaryAsList);
    }

    /** A tool marker's token, or its spelling as text when the vocabulary lacks it. */
    protected List<Integer> markerTokens(String spelling) {
        Integer id = tokenizer.getSpecialTokens().get(spelling);
        return id != null ? List.of(id) : tokenizer.encodeOrdinaryAsList(spelling);
    }

    /** Arguments as the templates' {@code tojson} writes a mapping; non-JSON text as it stands. */
    static String argumentsJson(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) {
            return "{}";
        }
        try {
            return ToolJson.dumps(ToolJson.parse(argumentsJson));
        } catch (IllegalArgumentException e) {
            return argumentsJson;
        }
    }
}
