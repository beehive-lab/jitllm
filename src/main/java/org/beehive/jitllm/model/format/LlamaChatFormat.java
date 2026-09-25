package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.tokenizer.Tokenizer;

public class LlamaChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final int pythonTag;
    protected final Set<Integer> stopTokens;

    public LlamaChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.pythonTag = specialTokens.getOrDefault("<|python_tag|>", -1); // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    @Override
    public double defaultTemperature() {
        return 0.3;
    }

    @Override
    public double defaultTopP() {
        return 0.95;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    /**
     * The Llama 3.1 / 3.2 templates' {@code date_string} default, used when the renderer provides
     * no {@code strftime_now}. A fixed value keeps the prompt — and so prefix reuse and tests —
     * deterministic.
     */
    static final String TEMPLATE_DATE = "26 Jul 2024";

    @Override
    public boolean supportsToolCalling() {
        return true;
    }

    /**
     * Llama 3.1 and 3.2 Instruct put the tool definitions in the <em>first user message</em>
     * ({@code tools_in_user_message = true} by default in both templates). The system message
     * receives only the environment and date lines.
     */
    @Override
    public boolean injectsToolsInUserMessage() {
        return true;
    }

    /**
     * The system-message lines the template writes ahead of the caller's system text when tools are
     * attached: {@code Environment: ipython}, the knowledge cutoff and the date.
     */
    @Override
    public String toolSystemMessagePrefix() {
        return "Environment: ipython\n"
                + "Cutting Knowledge Date: December 2023\n"
                + "Today Date: "
                + TEMPLATE_DATE
                + "\n\n";
    }

    /**
     * The system turn of a tool request: the environment and date lines, then the caller's system
     * text, trimmed, as the template writes it — present even when the conversation has no system
     * message.
     */
    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.SYSTEM, "")));
        tokens.addAll(
                tokenizer.encodeAsList(
                        toolSystemMessagePrefix()
                                + (systemContent == null ? "" : systemContent.strip())));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * The template's preamble to the first user message: the instructions, then each tool as {@code
     * tojson(indent=4)} followed by a blank line.
     */
    @Override
    public String toolFirstUserMessagePrefix(String toolsJson) {
        StringBuilder prefix =
                new StringBuilder(
                        "Given the following functions, please respond with a JSON for a function"
                                + " call with its proper arguments that best answers the given"
                                + " prompt.\n\n"
                                + "Respond in the format {\"name\": function name, \"parameters\":"
                                + " dictionary of argument name and its value}."
                                + "Do not use variables.\n\n");
        for (Object tool : ToolJson.parseSequence(toolsJson)) {
            prefix.append(ToolJson.dumps(tool, 4)).append("\n\n");
        }
        return prefix.toString();
    }

    /**
     * A prior tool call, as the templates replay a custom tool's call: {@code {"name": "…",
     * "parameters": {…}}<|eot_id|>}, the arguments through {@code tojson}. No {@code
     * <|python_tag|>} and no {@code <|eom_id|>}: those mark Llama 3.1's built-in tools, which the
     * facade does not offer.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        return encodeToolCallAssistantTurn(List.of(toolCall));
    }

    /**
     * The templates accept one call per assistant turn and reject more; several calls the model
     * made in one turn are replayed in that one turn, one JSON object per line.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.ASSISTANT, "")));
        StringBuilder body = new StringBuilder();
        for (int i = 0; i < toolCalls.size(); i++) {
            ToolCallExtract call = toolCalls.get(i);
            if (i > 0) {
                body.append('\n');
            }
            body.append("{\"name\": \"")
                    .append(call.name())
                    .append("\", \"parameters\": ")
                    .append(Qwen3ChatFormat.argumentsJson(call.argumentsJson()))
                    .append('}');
        }
        tokens.addAll(tokenizer.encodeAsList(body.toString()));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * A tool result in the {@code ipython} role: {@code
     * <|start_header_id|>ipython<|end_header_id|>…"result"<|eot_id|>}.
     *
     * <p>The result is written through {@code tojson}, so it arrives as a JSON string literal. That
     * is what the Llama 3.1 and 3.2 templates do with string content — their {@code content is
     * iterable} test is true for a string, in Jinja2 and in llama.cpp's renderer alike — and so
     * what a model served by either sees.
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(new Role("ipython"), "")));
        tokens.addAll(tokenizer.encodeAsList(ToolJson.dumps(result)));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * Detects a tool call in the decoded response text. Supports LLaMA 3.1 (native {@code
     * <|python_tag|>} + {@code "parameters"} key), LLaMA 3.2 ({@code "arguments"} key, tag often
     * absent), and a raw-JSON fallback for smaller models. Delegates to {@link
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

    /**
     * Adds {@code <|eom_id|>} to the stop tokens when tools are enabled. LLaMA 3.1 ends tool-call
     * turns with {@code <|eom_id|>} instead of {@code <|eot_id|>}.
     */
    @Override
    public Set<Integer> getToolAwareStopTokens() {
        if (endOfMessage != -1) {
            return Set.of(endOfText, endOfTurn, endOfMessage);
        }
        return stopTokens;
    }
}
