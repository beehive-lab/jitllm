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

    @Override
    public boolean supportsToolCalling() {
        return true;
    }

    /**
     * Llama 3.2 Instruct injects tool definitions into the <em>first user message</em> (the
     * GGUF-embedded chat template has {@code tools_in_user_message = true} by default). The system
     * message receives only an environment prefix; the tools and usage instructions go in the user
     * turn.
     */
    @Override
    public boolean injectsToolsInUserMessage() {
        return true;
    }

    /**
     * System-message prefix that signals tool availability to Llama 3.2. Matches the template's
     * {@code "Environment: ipython\n"} line.
     */
    @Override
    public String toolSystemMessagePrefix() {
        return "Environment: ipython\n\n";
    }

    /**
     * Prepends tool definitions and usage instructions to the first user message, matching the
     * Llama 3.2 GGUF chat template ({@code tools_in_user_message = true}).
     *
     * <p>Format mirrors:
     *
     * <pre>
     * Given the following functions, please respond with a JSON for a function call
     * with its proper arguments that best answers the given prompt.
     *
     * Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.
     * Do not use variables.
     *
     * {toolsJson}
     *
     * </pre>
     */
    @Override
    public String toolFirstUserMessagePrefix(String toolsJson) {
        return "Given the following functions, please respond with a JSON for a function call "
                + "with its proper arguments that best answers the given prompt.\n\n"
                + "Respond in the format {\"name\": function name, \"parameters\": dictionary of "
                + "argument name and its value}. Do not use variables.\n\n"
                + toolsJson
                + "\n\n";
    }

    /**
     * Re-encodes a prior assistant tool-call turn for multi-turn history using the Llama 3.2 native
     * JSON format: {@code {"name":"…","parameters":{…}}<|eot_id|>}.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.ASSISTANT, "")));
        // Preserve the <|python_tag|> prefix used by LLaMA 3.1/3.2 for tool calls so that
        // replayed history looks identical to what the model originally generated.
        if (pythonTag != -1) {
            tokens.add(pythonTag);
        }
        String json =
                "{\"name\": \""
                        + toolCall.name()
                        + "\", \"parameters\": "
                        + toolCall.argumentsJson()
                        + "}";
        tokens.addAll(tokenizer.encodeAsList(json));
        // LLaMA 3.1 ends tool-call turns with <|eom_id|>; fall back to <|eot_id|> for 3.2.
        tokens.add(endOfMessage != -1 ? endOfMessage : endOfTurn);
        return tokens;
    }

    /**
     * Encodes a tool result using the LLaMA "ipython" role. Format: {@code
     * <|start_header_id|>ipython<|end_header_id|>\nresult<|eot_id|>}
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList("ipython"));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        tokens.addAll(tokenizer.encodeAsList(result));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * Encodes multiple tool calls as a single assistant turn. For a single call, delegates to the
     * existing single-call method (preserving the {@code <|python_tag|>} prefix on LLaMA 3.1). For
     * multiple calls, LLaMA 3.1 prefixes each with {@code <|python_tag|>}; LLaMA 3.2 (no
     * python_tag) uses {@code <tool_call>} blocks.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) return List.of();
        if (toolCalls.size() == 1) return encodeToolCallAssistantTurn(toolCalls.get(0));
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.ASSISTANT, "")));
        for (ToolCallExtract tc : toolCalls) {
            String json =
                    "{\"name\": \"" + tc.name() + "\", \"parameters\": " + tc.argumentsJson() + "}";
            if (pythonTag != -1) {
                tokens.add(pythonTag);
                tokens.addAll(tokenizer.encodeAsList(json + "\n"));
            } else {
                tokens.addAll(tokenizer.encodeAsList("<tool_call>\n" + json + "\n</tool_call>\n"));
            }
        }
        tokens.add(endOfMessage != -1 ? endOfMessage : endOfTurn);
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
