package org.beehive.jllm.model.format;

import java.util.*;
import org.beehive.jllm.tokenizer.Qwen3Tokenizer;

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

    @Override
    public boolean supportsToolCalling() {
        return true;
    }

    /**
     * Qwen3 tool calling system prompt suffix. Appended to the system message; instructs the model
     * to wrap tool calls in {@code <tool_call>…</tool_call>} XML tags.
     */
    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return "\n\n# Tools\n\n"
                + "You may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n"
                + "<tools>\n"
                + toolsJson
                + "\n</tools>\n\n"
                + "For each function call, return a json object with function name and arguments "
                + "within <tool_call></tool_call> XML tags:\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>";
    }

    /**
     * Re-encodes a prior assistant tool-call turn for multi-turn history. Format: {@code
     * <|im_start|>assistant\n<tool_call>\nJSON\n</tool_call><|im_end|>}
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        String json =
                "{\"name\":\""
                        + toolCall.name()
                        + "\",\"arguments\":"
                        + toolCall.argumentsJson()
                        + "}";
        tokens.addAll(tokenizer.encodeOrdinaryAsList("<tool_call>\n" + json + "\n</tool_call>"));
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    /**
     * Encodes multiple tool calls as a single assistant turn: one {@code <|im_start|>assistant}
     * header, all {@code <tool_call>} blocks concatenated, then {@code <|im_end|>}. For a single
     * call, delegates to the existing single-call method.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) return List.of();
        if (toolCalls.size() == 1) return encodeToolCallAssistantTurn(toolCalls.get(0));
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        for (ToolCallExtract tc : toolCalls) {
            String json =
                    "{\"name\":\"" + tc.name() + "\",\"arguments\":" + tc.argumentsJson() + "}";
            tokens.addAll(
                    tokenizer.encodeOrdinaryAsList("<tool_call>\n" + json + "\n</tool_call>"));
        }
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    /**
     * Encodes a tool result in the native Qwen3 format: a {@code user} turn whose content is the
     * result wrapped in {@code <tool_response>…</tool_response>} tags, matching the official Qwen3
     * chat template. (Qwen3 has no dedicated "tool" role — results are delivered as user turns.)
     * Format: {@code <|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>}
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(
                tokenizer.encodeOrdinaryAsList(
                        "user\n<tool_response>\n" + result + "\n</tool_response>"));
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
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
}
