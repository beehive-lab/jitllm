package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.ToolCallExtract;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * Turns a facade conversation into the tokens the model sees.
 *
 * <p>Package-private, and the <b>only</b> place the facade names {@link ChatFormat}. Formatting
 * stays model-owned and internal [§9]: no facade type carries a chat format, a template or a
 * stop-token set, and a caller cannot influence how a family renders a turn.
 *
 * <p>The assembly order is the one both integrations already use — tool definitions into the system
 * message, or into the first user message for formats that want them there; then each turn; then an
 * assistant header to prime the response. It is copied deliberately rather than reinvented: it is
 * the shape that has been producing correct tool calls in their tests, and a second opinion about
 * where tool JSON belongs would be a new defect rather than a new idea.
 */
final class ConversationEncoder {

    private final Model model;
    private final ThinkingMode thinkingMode;

    ConversationEncoder(Model model, ThinkingMode thinkingMode) {
        this.model = model;
        this.thinkingMode = thinkingMode;
    }

    /**
     * The complete model input for a conversation — every token, in order.
     *
     * <p>This whole list is what prefix reuse compares [A1], which is why the tool specifications
     * are encoded here rather than being handled elsewhere: attaching a tool changes the system
     * content, so it changes the encoded input from its first tokens.
     */
    List<Integer> encode(List<ChatMessage> messages, List<ToolSpec> tools) {
        ChatFormat chatFormat = model.chatFormat();
        String toolsJson = tools.isEmpty() ? null : buildToolsJson(tools);
        if (toolsJson != null && !chatFormat.supportsToolCalling()) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.TOOLS_UNSUPPORTED.message("tool calling is not supported by ")
                            + model.getModelType()
                            + "'s chat format ("
                            + chatFormat.getClass().getSimpleName()
                            + ")");
        }

        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(chatFormat.getBeginOfText());
        }

        boolean injectInUserMessage = toolsJson != null && chatFormat.injectsToolsInUserMessage();
        boolean hasSystemMessage = messages.stream().anyMatch(m -> m.role() == ChatRole.SYSTEM);
        boolean toolsInjected = false;

        // A format that wants tools in the system message, and a conversation with no system turn
        // to put them in, gets one. So does a format that puts them in the user message but still
        // writes a system turn for them (Llama's environment lines).
        if (toolsJson != null
                && !hasSystemMessage
                && model.shouldAddSystemPrompt()
                && (!injectInUserMessage || !chatFormat.toolSystemMessagePrefix().isEmpty())) {
            tokens.addAll(chatFormat.encodeToolSystemMessage(null, toolsJson));
            toolsInjected = !injectInUserMessage;
        }

        // Whether an assistant turn is open: a format that writes tool results inside the turn
        // that made the calls leaves it open for the answer (Gemma 4). Never set otherwise.
        boolean openAssistantTurn = false;

        for (int index = 0; index < messages.size(); index++) {
            ChatMessage message = messages.get(index);
            if (openAssistantTurn
                    && message.role() != ChatRole.ASSISTANT
                    && message.role() != ChatRole.TOOL) {
                tokens.addAll(chatFormat.encodeOpenAssistantTurnEnd());
                openAssistantTurn = false;
            }
            switch (message.role()) {
                case USER -> {
                    String content = textOf(message);
                    if (injectInUserMessage && !toolsInjected) {
                        content = chatFormat.toolFirstUserMessagePrefix(toolsJson) + content;
                        toolsInjected = true;
                    }
                    tokens.addAll(
                            chatFormat.encodeMessage(
                                    new ChatFormat.Message(ChatFormat.Role.USER, content)));
                }
                case SYSTEM -> {
                    if (!model.shouldAddSystemPrompt()) {
                        continue; // this family has no system turn; dropping it is its rule
                    }
                    String content = textOf(message);
                    if (toolsJson != null) {
                        tokens.addAll(chatFormat.encodeToolSystemMessage(content, toolsJson));
                        toolsInjected |= !injectInUserMessage;
                        continue;
                    }
                    tokens.addAll(
                            chatFormat.encodeMessage(
                                    new ChatFormat.Message(ChatFormat.Role.SYSTEM, content)));
                }
                case ASSISTANT -> {
                    List<ToolCallExtract> calls =
                            message.content().stream()
                                    .filter(ChatContent.ToolCall.class::isInstance)
                                    .map(ChatContent.ToolCall.class::cast)
                                    .map(
                                            call ->
                                                    new ToolCallExtract(
                                                            call.name(),
                                                            call.argumentsJson(),
                                                            Optional.of(call.id())))
                                    .toList();
                    if (!calls.isEmpty()) {
                        tokens.addAll(
                                openAssistantTurn
                                        ? chatFormat.encodeToolCallContinuation(calls)
                                        : chatFormat.encodeToolCallAssistantTurn(calls));
                        openAssistantTurn = chatFormat.toolResultsStayInAssistantTurn();
                    } else if (openAssistantTurn) {
                        tokens.addAll(chatFormat.encodeAssistantContinuation(textOf(message)));
                        openAssistantTurn = false;
                    } else {
                        tokens.addAll(
                                chatFormat.encodeMessage(
                                        new ChatFormat.Message(
                                                ChatFormat.Role.ASSISTANT, textOf(message))));
                    }
                }
                case TOOL -> {
                    // The whole run of consecutive results at once: some templates wrap a run in
                    // a single turn.
                    List<ChatFormat.ToolResult> results = new ArrayList<>();
                    while (true) {
                        for (ChatContent piece : messages.get(index).content()) {
                            ChatContent.ToolResult result = (ChatContent.ToolResult) piece;
                            results.add(
                                    new ChatFormat.ToolResult(
                                            result.id(), result.name(), result.resultJson()));
                        }
                        if (index + 1 < messages.size()
                                && messages.get(index + 1).role() == ChatRole.TOOL) {
                            index++;
                        } else {
                            break;
                        }
                    }
                    tokens.addAll(chatFormat.encodeToolResults(results));
                }
            }
        }

        // Prime the model to start an assistant turn — unless one is still open, in which case the
        // model's answer continues it, as the template's generation prompt does after tool results.
        if (!openAssistantTurn) {
            tokens.addAll(
                    chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            appendThinkingControl(tokens);
        }
        return tokens;
    }

    /**
     * Encodes the resolved reasoning mode, if one was asked for.
     *
     * <p>After the assistant header, because the control primes the turn the model is about to
     * write. {@link ThinkingMode#DEFAULT} appends nothing, so a family with no reasoning phase and
     * a caller who said nothing are the same case — which is what makes DEFAULT the value that
     * changes nothing.
     *
     * <p>The translation from a mode to tokens is the family's, and it stays here rather than
     * anywhere a caller can see: no thinking token or control string reaches the facade.
     */
    void appendThinkingControl(List<Integer> tokens) {
        if (thinkingMode.isExplicit()) {
            tokens.addAll(
                    model.chatFormat().encodeThinkingControl(thinkingMode == ThinkingMode.ENABLED));
        }
    }

    /**
     * The tool calls in a response, or empty when there are none — including when the response
     * contains tool-shaped text that does not parse.
     *
     * <p>A format that cannot do tool calling returns empty rather than throwing: extraction is
     * asked of every response when tools were supplied, and "this format has no tool calls in it"
     * is a true answer, not an error.
     */
    List<ChatContent.ToolCall> extractToolCalls(String responseText) {
        ChatFormat chatFormat = model.chatFormat();
        if (!chatFormat.supportsToolCalling()) {
            return List.of();
        }
        List<ToolCallExtract> extracted;
        try {
            extracted = chatFormat.extractAllToolCalls(responseText);
        } catch (RuntimeException e) {
            // Malformed tool-shaped output is text, not a failure: the caller gets what the model
            // actually said. Reporting a call that did not parse would have them execute something
            // the model never asked for [A4].
            return List.of();
        }
        List<ChatContent.ToolCall> calls = new ArrayList<>(extracted.size());
        for (ToolCallExtract call : extracted) {
            if (call.name() == null || call.name().isBlank() || call.argumentsJson() == null) {
                continue; // not a valid call; not reported as one
            }
            // The id is optional in the internal extract and required on the facade, because a
            // caller has to match a result back to a call. Formats that do not emit one get a
            // positional identifier rather than a null the caller has to handle.
            String id =
                    call.id()
                            .filter(value -> !value.isBlank())
                            .orElse("call-" + (calls.size() + 1));
            calls.add(new ChatContent.ToolCall(id, call.name(), call.argumentsJson()));
        }
        return List.copyOf(calls);
    }

    /** Whether this model's format can do tool calling at all. */
    boolean supportsTools() {
        return model.chatFormat().supportsToolCalling();
    }

    /** The stop tokens for a request, tool-aware when the request carries tools [§6]. */
    java.util.Set<Integer> stopTokens(boolean withTools) {
        ChatFormat chatFormat = model.chatFormat();
        return withTools ? chatFormat.getToolAwareStopTokens() : chatFormat.getStopTokens();
    }

    private static String textOf(ChatMessage message) {
        return message.content().stream()
                .filter(ChatContent.Text.class::isInstance)
                .map(ChatContent.Text.class::cast)
                .map(ChatContent.Text::text)
                .collect(Collectors.joining());
    }

    /**
     * The tool specifications as the JSON the formats splice in.
     *
     * <p>One JSON object per tool, separated by blank lines — the shape the integrations build and
     * the formats were written against.
     */
    private static String buildToolsJson(List<ToolSpec> tools) {
        return tools.stream()
                .map(ConversationEncoder::toolJson)
                .collect(Collectors.joining("\n\n"));
    }

    /**
     * One tool, as the chat formats expect to read it.
     *
     * <p><b>Compact, and with an empty description omitted</b> — deliberately, because this is the
     * text a model reads and reasons about. The shape and spacing match what the LangChain4j and
     * Quarkus adapters produced before they moved onto this facade (a {@code LinkedHashMap} through
     * their JSON writer), and that is the text the formats' tool prompts were written against.
     * Rendering {@code "description": ""} where there was previously no key at all is exactly the
     * kind of difference a model notices and this project cannot predict.
     */
    private static String toolJson(ToolSpec tool) {
        StringBuilder json =
                new StringBuilder("{\"type\":\"function\",\"function\":{\"name\":\"")
                        .append(escape(tool.name()))
                        .append('"');
        if (!tool.description().isEmpty()) {
            json.append(",\"description\":\"").append(escape(tool.description())).append('"');
        }
        return json.append(",\"parameters\":")
                .append(tool.parametersJsonSchema())
                .append("}}")
                .toString();
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
