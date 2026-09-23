package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * How a family turns messages into tokens.
 *
 * <p>Constructed by each family's loader, which already knows which format it needs. There is
 * deliberately no central factory: a switch over tokenizer types here would mean adding a family
 * edits this file, which is what Rule 15 exists to prevent.
 */
public interface ChatFormat {

    default ChatTokens chatTokens() {
        throw new UnsupportedOperationException(
                "ChatFormat for Llama and Mistral does not support chatTokens");
    }

    List<Integer> encodeHeader(Message message);

    List<Integer> encodeMessage(Message message);

    int getBeginOfText();

    Set<Integer> getStopTokens();

    /**
     * Returns {@code true} when this chat format supports tool calling. Formats that implement
     * tool-calling methods must override this to return {@code true}. Callers should check this
     * before passing tool specifications to avoid hitting the default {@link
     * UnsupportedOperationException} deep inside a format method.
     */
    default boolean supportsToolCalling() {
        return false;
    }

    /**
     * Returns plain text to append to the system message content when tools are available. Used by
     * formats that inject tool definitions into the <em>system</em> message.
     *
     * <p>Formats that inject tools into the <em>user</em> message instead should override {@link
     * #injectsToolsInUserMessage()}, {@link #toolSystemMessagePrefix()}, and {@link
     * #toolFirstUserMessagePrefix(String)} rather than this method.
     *
     * @param toolsJson JSON array of tool definitions
     */
    default String toolSystemPromptSuffix(String toolsJson) {
        throw new UnsupportedOperationException(
                "Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Returns {@code true} when this format injects tool definitions into the <em>first user
     * message</em> instead of the system message.
     *
     * <p>When this returns {@code true}, callers should:
     *
     * <ol>
     *   <li>Prepend {@link #toolSystemMessagePrefix()} to the system message content.
     *   <li>Prepend {@link #toolFirstUserMessagePrefix(String)} to the first user message.
     * </ol>
     *
     * When {@code false} (default), callers should append {@link #toolSystemPromptSuffix} to the
     * system message as before.
     */
    default boolean injectsToolsInUserMessage() {
        return false;
    }

    /**
     * Returns text to <em>prepend</em> to the system message content when tools are active and
     * {@link #injectsToolsInUserMessage()} is {@code true}. Default: empty string (no prefix).
     */
    default String toolSystemMessagePrefix() {
        return "";
    }

    /**
     * Returns the preamble to <em>prepend</em> to the first user message when {@link
     * #injectsToolsInUserMessage()} is {@code true}. The preamble should include the tool
     * definitions and usage instructions.
     *
     * @param toolsJson JSON array of tool definitions
     */
    default String toolFirstUserMessagePrefix(String toolsJson) {
        return "";
    }

    /**
     * Re-encodes a prior assistant tool-call turn into the conversation token stream. Used when
     * replaying multi-turn history that contains a previous tool call.
     *
     * @param toolCall the tool call to encode (name + raw arguments JSON)
     */
    default List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        throw new UnsupportedOperationException(
                "Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Re-encodes a prior assistant turn that contained one or more tool calls as a <em>single</em>
     * assistant message. Implementations must emit all calls inside one header/footer pair so the
     * model does not see spurious assistant turn boundaries.
     *
     * <p>The default delegates to {@link #encodeToolCallAssistantTurn(ToolCallExtract)} for
     * single-element lists and naively concatenates individual encodings for larger lists — formats
     * that support batch tool calls should override this method.
     *
     * @param toolCalls the ordered list of tool calls from a single assistant turn
     */
    default List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) return List.of();
        if (toolCalls.size() == 1) return encodeToolCallAssistantTurn(toolCalls.get(0));
        List<Integer> tokens = new ArrayList<>();
        for (ToolCallExtract tc : toolCalls) {
            tokens.addAll(encodeToolCallAssistantTurn(tc));
        }
        return tokens;
    }

    /**
     * Encodes a tool execution result message in the model-native format.
     *
     * @param toolCallId the ID of the originating tool call (may be ignored by some formats)
     * @param toolName the name of the tool that was called
     * @param result the result content string
     */
    default List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        throw new UnsupportedOperationException(
                "Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Detects and extracts a tool call from fully decoded model response text. Returns {@link
     * Optional#empty()} when the response is a plain text answer.
     *
     * @param responseText the fully decoded response from the model
     */
    default Optional<ToolCallExtract> extractToolCall(String responseText) {
        return Optional.empty();
    }

    /**
     * Extracts ALL tool calls from a response. Models may emit multiple {@code <tool_call>} blocks
     * in a single turn (batch tool calls). The default delegates to {@link #extractToolCall} for
     * formats that do not support batch calls.
     *
     * @param responseText the fully decoded response from the model
     */
    default List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return extractToolCall(responseText).map(List::of).orElse(List.of());
    }

    /**
     * Returns the recommended default temperature for this chat format. Used when the caller has
     * not explicitly configured a temperature.
     */
    default double defaultTemperature() {
        return 0.7;
    }

    /**
     * Returns the recommended default top-p for this chat format. Used when the caller has not
     * explicitly configured a top-p value.
     */
    default double defaultTopP() {
        return 0.9;
    }

    /**
     * Stop tokens to use when tool calling is enabled. Some models (LLaMA 3.1+) use a different
     * end-of-turn token ({@code <|eom_id|>}) when emitting a tool call instead of a regular
     * response.
     */
    default Set<Integer> getToolAwareStopTokens() {
        return getStopTokens();
    }

    /**
     * Returns {@code true} when this chat format has a controllable thinking/reasoning mode that
     * {@link #encodeThinkingControl(boolean)} can toggle (e.g. Qwen3). Formats that return {@code
     * false} (the default) have no reasoning phase to switch on or off, so the {@code
     * enableThinking} flag is inert for them. Pure reasoning models that always think and offer no
     * off-switch (e.g. DeepSeek-R1) also return {@code false}.
     */
    default boolean supportsThinking() {
        return false;
    }

    /**
     * The token that closes a reasoning block in a response, or {@code -1} if this format has none.
     *
     * <p>A reasoning format's template drops earlier turns' reasoning from the history it renders;
     * a session that carries history in its cache uses this to do the same.
     */
    default int reasoningEndToken() {
        return -1;
    }

    /**
     * Returns the tokens to append immediately after the assistant header in order to control the
     * model's thinking/reasoning phase. Models that do not {@link #supportsThinking()} return an
     * empty list (the default), so callers can invoke this unconditionally.
     *
     * @param enableThinking when {@code false}, returns the model-native primer that suppresses
     *     reasoning (e.g. Qwen3's pre-closed {@code <think></think>} block); when {@code true},
     *     returns an empty list so the model decides for itself.
     */
    default List<Integer> encodeThinkingControl(boolean enableThinking) {
        return List.of();
    }

    record ChatTokens(
            String tStartHeader,
            String tEndHeader,
            String tEndOfTurn,
            String tEndOfText,
            String tEndOfTextFim) {}

    /**
     * Represents a single message in a LLM chat session.
     *
     * <p>Each message is associated with a specific role (system, user, or assistant) and contains
     * the textual content of that message.
     *
     * @param role the participant who issued the message (SYSTEM, USER, or ASSISTANT).
     * @param content the textual content of the message
     */
    record Message(Role role, String content) {}

    /**
     * Represents the role of a participant in a LLM chat conversation
     *
     * <p>There are three standard roles:
     *
     * <ul>
     *   <li><strong>SYSTEM</strong> - sets the behavior and context of the assistant at the start
     *       of the conversation.
     *   <li><strong>USER</strong> - represents input from the human user.
     *   <li><strong>ASSISTANT</strong> - represents output from the AI assistant.
     * </ul>
     *
     * @param name the string representation of the role
     */
    record Role(String name) {
        public static Role SYSTEM = new Role("system");
        public static Role USER = new Role("user");
        public static Role ASSISTANT = new Role("assistant");
        public static Role FIM_PREFIX = new ChatFormat.Role("fim_prefix");
        public static Role FIM_SUFFIX = new ChatFormat.Role("fim_suffix");
        public static Role FIM_MIDDLE = new ChatFormat.Role("fim_middle");

        @Override
        public String toString() {
            return name;
        }
    }
}
