package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jllm.tokenizer.Qwen35Tokenizer;

/**
 * The chat format for the {@code qwen35} architecture.
 *
 * <p>Everything about the turn structure is Qwen 3's and is inherited: ChatML headers, {@code
 * <|im_end|>}, the {@code <think>} control block, tool results delivered as a user turn wrapping
 * {@code <tool_response>}. What changed between the generations is <b>how a tool call is
 * written</b> — Qwen 3 put a JSON object inside {@code <tool_call>}, and Qwen 3.5 puts nested
 * pseudo-XML there instead:
 *
 * <pre>
 *   &lt;tool_call&gt;
 *   &lt;function=get_weather&gt;
 *   &lt;parameter=location&gt;
 *   Boston
 *   &lt;/parameter&gt;
 *   &lt;/function&gt;
 *   &lt;/tool_call&gt;
 * </pre>
 *
 * <p>Inheriting Qwen 3's version would have produced a model that answers, converses and reasons
 * correctly and gets tool calling silently wrong in both directions: prompted for a format it was
 * not trained on, and parsed for one it does not emit. The wire format itself lives in {@link
 * Qwen35ToolCalls}; this class is where the turn is assembled.
 *
 * <p>The instructions below are the ones in this family's own chat template, kept close to its
 * wording — including the reminder that reasoning may precede a call but not follow it, which is
 * the one the model was trained against.
 *
 * <h2>One deviation, stated</h2>
 *
 * <p>The template puts <b>consecutive</b> tool results in a single user turn, one {@code
 * <tool_response>} block after another. {@code ConversationEncoder} calls {@link
 * #encodeToolResultTurn} once per result, so several results in a row become several user turns.
 * Identical for one result, which is the common case; merging them would mean giving the shared
 * encoder a batched entry point, and that changes every family rather than this one.
 */
public class Qwen35ChatFormat extends Qwen3ChatFormat {

    public Qwen35ChatFormat(Qwen35Tokenizer tokenizer, ChatTokens chatTokens) {
        super(tokenizer, chatTokens);
    }

    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return "\n\n# Tools\n\n"
                + "You have access to the following functions:\n\n"
                + "<tools>\n"
                + toolsJson
                + "\n</tools>\n\n"
                + "If you choose to call a function ONLY reply in the following format with NO"
                + " suffix:\n\n"
                + "<tool_call>\n"
                + "<function=example_function_name>\n"
                + "<parameter=example_parameter_1>\n"
                + "value_1\n"
                + "</parameter>\n"
                + "<parameter=example_parameter_2>\n"
                + "This is the value for the second parameter\n"
                + "that can span\n"
                + "multiple lines\n"
                + "</parameter>\n"
                + "</function>\n"
                + "</tool_call>\n\n"
                + "<IMPORTANT>\n"
                + "Reminder:\n"
                + "- Function calls MUST follow the specified format: an inner"
                + " <function=...></function> block must be nested within <tool_call></tool_call>"
                + " XML tags\n"
                + "- Required parameters MUST be specified\n"
                + "- You may provide optional reasoning for your function call in natural language"
                + " BEFORE the function call, but NOT after\n"
                + "- If there is no function call available, answer the question like normal with"
                + " your current knowledge and do not tell the user about function calls\n"
                + "</IMPORTANT>";
    }

    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        return encodeToolCallAssistantTurn(List.of(toolCall));
    }

    /**
     * One assistant turn carrying every call, each in its own {@code <tool_call>} block.
     *
     * <p>Consecutive blocks are separated by a newline, as the template writes them, and the whole
     * turn ends at {@code <|im_end|>} — an assistant turn holding calls has no other content here,
     * because the engine records the calls and not the prose that preceded them.
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        StringBuilder body = new StringBuilder();
        for (int i = 0; i < toolCalls.size(); i++) {
            if (i > 0) {
                body.append('\n');
            }
            body.append("<tool_call>\n")
                    .append(Qwen35ToolCalls.renderFunctionBlock(toolCalls.get(i)))
                    .append("\n</tool_call>");
        }
        tokens.addAll(tokenizer.encodeOrdinaryAsList(body.toString()));
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return Qwen35ToolCalls.parseFirst(responseText);
    }

    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return Qwen35ToolCalls.parseAll(responseText);
    }
}
