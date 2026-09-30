package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.tokenizer.Qwen35Tokenizer;

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
 * <p>The tools block comes first in the system turn and the caller's system text after it, as the
 * template orders them; consecutive tool results share one user turn ({@link #encodeToolResults},
 * inherited).
 *
 * <p>The template's reasoning effort is rendered ({@link #reasoningEffortInstructions(String)}):
 * while thinking is on it writes an instruction at the top of the system turn, {@code xhigh} unless
 * told otherwise, and that instruction changes how long the model reasons several-fold. It used to
 * be left out, and the model reasoned three to four times less than under llama.cpp or Ollama.
 *
 * <h2>Deviations, stated</h2>
 *
 * <p>The {@code <think>…</think>} block the template writes in front of every replayed assistant
 * turn is not rendered, the same as for this family's conversations without tools. Nor is the
 * {@code <think>\n} the template's generation prompt opens a thinking turn with: the model writes
 * it itself, and the integrations recognise the reasoning by it.
 */
public class Qwen35ChatFormat extends Qwen3ChatFormat {

    public Qwen35ChatFormat(Qwen35Tokenizer tokenizer, ChatTokens chatTokens) {
        super(tokenizer, chatTokens);
    }

    /** The reasoning-effort control exists wherever the thinking control does. */
    @Override
    public boolean supportsReasoningEffort() {
        return supportsThinking();
    }

    /** The template's {@code reasoning_effort|default('xhigh')}. */
    @Override
    public String defaultReasoningEffort() {
        return "xhigh";
    }

    /**
     * The template's instructions, verbatim: one for {@code xhigh} (and {@code high}, which it
     * treats as {@code xhigh}), one for {@code low}, none for {@code medium}.
     */
    @Override
    public String reasoningEffortInstructions(String effort) {
        return switch (effort) {
            case "xhigh", "high" ->
                    "Reasoning effort is set to xhigh. Please think carefully through the task,"
                            + " validate key assumptions, consider plausible alternatives, and"
                            + " prioritize correctness, consistency, and clarity in the final"
                            + " answer.";
            case "medium" -> "";
            case "low" ->
                    "Reasoning effort is set to low. Keep your thinking brief and focused, moving"
                            + " directly to the conclusion without unnecessary elaboration.";
            default ->
                    throw new IllegalArgumentException(
                            "Unexpected reasoning effort "
                                    + effort
                                    + ". Supported types are xhigh (default), medium, and low.");
        };
    }

    @Override
    protected String toolsBlock(String toolsJson) {
        return "# Tools\n\n"
                + "You have access to the following functions:\n\n"
                + "<tools>"
                + toolLines(toolsJson)
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

    /**
     * The {@code qwen35} template writes the tools block <em>first</em> and the caller's system
     * text after it: {@code <|im_start|>system\n{tools block}\n\n{system}<|im_end|>\n}.
     */
    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        return encodeToolSystemMessage(systemContent, toolsJson, "");
    }

    /**
     * As {@link #encodeToolSystemMessage(String, String)}, with the reasoning-effort instructions
     * first, where the template writes them: {@code <|im_start|>system\n{instructions}\n\n{tools
     * block}\n\n{system}<|im_end|>\n}.
     */
    @Override
    public List<Integer> encodeToolSystemMessage(
            String systemContent, String toolsJson, String reasoningInstructions) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("system\n"));
        String preamble = reasoningInstructions.isEmpty() ? "" : reasoningInstructions + "\n\n";
        tokens.addAll(templateText(preamble + toolsBlock(toolsJson)));
        if (systemContent != null && !systemContent.isBlank()) {
            tokens.addAll(tokenizer.encodeOrdinaryAsList("\n\n" + systemContent.strip()));
        }
        tokens.addAll(endOfTurn());
        return tokens;
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
        tokens.addAll(templateText(body.toString()));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return Qwen35ToolCalls.parseFirst(responseText);
    }

    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return Qwen35ToolCalls.parseAll(responseText);
    }
}
