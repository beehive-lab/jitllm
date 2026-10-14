package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
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
 *   <li><b>The reasoning block is opened by the prompt.</b> The template's generation prompt is
 *       {@code <|im_start|>assistant\n<think>}, so asking for thinking writes {@code <think>} after
 *       the header rather than leaving it to the model.
 *   <li><b>Its sampling defaults</b>, from {@code general.sampling.temp} in the file.
 * </ul>
 *
 * <h2>Deviations, stated</h2>
 *
 * <p>The template has no switch that turns reasoning off. Asking for it off writes the closed
 * {@code <think></think>} block Qwen 3's template uses for the same purpose, which the model was
 * not trained against and may ignore.
 *
 * <p>Tool calling is not offered. The template writes calls as a Python-style list between {@code
 * <|tool_call_start|>} and {@code <|tool_call_end|>}, which no parser here reads yet; the inherited
 * check for Qwen's {@code <tool_call>} token already answers no for this vocabulary.
 */
public class Lfm2ChatFormat extends Qwen3ChatFormat {

    private final int startOfText;

    public Lfm2ChatFormat(Lfm2Tokenizer tokenizer, ChatTokens chatTokens) {
        super(tokenizer, chatTokens);
        this.startOfText = tokenizer.getSpecialTokens().getOrDefault("<|startoftext|>", -1);
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

    /** {@code general.sampling.temp} in the LFM2.5 files. */
    @Override
    public double defaultTemperature() {
        return 0.1;
    }
}
