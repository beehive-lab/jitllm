package org.beehive.jitllm.model.format;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Which LFM2 templates open the default turn with a reasoning block. */
public class Lfm2ChatFormatTest {

    /** LFM2.5-24B-A2B's generation prompt, as the GGUF template spells it. */
    @Test
    public void lfm25TemplateOpensThinking() {
        String template =
                "{%- if add_generation_prompt -%}\n"
                        + "    {{- \"<|im_start|>assistant\\n<think>\" -}}\n"
                        + "{%- endif -%}";
        assertTrue(Lfm2ChatFormat.opensThinking(template));
    }

    /** LFM2-8B-A1B's generation prompt writes the header only. */
    @Test
    public void lfm2TemplateDoesNot() {
        String template =
                "{%- if add_generation_prompt -%} {{- \"<|im_start|>assistant\\n\" -}}{%- endif -%}";
        assertFalse(Lfm2ChatFormat.opensThinking(template));
        assertFalse(Lfm2ChatFormat.opensThinking(null));
    }
}
