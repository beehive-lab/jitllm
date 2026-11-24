/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
