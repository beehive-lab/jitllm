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
package org.beehive.jitllm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * The single-instruction CLI path must not end silently when the prompt alone fills the capacity
 * {@code --max-tokens} sized: the diagnostic names both numbers and the remedy. (The session
 * already reports {@code FinishReason.CONTEXT_FULL} with zero generated tokens in that case; the
 * interactive loop reported it and the single-instruction path printed a zero-token metrics block
 * and exited 0.)
 */
public class JitllmAppContextFullTest {

    @Test
    public void theDiagnosticNamesThePromptTheCapacityAndTheRemedy() {
        String message = JitllmApp.contextFullMessage(296, 8);
        assertTrue(message, message.contains("296 tokens"));
        assertTrue(message, message.contains("--ctx-size 8"));
        assertTrue(message, message.contains("pass --ctx-size larger than the prompt"));
    }

    @Test
    public void theHelpDescribesMaxTokensAsACapacity() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        Options.printUsage(new java.io.PrintStream(out));
        String help = out.toString();
        assertTrue(help, help.contains("--max-tokens"));
        assertTrue(help, help.contains("capacity in positions, prompt plus generated tokens"));
        assertEquals(1, help.split("--max-tokens, -n").length - 1);
    }
}
