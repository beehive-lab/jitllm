package org.beehive.jllm;

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
public class JllmAppContextFullTest {

    @Test
    public void theDiagnosticNamesThePromptTheCapacityAndTheRemedy() {
        String message = JllmApp.contextFullMessage(296, 8);
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
