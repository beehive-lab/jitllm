package org.beehive.jitllm;

import static org.junit.Assert.*;

import org.junit.Test;

public class OptionsTest {
    @Test
    public void contextCapacityDoesNotBecomeTheGenerationLimit() {
        Options options =
                Options.parseOptions(
                        new String[] {
                            "run",
                            "-m",
                            "model.gguf",
                            "-p",
                            "hi",
                            "--ctx-size",
                            "2048",
                            "--max-new-tokens",
                            "8"
                        });
        assertEquals(2048, options.contextLength());
        assertEquals(8, options.maxNewTokens());
        assertFalse(options.interactive());
    }

    @Test
    public void legacyContextLimitRetainsItsMeaning() {
        Options options =
                Options.parseOptions(
                        new String[] {"-m", "model.gguf", "-p", "hi", "--max-tokens", "128"});
        assertEquals(128, options.contextLength());
        assertEquals(128, options.maxNewTokens());
    }

    @Test
    public void chatNeedsNoPromptAndConflictsAreRejected() {
        assertTrue(Options.parseOptions(new String[] {"chat", "-m", "model.gguf"}).interactive());
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Options.parseOptions(
                                new String[] {"chat", "-m", "model.gguf", "--instruct"}));
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parseOptions(new String[] {"run", "-m", "model.gguf"}));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Options.parseOptions(
                                new String[] {
                                    "run", "-m", "model.gguf", "-p", "hi", "-c", "100", "-n", "200"
                                }));
    }
}
