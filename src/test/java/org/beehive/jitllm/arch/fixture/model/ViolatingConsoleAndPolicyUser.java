package org.beehive.jllm.arch.fixture.model;

import org.beehive.jllm.Options;

/**
 * Deliberate violator for Rules 8a and 16: library-layer code that both prints to the console and
 * depends on the CLI options record. Used only by the self-test.
 */
public class ViolatingConsoleAndPolicyUser {

    /** Rule 8a — a lower layer reaching generation policy. */
    public String describe(Options options) {
        return String.valueOf(options);
    }

    /** Rule 16 — console I/O in library code. */
    public void chatter() {
        System.out.println("this belongs behind a sink");
        System.err.print("so does this");
    }
}
