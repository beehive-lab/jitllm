package org.beehive.jllm.integration.cli;

import java.io.IOException;

/**
 * How a command-line entry point reports a refusal it can explain.
 *
 * <p>A diagnostic ({@code [GPUL-…]} message) already says what was refused and what to do instead —
 * {@code --fp32-kv-cache}, a larger {@code --gpu-memory}, a supported mode. Printed as a stack
 * trace, that one actionable line is buried under frames the user cannot act on, so it is printed
 * alone and the process exits 1. {@code -Djllm.stacktrace=true} keeps the trace; any other failure
 * propagates unchanged.
 */
public final class CliErrors {

    private CliErrors() {}

    /** An entry point body. */
    @FunctionalInterface
    public interface Body {
        void run() throws IOException;
    }

    public static void reportDiagnostics(Body body) throws IOException {
        try {
            body.run();
        } catch (RuntimeException failure) {
            String message = failure.getMessage();
            if (message == null
                    || !message.startsWith("[GPUL-")
                    || Boolean.getBoolean("jllm.stacktrace")) {
                throw failure;
            }
            System.err.println("Error: " + message);
            System.exit(1);
        }
    }
}
