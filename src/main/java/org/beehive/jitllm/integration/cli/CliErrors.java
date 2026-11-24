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
package org.beehive.jitllm.integration.cli;

import java.io.IOException;

/**
 * How a command-line entry point reports a refusal it can explain.
 *
 * <p>A diagnostic ({@code [GPUL-…]} message) already says what was refused and what to do instead —
 * {@code --fp32-kv-cache}, a larger {@code --gpu-memory}, a supported mode. Printed as a stack
 * trace, that one actionable line is buried under frames the user cannot act on, so it is printed
 * alone and the process exits 1. {@code -Djitllm.stacktrace=true} keeps the trace; any other
 * failure propagates unchanged.
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
                    || Boolean.getBoolean("jitllm.stacktrace")) {
                throw failure;
            }
            System.err.println("Error: " + message);
            System.exit(1);
        }
    }
}
