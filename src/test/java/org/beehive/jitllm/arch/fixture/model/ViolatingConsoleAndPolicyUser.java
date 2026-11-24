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
package org.beehive.jitllm.arch.fixture.model;

import org.beehive.jitllm.Options;

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
