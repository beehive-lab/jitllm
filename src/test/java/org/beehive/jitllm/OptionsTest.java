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
