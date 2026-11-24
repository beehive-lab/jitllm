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
package org.beehive.jitllm.server;

import static org.junit.Assert.*;

import org.junit.Test;

public class ServerOptionsTest {
    @Test
    public void serverSettingsAreIndependentOfTerminalPromptValidation() {
        var options =
                ServerOptions.parse(
                        new String[] {
                            "serve",
                            "--model",
                            "model.gguf",
                            "--ctx-size",
                            "4096",
                            "--host",
                            "127.0.0.2",
                            "--port",
                            "0"
                        });
        assertEquals(4096, options.model().modelOptions().contextLength());
        assertEquals("127.0.0.2", options.host());
        assertEquals(0, options.port());
        assertEquals(1, options.batchSlots());
        assertFalse(options.continuousBatching());
    }

    @Test
    public void defaultsBindToLoopbackAndKeepLegacyContextAlias() {
        assertEquals("127.0.0.1", ServerOptions.parse(new String[] {"-m", "model.gguf"}).host());
        assertEquals(
                1024,
                ServerOptions.parse(new String[] {"-m", "model.gguf", "--max-tokens", "1024"})
                        .model()
                        .contextLength());
    }

    @Test
    public void contextDefaultsToTheModelsOwnAndAcceptsCtxAliases() {
        assertEquals(
                0, ServerOptions.parse(new String[] {"-m", "model.gguf"}).model().contextLength());
        var options =
                ServerOptions.parse(
                        new String[] {"--model", "model.gguf", "--port", "8090", "--ctx", "8192"});
        assertEquals(8192, options.model().contextLength());
        assertEquals(8090, options.port());
        assertEquals(
                2048,
                ServerOptions.parse(new String[] {"-m", "model.gguf", "--context-length", "2048"})
                        .model()
                        .contextLength());
    }

    @Test
    public void rejectsUnknownOptionsInvalidCapacityAndDuplicateContext() {
        for (String[] extra :
                new String[][] {
                    {"--temperature", "1"},
                    {"--ctx-size", "-1"},
                    {"--port", "65536"},
                    {"--continuous-batching", "1"},
                    {"--prefix-cache-entries", "4"},
                    {"--ctx-size", "100", "--max-tokens", "200"}
                }) {
            String[] args = new String[extra.length + 2];
            args[0] = "--model";
            args[1] = "model.gguf";
            System.arraycopy(extra, 0, args, 2, extra.length);
            assertThrows(IllegalArgumentException.class, () -> ServerOptions.parse(args));
        }
    }

    @Test
    public void continuousBatchingTakesEitherCache() {
        String fp32 = org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY;
        String previous = System.getProperty(fp32);
        try {
            System.clearProperty(fp32);
            assertEquals(
                    2,
                    ServerOptions.parse(
                                    new String[] {
                                        "-m", "model.gguf", "--gpu", "--continuous-batching", "2"
                                    })
                            .batchSlots());
            assertEquals(
                    2,
                    ServerOptions.parse(
                                    new String[] {
                                        "-m",
                                        "model.gguf",
                                        "--gpu",
                                        "--continuous-batching",
                                        "2",
                                        "--fp32-kv-cache"
                                    })
                            .batchSlots());
        } finally {
            if (previous == null) System.clearProperty(fp32);
            else System.setProperty(fp32, previous);
        }
    }

    @Test
    public void theRemovedFp16FlagIsRefusedWithTheMigration() {
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                ServerOptions.parse(
                                        new String[] {"-m", "model.gguf", "--fp16-kv-cache"}));
        assertTrue(refused.getMessage(), refused.getMessage().contains("--fp32-kv-cache"));
    }
}
