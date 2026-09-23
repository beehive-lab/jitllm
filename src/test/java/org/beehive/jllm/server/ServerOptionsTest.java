package org.beehive.jllm.server;

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
        assertEquals(1, options.parallel());
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
                    {"--parallel", "0"},
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
    public void parallelServingRejectsHalfPrecisionKvCache() {
        String previous = System.getProperty("jllm.kvcache.fp16");
        try {
            System.setProperty("jllm.kvcache.fp16", "true");
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            ServerOptions.parse(
                                    new String[] {"-m", "model.gguf", "--gpu", "--parallel", "2"}));
        } finally {
            if (previous == null) System.clearProperty("jllm.kvcache.fp16");
            else System.setProperty("jllm.kvcache.fp16", previous);
        }
    }
}
