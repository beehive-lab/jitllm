package org.beehive.jitllm;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.JitllmApp.EntryPoint;
import org.junit.Test;

/**
 * A native image has one entry point, {@link JitllmApp}, so it dispatches the subcommands the
 * {@code jitllm} launcher would have mapped to another main class.
 */
public class JitllmAppEntryPointTest {

    @Test
    public void serveStartsTheServer() {
        assertEquals(
                EntryPoint.SERVE, JitllmApp.entryPoint(new String[] {"serve", "-m", "m.gguf"}));
        assertEquals(EntryPoint.SERVE, JitllmApp.entryPoint(new String[] {"--server", "-m", "x"}));
        assertEquals(EntryPoint.SERVE, JitllmApp.entryPoint(new String[] {"--serve"}));
    }

    @Test
    public void benchStartsTheBenchmark() {
        assertEquals(
                EntryPoint.BENCH, JitllmApp.entryPoint(new String[] {"bench", "-m", "m.gguf"}));
    }

    @Test
    public void everythingElseRunsTheApplication() {
        assertEquals(EntryPoint.RUN, JitllmApp.entryPoint(new String[] {}));
        assertEquals(
                EntryPoint.RUN, JitllmApp.entryPoint(new String[] {"-m", "m.gguf", "-p", "hi"}));
        assertEquals(EntryPoint.RUN, JitllmApp.entryPoint(new String[] {"--instruct", "serve"}));
    }
}
