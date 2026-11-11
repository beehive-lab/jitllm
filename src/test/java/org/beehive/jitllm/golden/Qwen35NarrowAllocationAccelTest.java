package org.beehive.jitllm.golden;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Qwen3.5-0.8B F16: the narrow prefill family's graphs allocate nothing on the device. They bind
 * every array from their twins, the primary's graphs, so an allocation in a narrow graph is an
 * array the layer class uses without declaring it.
 *
 * <p>Example: the full-attention layers' rotation tables, once only the rope task's arguments, were
 * allocated again by each of the six narrow full-attention graphs (twelve copies).
 *
 * <p>TornadoVM logs each allocation before the tasks of the graph it serves when {@code
 * tornado.print.bytecodes} is set before TornadoVM loads: this class sets it statically, and
 * surefire runs each accelerator test class in its own JVM.
 */
public class Qwen35NarrowAllocationAccelTest {

    static {
        System.setProperty("tornado.print.bytecodes", "True");
    }

    private static final Pattern ANSI = Pattern.compile("\u001B\\[[0-9;]*m");
    private static final Pattern ALLOC = Pattern.compile(" ALLOC (\\S+) .*size=(\\d+)");
    private static final Pattern LAUNCH = Pattern.compile(" LAUNCH\\s+task (\\S+?)\\.");

    @Test
    public void nativeLibraries() throws Exception {
        assertNarrowGraphsAllocateNothing(true);
    }

    @Test
    public void jitKernels() throws Exception {
        assertNarrowGraphsAllocateNothing(false);
    }

    private static void assertNarrowGraphsAllocateNothing(boolean nativeLibraries)
            throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-F16.gguf");
        assumeTrue("model absent: " + file, Files.isRegularFile(file));
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", "512");
        System.setProperty(
                ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, String.valueOf(nativeLibraries));
        Model model = KvPrecisionHarness.load(file, 1024, true);
        // One chunk on the narrow family at position 0; a full chunk, then one at position 512.
        for (int tokens : new int[] {40, 560}) {
            List<String> log = new ArrayList<>();
            try {
                capture(log, () -> run(model, tokens));
            } catch (UnsupportedOperationException refused) {
                assumeTrue("no native libraries here: " + refused.getMessage(), !nativeLibraries);
                throw refused;
            }
            List<String> narrowAllocs = new ArrayList<>();
            List<String> pending = new ArrayList<>();
            int allocs = 0;
            int narrowLaunches = 0;
            for (String raw : log) {
                String line = ANSI.matcher(raw).replaceAll("");
                Matcher a = ALLOC.matcher(line);
                if (a.find()) {
                    allocs++;
                    pending.add(a.group(1) + " " + a.group(2) + " B");
                    continue;
                }
                Matcher l = LAUNCH.matcher(line);
                if (l.find()) {
                    boolean narrow = l.group(1).contains("Narrow");
                    if (narrow) {
                        narrowLaunches++;
                        pending.forEach(p -> narrowAllocs.add(l.group(1) + ": " + p));
                    }
                    pending.clear();
                }
            }
            assertTrue("bytecode logging is on (no ALLOC logged)", allocs > 0);
            assertTrue("a narrow graph ran for " + tokens + " tokens", narrowLaunches > 0);
            assertTrue(
                    "narrow graphs allocated " + narrowAllocs + " for " + tokens + " tokens",
                    narrowAllocs.isEmpty());
        }
    }

    /** The batched prefill of {@code tokens} tokens and two decode steps, FP16 cache. */
    private static void run(Model model, int tokens) {
        List<Integer> text = KvPrecisionHarness.longPrompt(model, tokens * 2);
        State fresh = model.createNewState();
        int size = PromptIngestion.of(fresh, text, 0).firstIndex() == 1 ? tokens + 1 : tokens;
        KvPrecisionHarness.run(
                model, true, DataType.F16, 512, new ArrayList<>(text.subList(0, size)), 2, null);
    }

    /** Runs {@code body} with standard output also collected, line by line, into {@code lines}. */
    private static void capture(List<String> lines, Runnable body) {
        PrintStream original = System.out;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        System.setOut(new PrintStream(bytes, true, StandardCharsets.UTF_8));
        try {
            body.run();
        } finally {
            System.out.flush();
            System.setOut(original);
            lines.addAll(List.of(bytes.toString(StandardCharsets.UTF_8).split("\n")));
        }
    }
}
