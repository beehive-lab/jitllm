package org.beehive.jitllm.golden;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * A plan 512 wide reading its weights in place ({@link UnifiedMemoryWeights}): no graph uploads a
 * weight mapped from the model file, the narrow family's graphs included; each binds it in place.
 * With the weights copied, the same check finds them uploaded, so it sees what it looks for.
 *
 * <p>Example: Qwen3-0.6B F16, a prompt of 600 tokens, a full chunk and then 88 tokens on the narrow
 * graphs, and 16 decode steps: no transfer names a mapped weight.
 *
 * <p>TornadoVM logs each binding and transfer when {@code tornado.print.bytecodes} is set before it
 * loads: this class sets it statically, and surefire runs each accelerator test class in its own
 * JVM.
 */
public class UnifiedMemoryNarrowTraceAccelTest {

    static {
        System.setProperty("tornado.print.bytecodes", "True");
    }

    private static final Pattern ANSI = Pattern.compile("\u001B\\[[0-9;]*m");
    private static final Pattern TOUCH =
            Pattern.compile(" (ALLOC|TRANSFER_HOST_TO_DEVICE_\\w+) .*?([\\w.]+Array@[0-9a-f]+)");
    private static final Pattern LAUNCH = Pattern.compile(" LAUNCH\\s+task (\\S+?)\\.");

    @Test
    public void qwen3F16() throws Exception {
        check(Fixture.QWEN3_0_6B_F16);
    }

    @Test
    public void llamaQ8_0() throws Exception {
        check(Fixture.LLAMA_3_2_1B_Q8_0);
    }

    private static void check(Fixture fixture) throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not share memory with the host",
                UnifiedMemoryWeights.deviceSupports());
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", "512");
        System.setProperty("jitllm.nativeLibraries", "false");
        Model model = KvPrecisionHarness.load(file, 1024, true);
        Set<String> mapped = mappedWeights(model.weights());
        assertTrue("mapped weights found", !mapped.isEmpty());
        List<Integer> prompt = KvPrecisionHarness.longPrompt(model, 600);
        for (boolean inPlace : new boolean[] {true, false}) {
            System.setProperty(UnifiedMemoryWeights.PROPERTY, Boolean.toString(inPlace));
            List<String> log = new ArrayList<>();
            capture(
                    log,
                    () -> KvPrecisionHarness.run(model, true, DataType.F16, 512, prompt, 16, null));
            // Each graph that reads a weight in place binds it (an ALLOC, no copy), the narrow and
            // decode graphs included; only a copied weight is transferred.
            Set<String> uploaded = new HashSet<>();
            Map<String, Integer> byGraph = new TreeMap<>();
            List<String> pending = new ArrayList<>();
            int narrowLaunches = 0;
            for (String raw : log) {
                String line = ANSI.matcher(raw).replaceAll("");
                Matcher t = TOUCH.matcher(line);
                if (t.find() && mapped.contains(t.group(2))) {
                    pending.add(t.group(1));
                    if (t.group(1).startsWith("TRANSFER")) {
                        uploaded.add(t.group(2));
                    }
                }
                Matcher l = LAUNCH.matcher(line);
                if (l.find()) {
                    String graph = l.group(1).replaceAll("_\\d+$", "_N");
                    pending.forEach(kind -> byGraph.merge(kind + " " + graph, 1, Integer::sum));
                    pending.clear();
                    if (l.group(1).contains("Narrow")) {
                        narrowLaunches++;
                    }
                }
            }
            System.out.printf(
                    "[um] %s in place %s: %d of %d mapped weights uploaded, %d narrow launches; %s%n",
                    fixture, inPlace, uploaded.size(), mapped.size(), narrowLaunches, byGraph);
            assertTrue(fixture + ": a narrow graph ran", narrowLaunches > 0);
            if (inPlace) {
                assertTrue(fixture + ": in place, yet uploaded " + uploaded, uploaded.isEmpty());
            } else {
                assertTrue(fixture + ": copied, yet no weight uploaded", !uploaded.isEmpty());
            }
        }
        System.clearProperty(UnifiedMemoryWeights.PROPERTY);
    }

    /** The arrays of {@code weights} mapped from the model file, as TornadoVM's log names them. */
    private static Set<String> mappedWeights(Object weights) throws IllegalAccessException {
        Set<String> names = new HashSet<>();
        for (Class<?> c = weights.getClass(); c != Object.class; c = c.getSuperclass()) {
            for (Field f : c.getDeclaredFields()) {
                if (Modifier.isStatic(f.getModifiers())) {
                    continue;
                }
                f.setAccessible(true);
                Object v = f.get(weights);
                if (v instanceof Object[] values) {
                    for (Object e : values) {
                        addIfMapped(names, e);
                    }
                } else {
                    addIfMapped(names, v);
                }
            }
        }
        return names;
    }

    private static void addIfMapped(Set<String> names, Object value) {
        Object array = value instanceof TornadoTensor t ? t.deviceArray() : value;
        if (array instanceof TornadoNativeArray a && a.getSegment().isMapped()) {
            names.add(a.getClass().getName() + "@" + Integer.toHexString(a.hashCode()));
        }
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
