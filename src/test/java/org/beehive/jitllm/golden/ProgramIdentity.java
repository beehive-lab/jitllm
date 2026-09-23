package org.beehive.jllm.golden;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;

/**
 * What "identity" observes is defined in {@code verification-gates.md}: the number of task graphs,
 * the ordered task names, the grid-scheduler entry set, and a SHA-256 over the generated kernel
 * source of every task — plus the absence of any further compilation once warm-up is done.
 *
 * <h2>How the generated source is obtained</h2>
 *
 * <p>TornadoVM exposes the generated kernel through {@code TornadoExecutionPlan.withPrintKernel()},
 * which makes each backend print the source of every task it installs. That print is a single
 * {@code System.out.println(source)} per compiled task ({@code RuntimeUtilities.dumpKernel}), so a
 * {@link SourceRecorder} substituted for {@code System.out} sees exactly one string per
 * compilation, in compile order. That gives all three of: the ordered task names (parsed from the
 * kernel entry points), the per-task source hash, and a compilation counter — a task that is
 * recompiled prints again, so a count that does not move across decode <b>is</b> the "no additional
 * compilation" assertion.
 *
 * <p>{@code withPrintKernel()} must be called before the plan compiles; a plan that has already
 * compiled prints nothing, because the dump sits on the install path. That is why the gate builds
 * its own {@code TornadoExecutionPlan} over the production task graphs rather than reusing an
 * already-warmed {@code TornadoVMMasterPlan}.
 *
 * <p>{@code -Dtornado.print.kernel.dir} redirects the dump to a file instead of stdout, which would
 * make the recorder see nothing. The gate refuses to run in that case rather than passing on an
 * empty observation.
 */
public final class ProgramIdentity {

    /**
     * Kernel entry points, per backend source language: CUDA C from the CUDA backend's NVRTC path
     * ({@code extern "C" __global__ void name(}), OpenCL C ({@code __kernel void name(}), and PTX
     * ({@code.visible.entry name}) for backends that emit it directly. One alternation keeps the
     * declaration order of a module that contains several.
     */
    private static final Pattern ENTRY_POINT =
            Pattern.compile(
                    "(?:__global__\\s+void|__kernel\\s+void|\\.entry)\\s+([A-Za-z_$][A-Za-z0-9_$]*)");

    private ProgramIdentity() {}

    /** The compiled program as observed at one point in time. */
    public record Snapshot(
            int graphCount,
            List<String> taskNames,
            List<String> sourceHashes,
            List<String> gridEntries) {

        public String describe() {
            return "graphs="
                    + graphCount
                    + " tasks="
                    + taskNames.size()
                    + " gridEntries="
                    + gridEntries.size();
        }
    }

    public static Snapshot snapshot(
            int graphCount, List<String> kernelSources, GridScheduler scheduler) {
        List<String> names = new ArrayList<>();
        List<String> hashes = new ArrayList<>();
        for (String source : kernelSources) {
            names.addAll(entryPoints(source));
            hashes.add(sha256(source));
        }
        return new Snapshot(
                graphCount, List.copyOf(names), List.copyOf(hashes), gridEntries(scheduler));
    }

    /**
     * The grid-scheduler entry set, sorted so the comparison does not depend on the iteration order
     * of the {@code ConcurrentHashMap} behind {@link GridScheduler#keySet()}. The work dimensions
     * are part of the entry: a task launched with a different grid is a different program.
     */
    public static List<String> gridEntries(GridScheduler scheduler) {
        List<String> entries = new ArrayList<>();
        for (String taskName : new TreeSet<>(scheduler.keySet())) {
            WorkerGrid grid = scheduler.get(taskName);
            entries.add(
                    taskName
                            + " global="
                            + dims(grid == null ? null : grid.getGlobalWork())
                            + " local="
                            + dims(grid == null ? null : localWork(grid)));
        }
        return List.copyOf(entries);
    }

    private static long[] localWork(WorkerGrid grid) {
        try {
            return grid.getLocalWork();
        } catch (RuntimeException e) {
            // A worker grid with no local work set throws rather than returning null.
            return null;
        }
    }

    private static String dims(long[] work) {
        return work == null ? "default" : Arrays.toString(work);
    }

    /** Every kernel entry point declared by one compiled module, in declaration order. */
    public static List<String> entryPoints(String source) {
        List<String> names = new ArrayList<>();
        collect(ENTRY_POINT.matcher(source), names);
        return names.isEmpty()
                ? List.of("<unnamed:" + sha256(source).substring(0, 12) + ">")
                : names;
    }

    private static void collect(Matcher matcher, List<String> into) {
        while (matcher.find()) {
            into.add(matcher.group(1));
        }
    }

    public static String sha256(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(text.getBytes(StandardCharsets.UTF_8)));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    /** True when the dump would go to a file, where {@link SourceRecorder} cannot see it. */
    public static boolean kernelDumpRedirectedToFile() {
        return !System.getProperty("tornado.print.kernel.dir", "").isBlank();
    }

    /**
     * Stands in for {@code System.out} and keeps every kernel source TornadoVM prints, passing
     * everything else through untouched.
     */
    public static final class SourceRecorder extends PrintStream {

        private final PrintStream delegate;
        private final List<String> sources = Collections.synchronizedList(new ArrayList<>());
        private PrintStream previous;

        public SourceRecorder(PrintStream delegate) {
            super(delegate, true);
            this.delegate = Objects.requireNonNull(delegate);
        }

        public static SourceRecorder install() {
            SourceRecorder recorder = new SourceRecorder(System.out);
            recorder.previous = System.out;
            System.setOut(recorder);
            return recorder;
        }

        public void uninstall() {
            if (previous != null) {
                System.setOut(previous);
                previous = null;
            }
        }

        @Override
        public void println(String x) {
            if (isKernelSource(x)) {
                sources.add(x);
            } else {
                delegate.println(x);
            }
        }

        /** Immutable copy of what has been compiled so far, in compile order. */
        public List<String> sources() {
            synchronized (sources) {
                return List.copyOf(sources);
            }
        }

        public int count() {
            return sources.size();
        }

        private static boolean isKernelSource(String text) {
            if (text == null || text.length() < 64) {
                return false;
            }
            return text.contains("__global__ void")
                    || text.contains("__kernel void")
                    || text.contains(".entry ");
        }
    }
}
