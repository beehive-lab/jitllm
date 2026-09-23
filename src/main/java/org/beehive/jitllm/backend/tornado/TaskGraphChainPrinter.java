package org.beehive.jitllm.backend.tornado;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.beehive.jitllm.model.Model;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoTaskGraphInterface;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.common.SchedulableTask;

/**
 * Summarizes the TornadoVM task-graph chain of a plan ({@code --print-taskgraph-chain}): when each
 * graph runs, and the tasks each graph runs, in order.
 *
 * <p>Consecutive graphs that run the same tasks print once ("layer_0 … layer_15, 16 graphs"); a
 * graph that holds several layers lists one layer's tasks. Each task is one line: its name, the
 * kernel method and its worker grid, or — tagged {@code libraryTask} — the vendor library call and
 * the sizes it computes.
 *
 * <p>TornadoVM keeps no public accessor for a built graph's tasks, so the graph is reached through
 * two private fields ({@code ImmutableTaskGraph.taskGraph}, {@code TaskGraph.taskGraphImpl}) and
 * its tasks through the public {@code apply}. A read that fails prints "(unavailable)" instead of
 * failing the run.
 */
public final class TaskGraphChainPrinter {

    /** Set by the launcher's {@code --print-taskgraph-chain}. */
    public static final String PROPERTY = "jitllm.printTaskGraphChain";

    private static final Pattern LAYER_SLOT = Pattern.compile("^[lL](\\d+)_");
    private static final Pattern FAMILY_POSITION = Pattern.compile(" \\d+/\\d+$");
    private static final String INDENT = "          ";

    /**
     * Where a printed chain goes. A library logs it; the CLI entry points install a plain stderr
     * writer (Rule 16: no console I/O outside the CLI integration).
     */
    private static volatile Consumer<String> output = TaskGraphChainPrinter::log;

    private TaskGraphChainPrinter() {}

    /** One step of the schedule: a name, when it runs, and the graphs it runs in order. */
    public record Phase(String name, String when, List<Integer> graphs) {}

    /** What a plan hands over to be described. */
    public record Chain(
            String mode,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler,
            List<Phase> phases,
            Map<Integer, String> roles) {}

    /** Sends printed chains to {@code sink}; {@code null} restores the logger. */
    public static void output(Consumer<String> sink) {
        output = sink == null ? TaskGraphChainPrinter::log : sink;
    }

    private static void log(String text) {
        System.getLogger(TaskGraphChainPrinter.class.getName()).log(System.Logger.Level.INFO, text);
    }

    public static boolean requested() {
        return Boolean.getBoolean(PROPERTY);
    }

    /** Emits {@code chain} to the {@link #output} when {@link #PROPERTY} is set. */
    public static void printIfRequested(Chain chain, Model model) {
        if (!requested()) {
            return;
        }
        output.accept(render(chain, model));
    }

    // ── rendering ────────────────────────────────────────────────────────────

    /** The summary, for the printer and for tests. */
    public static String render(Chain chain, Model model) {
        List<GraphView> views = new ArrayList<>();
        for (int i = 0; i < chain.graphs().size(); i++) {
            views.add(GraphView.read(i, chain));
        }

        StringBuilder out = new StringBuilder();
        out.append("Plan  ")
                .append(model == null ? "model" : model.getModelType().name())
                .append(model == null ? "" : " " + model.weights().dataType())
                .append(" · ")
                .append(chain.mode())
                .append(" · ")
                .append(chain.graphs().size())
                .append(" graphs · CUDA graphs ")
                .append(TornadoVMMasterPlan.CUDA_GRAPHS ? "on" : "off")
                .append('\n');
        List<String[]> runs = new ArrayList<>();
        for (Phase phase : chain.phases()) {
            // Warm-up runs everything once; its sequence would only repeat the others.
            boolean everything = phase.name().equals("warm-up");
            runs.add(
                    new String[] {
                        phase.name(),
                        everything ? "every graph" : sequence(phase, chain),
                        phase.when()
                    });
        }
        int nameWidth = runs.stream().mapToInt(r -> r[0].length()).max().orElse(0);
        int sequenceWidth = runs.stream().mapToInt(r -> r[1].length()).max().orElse(0);
        for (int r = 0; r < runs.size(); r++) {
            out.append(r == 0 ? "Runs  " : "      ")
                    .append(pad(runs.get(r)[0], nameWidth + 2))
                    .append(pad(runs.get(r)[1], sequenceWidth + 2))
                    .append(runs.get(r)[2])
                    .append('\n');
        }

        for (List<GraphView> group : groups(views)) {
            out.append('\n');
            renderGroup(out, group);
        }
        return out.toString();
    }

    /** A phase's graphs as families: "activation → 16 × layers → logits". */
    private static String sequence(Phase phase, Chain chain) {
        List<String> parts = new ArrayList<>();
        String last = null;
        int count = 0;
        for (int index : phase.graphs()) {
            String family = family(chain.roles().getOrDefault(index, "graph " + index));
            if (family.equals(last)) {
                count++;
                continue;
            }
            if (last != null) parts.add(count > 1 ? count + " × " + last : last);
            last = family;
            count = 1;
        }
        if (last != null) parts.add(count > 1 ? count + " × " + last : last);
        return String.join(" → ", parts);
    }

    private static String family(String role) {
        return FAMILY_POSITION.matcher(role).replaceFirst("");
    }

    /** Consecutive graphs of one family that run the same tasks on the same kind of buffers. */
    private static List<List<GraphView>> groups(List<GraphView> views) {
        List<List<GraphView>> groups = new ArrayList<>();
        for (GraphView view : views) {
            List<GraphView> current = groups.isEmpty() ? null : groups.get(groups.size() - 1);
            if (current != null
                    && current.get(0).family.equals(view.family)
                    && current.get(0).signature.equals(view.signature)) {
                current.add(view);
            } else {
                List<GraphView> fresh = new ArrayList<>();
                fresh.add(view);
                groups.add(fresh);
            }
        }
        return groups;
    }

    private static void renderGroup(StringBuilder out, List<GraphView> group) {
        GraphView first = group.get(0);
        GraphView last = group.get(group.size() - 1);
        int n = group.size();
        String indices =
                n == 1 ? "[" + first.index + "]" : "[" + first.index + "-" + last.index + "]";
        String name = n == 1 ? first.name : first.name + " … " + last.name;
        List<String> about = new ArrayList<>();
        if (!first.family.isEmpty()) about.add(first.family);
        if (n > 1) about.add(n + " graphs");
        Set<Integer> layerCounts = new TreeSet<>();
        List<Integer> perGraph = new ArrayList<>();
        for (GraphView g : group) {
            layerCounts.add(g.layers);
            perGraph.add(g.layers);
        }
        if (layerCounts.size() > 1) {
            about.add(joinInts(perGraph, "/") + " layers");
        } else if (first.layers > 1) {
            about.add(first.layers + " layers each");
        } else if (first.family.contains("layers")) {
            about.add("1 layer each");
        }
        out.append(pad(indices, 8)).append(name);
        if (!about.isEmpty()) out.append("   (").append(String.join(", ", about)).append(')');
        out.append('\n');

        if (first.unreadable) {
            out.append(INDENT).append("(unavailable: TornadoVM internals not readable)\n");
            return;
        }
        int idWidth = first.tasks.stream().mapToInt(t -> t.id().length()).max().orElse(0);
        int targetWidth = first.tasks.stream().mapToInt(t -> t.target().length()).max().orElse(0);
        for (TaskView task : first.tasks) {
            out.append(INDENT)
                    .append(pad(task.library() ? "libraryTask" : "task", 13))
                    .append(pad(task.id(), idWidth + 2))
                    .append(pad(task.target(), targetWidth + 2))
                    .append(task.detail())
                    .append('\n');
        }
        if (first.layers > 1) {
            out.append(INDENT)
                    .append("× ")
                    .append(
                            layerCounts.size() == 1
                                    ? first.layers + " layers"
                                    : joinInts(perGraph, "/") + " layers")
                    .append(" per graph")
                    .append(first.slotPrefixes.isEmpty() ? "" : " (" + first.slotPrefixes + ")")
                    .append('\n');
        }
    }

    // ── one graph, read once ─────────────────────────────────────────────────

    private record TaskView(String id, String target, String detail, String key, boolean library) {}

    private static final class GraphView {
        int index;
        String name = "?";
        String family = "";
        boolean unreadable;
        int layers = 1;
        String slotPrefixes = "";
        final List<TaskView> tasks = new ArrayList<>();
        String signature = "";

        static GraphView read(int index, Chain chain) {
            GraphView view = new GraphView();
            view.index = index;
            view.family = family(chain.roles().getOrDefault(index, ""));
            TaskGraph taskGraph = field(chain.graphs().get(index), "taskGraph", TaskGraph.class);
            TornadoTaskGraphInterface impl =
                    taskGraph == null
                            ? null
                            : field(taskGraph, "taskGraphImpl", TornadoTaskGraphInterface.class);
            if (taskGraph != null) view.name = taskGraph.getTaskGraphName();
            if (impl == null) {
                view.unreadable = true;
                view.signature = "?" + index;
                return view;
            }
            List<SchedulableTask> tasks = new ArrayList<>();
            impl.apply(tasks::add);
            // Grouped graphs prefix every layer after the first ("l1_", ...) or every layer
            // ("L0_", ... in batched prefill); list one layer's tasks, prefix removed.
            Map<String, List<SchedulableTask>> bySlot = new LinkedHashMap<>();
            for (SchedulableTask task : tasks) {
                Matcher m = LAYER_SLOT.matcher(shortId(view.name, task.getId()));
                bySlot.computeIfAbsent(m.find() ? m.group(1) : "", k -> new ArrayList<>())
                        .add(task);
            }
            view.layers = Math.max(1, bySlot.size());
            List<String> prefixed = bySlot.keySet().stream().filter(k -> !k.isEmpty()).toList();
            if (view.layers > 1 && !prefixed.isEmpty()) {
                String letter =
                        shortId(view.name, bySlot.get(prefixed.get(0)).get(0).getId())
                                .substring(0, 1);
                view.slotPrefixes =
                        (bySlot.containsKey("") ? "first layer unprefixed, then " : "")
                                + letter
                                + prefixed.get(0)
                                + "_ … "
                                + letter
                                + prefixed.get(prefixed.size() - 1)
                                + "_";
            }
            if (!bySlot.isEmpty()) {
                for (SchedulableTask task : bySlot.values().iterator().next()) {
                    view.tasks.add(task(view.name, task, chain.scheduler()));
                }
            }

            StringBuilder signature = new StringBuilder();
            for (TaskView t : view.tasks) signature.append(t.key()).append(';');
            view.signature = signature.toString();
            return view;
        }
    }

    private static TaskView task(String graphName, SchedulableTask task, GridScheduler scheduler) {
        String fullId = shortId(graphName, task.getId());
        String id = LAYER_SLOT.matcher(fullId).replaceFirst("");
        LibraryTaskDescriptor library = libraryDescriptor(task);
        String target;
        String detail;
        if (library != null) {
            target = libraryName(library.getLibraryName()) + " " + library.getFunctionName();
            detail = libraryParameters(library);
        } else {
            target = kernelName(task);
            detail = grid(scheduler, graphName, fullId);
        }
        return new TaskView(id, target, detail, id + "=" + target + "@" + detail, library != null);
    }

    private static String libraryName(String name) {
        String last = name == null ? "" : name.substring(name.lastIndexOf('/') + 1);
        return switch (last.toLowerCase(java.util.Locale.ROOT)) {
            case "cublas" -> "cuBLAS";
            case "cudnn" -> "cuDNN";
            default -> last;
        };
    }

    /** The scalar parameters that say what a library call computes. */
    private static String libraryParameters(LibraryTaskDescriptor library) {
        Object[] p = library.getParameters();
        if (p == null) return "";
        String function = library.getFunctionName();
        // Positions from the TornadoVM wrappers: CuBlas gemm (transa, transb, m, n, k, ...) and
        // CuDnn.sdpaForward (q, k, v, o, b, h, sQ, sKv, d, scale, causal).
        if (function.startsWith("cublasGemm") && p.length >= 5) {
            return "m=" + p[2] + " n=" + p[3] + " k=" + p[4];
        }
        if (function.equals("sdpaForward") && p.length >= 11) {
            return "b="
                    + p[4]
                    + " h="
                    + p[5]
                    + " q="
                    + p[6]
                    + " kv="
                    + p[7]
                    + " d="
                    + p[8]
                    + (Boolean.TRUE.equals(p[10]) ? " causal" : "");
        }
        List<String> scalars = new ArrayList<>();
        for (Object o : p) {
            if (o instanceof Number || o instanceof Boolean) scalars.add(String.valueOf(o));
        }
        return String.join(" ", scalars);
    }

    private static String grid(GridScheduler scheduler, String graphName, String taskId) {
        if (scheduler == null) return "";
        try {
            String key = graphName + "." + taskId;
            if (!scheduler.keySet().contains(key)) return "";
            WorkerGrid grid = scheduler.get(key);
            String local = grid.getLocalWork() == null ? "auto" : dims(grid.getLocalWork());
            return dims(grid.getGlobalWork()) + " / " + local;
        } catch (RuntimeException e) {
            return "";
        }
    }

    /** {@code [2048, 1, 1]} as {@code 2048}, {@code [64, 32, 1]} as {@code 64×32}. */
    private static String dims(long[] d) {
        int n = d.length;
        while (n > 1 && d[n - 1] == 1) n--;
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < n; i++) out.append(i == 0 ? "" : "×").append(d[i]);
        return out.toString();
    }

    // ── schedule helpers ─────────────────────────────────────────────────────

    /** Indices {@code from..to} inclusive, for building phases. */
    public static List<Integer> span(int from, int to) {
        List<Integer> list = new ArrayList<>();
        for (int i = from; i <= to; i++) list.add(i);
        return list;
    }

    /** Joins index lists, for building phases. */
    @SafeVarargs
    public static List<Integer> concat(List<Integer>... parts) {
        List<Integer> list = new ArrayList<>();
        for (List<Integer> part : parts) list.addAll(part);
        return list;
    }

    /** Labels {@code count} graphs from {@code first} as "{@code family} g/count". */
    public static void label(Map<Integer, String> roles, int first, int count, String family) {
        for (int g = 0; g < count; g++) {
            roles.put(first + g, count == 1 ? family : family + " " + (g + 1) + "/" + count);
        }
    }

    static String ranges(List<Integer> graphs) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < graphs.size()) {
            int start = graphs.get(i);
            int end = start;
            while (i + 1 < graphs.size() && graphs.get(i + 1) == end + 1) {
                end = graphs.get(++i);
            }
            if (out.length() > 0) out.append(" + ");
            out.append('[').append(start).append(end == start ? "" : ".." + end).append(']');
            i++;
        }
        return out.toString();
    }

    // ── reading TornadoVM ────────────────────────────────────────────────────

    private static String shortId(String graphName, String id) {
        return id.startsWith(graphName + ".") ? id.substring(graphName.length() + 1) : id;
    }

    private static LibraryTaskDescriptor libraryDescriptor(SchedulableTask task) {
        try {
            Method m = task.getClass().getMethod("getDescriptor");
            return m.invoke(task) instanceof LibraryTaskDescriptor l ? l : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** The kernel method's name, or the task's own name when the method is not reachable. */
    private static String kernelName(SchedulableTask task) {
        try {
            Method m = task.getClass().getMethod("getMethod");
            if (m.invoke(task) instanceof Method kernel) return kernel.getName();
        } catch (ReflectiveOperationException | RuntimeException e) {
            // fall through
        }
        return task.getTaskName();
    }

    private static <T> T field(Object owner, String name, Class<T> type) {
        for (Class<?> c = owner.getClass(); c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                Object value = f.get(owner);
                return type.isInstance(value) ? type.cast(value) : null;
            } catch (NoSuchFieldException e) {
                // try the superclass
            } catch (ReflectiveOperationException | RuntimeException e) {
                return null;
            }
        }
        return null;
    }

    private static String pad(String s, int width) {
        return s.length() >= width ? s + " " : s + " ".repeat(width - s.length());
    }

    private static String joinInts(List<Integer> values, String separator) {
        List<String> parts = new ArrayList<>();
        for (int v : values) parts.add(String.valueOf(v));
        return String.join(separator, parts);
    }
}
