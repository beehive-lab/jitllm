package org.beehive.jitllm.backend.tornado.ops;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.program.op.Operation;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Splits each region's operations into steps, one implementation per step, and builds them.
 *
 * <p>Walking a region from its first operation: the candidates are tried in order, and for each,
 * its fusions longest first. The first fusion that matches and that its implementation accepts
 * makes the next step. The memory check ({@link #needs}) and the plan ({@link #of}) split the
 * regions the same way, so the memory the check predicts is the memory the plan allocates.
 */
public final class WorkPlan {

    private record Chosen(Implementation implementation, Step step) {}

    private final Map<Implementation, Instance> instances;
    private final Map<Region, List<Chosen>> steps;

    private WorkPlan(Map<Implementation, Instance> instances, Map<Region, List<Chosen>> steps) {
        this.instances = instances;
        this.steps = steps;
    }

    /** {@code region} split into steps, each with the implementation that runs it. */
    private static List<Chosen> split(
            Region region, List<Implementation> candidates, Target target) {
        List<Operation> ops = region.ops();
        List<Chosen> chosen = new ArrayList<>();
        int at = 0;
        while (at < ops.size()) {
            Chosen next = null;
            for (Implementation implementation : candidates) {
                for (Fusion fusion : implementation.fusions()) {
                    var match = fusion.match(ops, at);
                    if (match.isEmpty()) {
                        continue;
                    }
                    Step step = new Step(region.layer(), region.tokens(), fusion, match.get());
                    if (implementation.accepts(step, target)) {
                        next = new Chosen(implementation, step);
                        break;
                    }
                }
                if (next != null) {
                    break;
                }
            }
            if (next == null) {
                throw new IllegalStateException(
                        "no implementation runs "
                                + ops.get(at).kind()
                                + " in layer "
                                + region.layer()
                                + " ("
                                + region.name()
                                + ")");
            }
            chosen.add(next);
            at += next.step().ops().size();
        }
        return chosen;
    }

    /** The steps of all regions, grouped by implementation. */
    private static Map<Implementation, List<Step>> byImplementation(
            Map<Region, List<Chosen>> steps) {
        Map<Implementation, List<Step>> grouped = new LinkedHashMap<>();
        steps.values()
                .forEach(
                        list ->
                                list.forEach(
                                        c ->
                                                grouped.computeIfAbsent(
                                                                c.implementation(),
                                                                i -> new ArrayList<>())
                                                        .add(c.step())));
        return grouped;
    }

    private static Map<Region, List<Chosen>> splitAll(
            List<Region> regions, List<Implementation> candidates, Target target) {
        Map<Region, List<Chosen>> steps = new LinkedHashMap<>();
        for (Region region : regions) {
            steps.put(region, split(region, candidates, target));
        }
        return steps;
    }

    /** Splits {@code regions} into steps and allocates what each implementation needs. */
    public static WorkPlan of(
            List<Region> regions,
            List<Implementation> candidates,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        Map<Region, List<Chosen>> steps = splitAll(regions, candidates, target);
        Map<Implementation, Instance> instances = new LinkedHashMap<>();
        byImplementation(steps)
                .forEach(
                        (implementation, its) ->
                                instances.put(
                                        implementation,
                                        implementation.create(its, target, context, prepacking)));
        return new WorkPlan(instances, steps);
    }

    /** {@link #of}, or {@link #narrowed} from {@code primary} where {@code primary} is not null. */
    public static WorkPlan of(
            WorkPlan primary,
            List<Region> regions,
            List<Implementation> candidates,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        return primary == null
                ? of(regions, candidates, target, context, prepacking)
                : narrowed(primary, regions, candidates, target, context, prepacking);
    }

    /**
     * {@link #of} for the narrow family ({@code PrefillFamily}): {@code regions} are those of
     * {@code primary} over fewer rows, and each implementation takes its primary instance's buffers
     * ({@link Implementation#narrowed}).
     *
     * @throws IllegalStateException if an implementation has no instance in {@code primary}
     */
    public static WorkPlan narrowed(
            WorkPlan primary,
            List<Region> regions,
            List<Implementation> candidates,
            Target target,
            KernelContext context,
            WeightPrepacking prepacking) {
        Map<Region, List<Chosen>> steps = splitAll(regions, candidates, target);
        Map<Implementation, Instance> instances = new LinkedHashMap<>();
        byImplementation(steps)
                .forEach(
                        (implementation, its) ->
                                instances.put(
                                        implementation,
                                        implementation.narrowed(
                                                primary.instanceOf(implementation.getClass()),
                                                its,
                                                target,
                                                context,
                                                prepacking)));
        return new WorkPlan(instances, steps);
    }

    /** The instance of the implementation of {@code type} this plan created. */
    private Instance instanceOf(Class<?> type) {
        for (var entry : instances.entrySet()) {
            if (entry.getKey().getClass() == type) {
                return entry.getValue();
            }
        }
        throw new IllegalStateException(
                "the primary plan has no " + type.getSimpleName() + " to narrow");
    }

    /** The device memory {@link #of} would allocate for {@code regions}, without allocating it. */
    public static BatchPrefillWorkspace needs(
            List<Region> regions, List<Implementation> candidates, Target target) {
        BatchPrefillWorkspace total = BatchPrefillWorkspace.withoutResidencyChoice(0);
        for (var entry : byImplementation(splitAll(regions, candidates, target)).entrySet()) {
            total = total.plus(entry.getKey().needs(entry.getValue(), target));
        }
        return total;
    }

    /** The fusions {@code region} splits into, in order, each with its library; for tests. */
    public static List<String> describe(
            Region region, List<Implementation> candidates, Target target) {
        List<String> names = new ArrayList<>();
        for (Chosen c : split(region, candidates, target)) {
            names.add(c.step().fusion().name() + " (" + c.implementation().library() + ")");
        }
        return names;
    }

    /** The buffers all layer graphs share, from every implementation used. */
    public Object[] buffers() {
        List<Object> buffers = new ArrayList<>();
        instances.values().forEach(instance -> buffers.addAll(List.of(instance.buffers())));
        return buffers.toArray();
    }

    /** Adds the tasks of every step of {@code region} to {@code graph}, in order. */
    public void emit(TaskGraph graph, Region region, Operands operands, WeightBindings bindings) {
        List<Chosen> chosen = steps.get(region);
        if (chosen == null) {
            throw new IllegalArgumentException("not a region of this plan: " + region.name());
        }
        for (Chosen c : chosen) {
            instances.get(c.implementation()).emit(graph, c.step(), operands, bindings);
        }
    }

    /** Adds the worker grids of every task this plan added. */
    public void addGrids(GridScheduler scheduler) {
        instances.values().forEach(instance -> instance.addGrids(scheduler));
    }

    /** Whether a step of {@code region} runs on {@code library}, for example {@code "cuBLAS"}. */
    public boolean runs(Region region, String library) {
        List<Chosen> chosen = steps.get(region);
        if (chosen == null) {
            throw new IllegalArgumentException("not a region of this plan: " + region.name());
        }
        for (Chosen c : chosen) {
            if (library.equals(c.implementation().library())) {
                return true;
            }
        }
        return false;
    }

    /** The vendor libraries used, for example {@code "cuBLAS"}, or {@code "none"}. */
    public String libraries() {
        List<String> libraries = new ArrayList<>();
        for (Implementation implementation : instances.keySet()) {
            if (!"none".equals(implementation.library())
                    && !libraries.contains(implementation.library())) {
                libraries.add(implementation.library());
            }
        }
        return libraries.isEmpty() ? "none" : String.join(", ", libraries);
    }
}
