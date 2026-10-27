package org.beehive.jitllm.backend.tornado.plan.prepacking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.kernels.PackedRepackKernels;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Prepacks, on the GPU and before inference, the weights a plan's {@link WeightPrepacking} selects.
 * Each session has its own instance, in {@code TornadoWorkspace.prepacker}.
 *
 * <p>For each layer graph that reads selected weights, a separate prepacking graph uploads their
 * original GGUF bytes and rearranges them on the GPU into the prepacked layout. The layer graph
 * then consumes those prepacked device buffers directly, with no host-to-device transfer of its
 * own. The weights on the host are never changed.
 *
 * <pre>
 *   host tensor (original layout)
 *        |  upload, by the prepacking graph
 *        v
 *   device buffer (original layout) --copy--> scratch buffer
 *                                                  |  rearrange
 *        +-----------------------------------------+
 *        v
 *   device buffer (prepacked layout) ----> consumed by the layer graph
 * </pre>
 *
 * <p>TornadoVM graphs only declare transfers and kernels; nothing moves until a graph executes. So
 * the layer builders declare their inputs through {@link #configureGraphInputs}, {@link
 * #appendGraphs} then creates the prepacking graphs, and {@link #run} executes them before the
 * layer graphs run. A completion flag on the device stops the weights from being rearranged again
 * if those graphs execute later.
 */
// @formatter:on
public final class WeightPrepacker {

    private static final System.Logger LOGGER = System.getLogger(WeightPrepacker.class.getName());

    /** Prefix of a prepacking graph's name: {@code prepack_layer_3} serves {@code layer_3}. */
    private static final String PREFIX = "prepack_";

    /** A selected weight waiting for its prepacking graph. */
    private record Weight(ByteArray bytes, int rows, int cols, boolean q4) {

        /**
         * Bytes of the weight: GGUF stores each row in blocks of 32 values, 18 bytes per Q4_0 block
         * (a 2-byte scale and 32 four-bit values) and 34 per Q8_0 block (a 2-byte scale and 32
         * bytes).
         */
        int size() {
            return rows * (cols / 32) * (q4 ? 18 : 34);
        }
    }

    // @formatter:off
    /**
     * Identifies the prepacking graphs {@link #appendGraphs} added to an execution plan, for {@link
     * #run}.
     *
     * <p>For example, after {@link #appendGraphs} on a plan with two layer graphs:
     *
     * <pre>
     *   index   graph
     *   0       layer_0
     *   1       layer_1
     *   2       prepack_layer_0     firstGraphIndex = 2
     *   3       prepack_layer_1     graphCount      = 2
     * </pre>
     *
     * @param firstGraphIndex position of the first prepacking graph in the plan's graph list
     * @param graphCount number of prepacking graphs, which follow one another
     * @param scheduler the work grids their kernels run with
     */
    // @formatter:on
    public record PrepackGraphs(int firstGraphIndex, int graphCount, GridScheduler scheduler) {}

    /**
     * Selected weights declared by {@link #configureGraphInputs} and still waiting for {@link
     * #appendGraphs} to create their prepacking graphs, by the name of the layer graph that
     * consumes them, in the order the layer graphs were built.
     */
    private final Map<String, List<Weight>> pendingWeightsByGraph = new LinkedHashMap<>();

    /**
     * The weights in {@link #pendingWeightsByGraph}, by identity. A weight registered twice for one
     * plan would be rearranged twice in the same device buffer, which scrambles it, so a second
     * registration is refused. A later plan has device buffers of its own and may register it
     * again.
     */
    private final Set<TornadoTensor> registered =
            Collections.newSetFromMap(new IdentityHashMap<>());

    /**
     * Declares how {@code graph} obtains {@code objects}, its inputs. No data moves during this
     * call: the transfers happen when the plan executes.
     *
     * <p>A selected weight is registered for a prepacking graph, and {@code graph} is declared to
     * consume its prepacked device buffer from that graph. Every other input, a tensor or a plain
     * array, is declared for transfer from the host on the graph's first execution.
     *
     * @throws IllegalStateException if a selected weight is already registered for this plan
     */
    public synchronized void configureGraphInputs(
            TaskGraph graph, WeightPrepacking prepacking, Object... objects) {
        List<Object> plain = new ArrayList<>();
        List<Object> prepacked = new ArrayList<>();
        for (Object o : objects) {
            if (o instanceof TornadoTensor t && prepacking.prepacked(t)) {
                if (!registered.add(t)) {
                    throw new IllegalStateException(
                            "weight already registered for prepacking, again by graph "
                                    + graph.getTaskGraphName());
                }
                // GGUF lists dimensions inputs first: dimension 0 is a row's inputs, 1 the rows.
                Weight w =
                        new Weight(
                                t.asByteArray(),
                                Math.toIntExact(t.shape().dimension(1)),
                                Math.toIntExact(t.shape().dimension(0)),
                                t.dataType() == DataType.Q4_0);
                prepacked.add(w.bytes());
                pendingWeightsByGraph
                        .computeIfAbsent(graph.getTaskGraphName(), k -> new ArrayList<>())
                        .add(w);
            } else if (o instanceof TornadoTensor t) {
                plain.add(t.deviceArray());
            } else {
                plain.add(o);
            }
        }
        if (!plain.isEmpty()) {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, plain.toArray());
        }
        if (!prepacked.isEmpty()) {
            graph.consumeFromDevice(PREFIX + graph.getTaskGraphName(), prepacked.toArray());
        }
    }

    /** {@code objects} with every {@link TornadoTensor} replaced by its device array. */
    public static Object[] deviceArrays(Object... objects) {
        Object[] arrays = new Object[objects.length];
        for (int i = 0; i < objects.length; i++) {
            arrays[i] = objects[i] instanceof TornadoTensor t ? t.deviceArray() : objects[i];
        }
        return arrays;
    }

    /**
     * Creates the prepacking graphs for the weights registered since the last call, and appends
     * them to {@code graphs}, with their work grids in {@code scheduler}.
     *
     * <p>Each graph uploads its weights in the original layout, then, for each weight, copies it
     * into a scratch buffer and rearranges it from there back into its own device buffer: the
     * rearranging cannot be done in place, because writing the prepacked layout would overwrite
     * bytes not yet read. The prepacked buffers stay on the device for the layer graphs to consume.
     *
     * <p>All graphs created by one call share one scratch buffer, as large as the largest weight,
     * and one completion flag.
     *
     * @return where the graphs are in {@code graphs}, or {@code null} when no weight was registered
     */
    public synchronized PrepackGraphs appendGraphs(
            List<ImmutableTaskGraph> graphs, GridScheduler scheduler) {
        if (pendingWeightsByGraph.isEmpty()) {
            return null;
        }
        int scratchBytes = 0;
        for (List<Weight> ws : pendingWeightsByGraph.values()) {
            for (Weight w : ws) {
                scratchBytes = Math.max(scratchBytes, w.size());
            }
        }
        ByteArray scratch = new ByteArray(scratchBytes);
        IntArray done = new IntArray(1);
        List<String> names = new ArrayList<>();
        int first = graphs.size();
        int remaining = pendingWeightsByGraph.size();
        for (Map.Entry<String, List<Weight>> e : pendingWeightsByGraph.entrySet()) {
            String name = PREFIX + e.getKey();
            List<Weight> ws = e.getValue();
            Object[] bytes = ws.stream().map(Weight::bytes).toArray();
            TaskGraph g =
                    new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, bytes);
            // The first graph uploads the scratch buffer and the flag; the others consume them
            // from it, so the device holds one of each.
            if (names.isEmpty()) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch, done);
            } else {
                g.consumeFromDevice(names.get(0), scratch, done);
            }
            for (int i = 0; i < ws.size(); i++) {
                Weight w = ws.get(i);
                String copy = "copy" + i;
                String prepack = "prepack" + i;
                g.task(
                        copy,
                        PackedRepackKernels::copyOnce,
                        new KernelContext(),
                        w.bytes(),
                        scratch,
                        w.size(),
                        done);
                if (w.q4()) {
                    g.task(
                            prepack,
                            PackedRepackKernels::repackQ4_0Once,
                            new KernelContext(),
                            scratch,
                            w.bytes(),
                            w.rows(),
                            w.cols(),
                            done);
                } else {
                    g.task(
                            prepack,
                            PackedRepackKernels::repackQ8_0Once,
                            new KernelContext(),
                            scratch,
                            w.bytes(),
                            w.rows(),
                            w.cols(),
                            done);
                }
                // The rearranging kernels run a fixed number of lanes per 128 x 64 group.
                int groups = (w.rows() / 128) * (w.cols() / 64);
                scheduler.addWorkerGrid(name + "." + copy, grid((w.size() + 7) / 8));
                scheduler.addWorkerGrid(
                        name + "." + prepack,
                        grid(
                                groups
                                        * (w.q4()
                                                ? PackedRepackKernels.Q4_0_LANES_PER_TILE
                                                : PackedRepackKernels.Q8_0_LANES_PER_TILE)));
            }
            if (--remaining == 0) {
                // Set the flag only after every weight is prepacked: from then on the copy and
                // rearrange kernels skip their work, so a later execution of any prepacking graph
                // cannot rearrange prepacked bytes again, which would scramble them.
                g.task("markDone", PackedRepackKernels::markDone, new KernelContext(), done);
                scheduler.addWorkerGrid(name + ".markDone", grid(1));
            }
            g.persistOnDevice(bytes);
            g.persistOnDevice(scratch, done);
            graphs.add(g.snapshot());
            names.add(name);
        }
        int count = pendingWeightsByGraph.size();
        pendingWeightsByGraph.clear();
        registered.clear();
        return new PrepackGraphs(first, count, scheduler);
    }

    /**
     * Executes the prepacking graphs of {@code plan} before inference: each uploads its weights and
     * rearranges them into the prepacked layout. Afterwards the plan selects all its graphs again.
     * Does nothing when {@code prepack} is {@code null}.
     */
    public static void run(TornadoExecutionPlan plan, PrepackGraphs prepack) {
        if (prepack == null) {
            return;
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < prepack.graphCount(); i++) {
            plan.withGraph(prepack.firstGraphIndex() + i)
                    .withGridScheduler(prepack.scheduler())
                    .execute();
        }
        LOGGER.log(
                System.Logger.Level.INFO,
                String.format(
                        "weights prepacked on the GPU in %.1f ms (%d graphs)",
                        (System.nanoTime() - t0) / 1e6, prepack.graphCount()));
        plan.withAllGraphs();
    }

    /** A one-dimensional grid of at least {@code lanes} lanes, in whole workgroups. */
    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }
}
