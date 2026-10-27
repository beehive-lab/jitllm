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
 * This is achieved by placing prepacking graphs first in the {@link TornadoExecutionPlan}'s list of
 * {@link TaskGraph}s. Each session has its own instance, in {@code TornadoWorkspace.prepacker}.
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
 * the orders differ:
 *
 * <ul>
 *   <li><b>construction:</b> the layer builders declare their inputs through {@link
 *       #configureGraphInputs}, which registers the weights to prepack; then {@link
 *       #buildTaskGraphs} builds the prepacking graphs from those registrations;
 *   <li><b>plan assembly:</b> the plan lists the prepacking graphs first, its initialization, then
 *       its inference graphs, and merges their grids into its scheduler;
 *   <li><b>execution:</b> the plan runs them once, before its warm-up, and its inference steps
 *       never select them ({@link WeightPrepackingTaskGraphs#run}).
 * </ul>
 */
// @formatter:on
public final class WeightPrepacker {

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

    /**
     * Selected weights declared by {@link #configureGraphInputs} and still waiting for {@link
     * #buildTaskGraphs} to build their prepacking graphs, by the name of the layer graph that
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
     * Builds the prepacking graphs for the weights registered since the last call, one per layer
     * graph that registered any, and returns them with their work grids; the plan then lists them
     * first and merges the grids into its scheduler. Call it once the plan's layer graphs are
     * built, since they register the weights. With no weight registered it allocates nothing and
     * returns {@link WeightPrepackingTaskGraphs#NONE}.
     *
     * <p>Each graph uploads its weights in the original layout, then, for each weight, copies it
     * into a scratch buffer and rearranges it from there back into its own device buffer: the
     * rearranging cannot be done in place, because writing the prepacked layout would overwrite
     * bytes not yet read. The prepacked buffers stay on the device for the layer graphs to consume.
     *
     * <p>All graphs created by one call share one scratch buffer, as large as the largest weight,
     * and one completion flag, which the last of them sets once every weight is prepacked; from
     * then on, any execution of these graphs skips the work.
     */
    public synchronized WeightPrepackingTaskGraphs buildTaskGraphs() {
        if (pendingWeightsByGraph.isEmpty()) {
            return WeightPrepackingTaskGraphs.NONE;
        }
        int scratchBytes = 0;
        for (List<Weight> ws : pendingWeightsByGraph.values()) {
            for (Weight w : ws) {
                scratchBytes = Math.max(scratchBytes, w.size());
            }
        }
        ByteArray scratch = new ByteArray(scratchBytes);
        IntArray done = new IntArray(1);
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        Map<String, WorkerGrid> grids = new LinkedHashMap<>();
        String firstName = null;
        int remaining = pendingWeightsByGraph.size();
        for (Map.Entry<String, List<Weight>> e : pendingWeightsByGraph.entrySet()) {
            String name = PREFIX + e.getKey();
            List<Weight> ws = e.getValue();
            Object[] bytes = ws.stream().map(Weight::bytes).toArray();
            TaskGraph g =
                    new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, bytes);
            // The first graph uploads the scratch buffer and the flag; the others consume them
            // from it, so the device holds one of each.
            if (firstName == null) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch, done);
                firstName = name;
            } else {
                g.consumeFromDevice(firstName, scratch, done);
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
                grids.put(name + "." + copy, grid((w.size() + 7) / 8));
                grids.put(
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
                grids.put(name + ".markDone", grid(1));
            }
            g.persistOnDevice(bytes);
            g.persistOnDevice(scratch, done);
            graphs.add(g.snapshot());
        }
        pendingWeightsByGraph.clear();
        registered.clear();
        return new WeightPrepackingTaskGraphs(graphs, grids);
    }

    /** A one-dimensional grid of at least {@code lanes} lanes, in whole workgroups. */
    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }
}
