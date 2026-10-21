package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.PackedRepackKernels;
import org.beehive.jitllm.backend.tornado.tensor.PackedTiles;
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
 * Packing on the GPU at every load: a packed projection ({@link PackedTiles}) is uploaded as the
 * file stores it and repacked on the device, so the host never holds or writes the packed form.
 *
 * <p>One instance per session, on its workspace. A layer graph hands its weights to {@link
 * #upload}, packed tensors among them: those are uploaded by a repack graph of their own, {@code
 * packedRepack_<layer graph>}, which the layer graph consumes them from. {@link #appendGraphs} adds
 * those graphs to the plan and {@link #run} runs them once, before the plan's warm-up, so no layer
 * graph ever computes on the file's bytes. Each repack graph copies a weight into one scratch
 * buffer shared by the plan and repacks it from there over the weight's own device buffer; a flag
 * on the device makes a later run of any of them a no-op.
 */
// @formatter:on
public final class PackedRepack {

    private static final System.Logger LOGGER = System.getLogger(PackedRepack.class.getName());

    private static final String PREFIX = "packedRepack_";

    private record Weight(ByteArray bytes, int rows, int cols, boolean q4) {
        int size() {
            return rows * (cols / 32) * (q4 ? 18 : 34);
        }
    }

    /** The repack graphs appended to one plan: indices {@code first .. first + count - 1}. */
    public record Graphs(int first, int count, GridScheduler scheduler) {}

    /** Layer graph name to the registered weights it consumes, not yet given repack graphs. */
    private final Map<String, List<Weight>> owned = new LinkedHashMap<>();

    /**
     * Uploads {@code objects} to {@code graph} once ({@code FIRST_EXECUTION}): device arrays as
     * they are, and {@link TornadoTensor}s as their device arrays, except packed ones, which {@code
     * graph} consumes from its repack graph instead.
     */
    public synchronized void upload(TaskGraph graph, Object... objects) {
        List<Object> plain = new ArrayList<>();
        List<Object> repacked = new ArrayList<>();
        for (Object o : objects) {
            if (o instanceof TornadoTensor t && t.isPacked()) {
                PackedTiles tiles = t.packedTiles();
                Weight w =
                        new Weight(
                                t.asByteArray(),
                                tiles.rows(),
                                tiles.cols(),
                                t.dataType() == DataType.Q4_0);
                repacked.add(w.bytes());
                owned.computeIfAbsent(graph.getTaskGraphName(), k -> new ArrayList<>()).add(w);
            } else if (o instanceof TornadoTensor t) {
                plain.add(t.deviceArray());
            } else {
                plain.add(o);
            }
        }
        if (!plain.isEmpty()) {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, plain.toArray());
        }
        if (!repacked.isEmpty()) {
            graph.consumeFromDevice(PREFIX + graph.getTaskGraphName(), repacked.toArray());
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
     * Appends to {@code graphs} the repack graphs of the layer graphs built since the last call,
     * with their grids in {@code scheduler}; null when there is nothing to repack.
     */
    public synchronized Graphs appendGraphs(
            List<ImmutableTaskGraph> graphs, GridScheduler scheduler) {
        if (owned.isEmpty()) {
            return null;
        }
        int scratchBytes = 0;
        for (List<Weight> ws : owned.values()) {
            for (Weight w : ws) {
                scratchBytes = Math.max(scratchBytes, w.size());
            }
        }
        ByteArray scratch = new ByteArray(scratchBytes);
        IntArray done = new IntArray(1);
        List<String> names = new ArrayList<>();
        int first = graphs.size();
        int remaining = owned.size();
        for (Map.Entry<String, List<Weight>> e : owned.entrySet()) {
            String name = PREFIX + e.getKey();
            List<Weight> ws = e.getValue();
            Object[] bytes = ws.stream().map(Weight::bytes).toArray();
            TaskGraph g =
                    new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, bytes);
            if (names.isEmpty()) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch, done);
            } else {
                g.consumeFromDevice(names.get(0), scratch, done);
            }
            for (int i = 0; i < ws.size(); i++) {
                Weight w = ws.get(i);
                String copy = "copy" + i;
                String repack = "repack" + i;
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
                            repack,
                            PackedRepackKernels::repackQ4_0Once,
                            new KernelContext(),
                            scratch,
                            w.bytes(),
                            w.rows(),
                            w.cols(),
                            done);
                } else {
                    g.task(
                            repack,
                            PackedRepackKernels::repackQ8_0Once,
                            new KernelContext(),
                            scratch,
                            w.bytes(),
                            w.rows(),
                            w.cols(),
                            done);
                }
                int tiles = (w.rows() / 128) * (w.cols() / 64);
                scheduler.addWorkerGrid(name + "." + copy, grid((w.size() + 7) / 8));
                scheduler.addWorkerGrid(
                        name + "." + repack,
                        grid(
                                tiles
                                        * (w.q4()
                                                ? PackedRepackKernels.Q4_0_LANES_PER_TILE
                                                : PackedRepackKernels.Q8_0_LANES_PER_TILE)));
            }
            if (--remaining == 0) {
                g.task("markDone", PackedRepackKernels::markDone, new KernelContext(), done);
                scheduler.addWorkerGrid(name + ".markDone", grid(1));
            }
            g.persistOnDevice(bytes);
            g.persistOnDevice(scratch, done);
            graphs.add(g.snapshot());
            names.add(name);
        }
        int count = owned.size();
        owned.clear();
        return new Graphs(first, count, scheduler);
    }

    /** Runs the repack graphs of {@code plan} once, before any graph that consumes the weights. */
    public static void run(TornadoExecutionPlan plan, Graphs repack) {
        if (repack == null) {
            return;
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < repack.count(); i++) {
            plan.withGraph(repack.first() + i).withGridScheduler(repack.scheduler()).execute();
        }
        LOGGER.log(
                System.Logger.Level.INFO,
                String.format(
                        "packed weights repacked on the GPU in %.1f ms (%d graphs)",
                        (System.nanoTime() - t0) / 1e6, repack.count()));
        plan.withAllGraphs();
    }

    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }
}
