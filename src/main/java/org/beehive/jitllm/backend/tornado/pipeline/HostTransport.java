package org.beehive.jitllm.backend.tornado.pipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Moves activations through host memory: the sending stage copies them into a host buffer and
 * brings that back, the next stage uploads it. The stages run one after the other on the calling
 * thread. No NCCL needed, and a reference to compare the NCCL transport against.
 */
final class HostTransport implements PipelineTransport {

    private static final int LOCAL_SIZE = 256;

    /** One host buffer per (sending stage, width): the decode state and a prefill chunk differ. */
    private final Map<String, FloatArray> staging = new HashMap<>();

    /** Per stage: the copy tasks added, as "graph.task" and element count, for their grids. */
    private final List<List<Map.Entry<String, Integer>>> copies = new ArrayList<>();

    HostTransport(int stages) {
        for (int s = 0; s < stages; s++) {
            copies.add(new ArrayList<>());
        }
    }

    private FloatArray staging(int fromStage, int width) {
        return staging.computeIfAbsent(fromStage + ":" + width, key -> new FloatArray(width));
    }

    @Override
    public void addSend(TaskGraph graph, int stage, String producer, FloatArray x, int toStage) {
        int width = x.getSize();
        FloatArray out = staging(stage, width);
        graph.consumeFromDevice(producer, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, out)
                .task("copyOut", PipelineKernels::copy, new KernelContext(), x, out, width)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        copies.get(stage).add(Map.entry(graph.getTaskGraphName() + ".copyOut", width));
    }

    @Override
    public void addReceive(TaskGraph graph, int stage, FloatArray x, int fromStage) {
        int width = x.getSize();
        FloatArray in = staging(fromStage, width);
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                .task("copyIn", PipelineKernels::copy, new KernelContext(), in, x, width)
                .persistOnDevice(x);
        copies.get(stage).add(Map.entry(graph.getTaskGraphName() + ".copyIn", width));
    }

    @Override
    public void updateGridScheduler(int stage, GridScheduler scheduler) {
        for (Map.Entry<String, Integer> copy : copies.get(stage)) {
            WorkerGrid worker = new WorkerGrid1D(roundUp(copy.getValue()));
            worker.setLocalWork(LOCAL_SIZE, 1, 1);
            scheduler.addWorkerGrid(copy.getKey(), worker);
        }
    }

    private static int roundUp(int n) {
        return (n + LOCAL_SIZE - 1) / LOCAL_SIZE * LOCAL_SIZE;
    }

    @Override
    public void execute(TornadoExecutionPlan[] plans) {
        // Stage s's transferToHost has completed when execute returns, so stage s + 1 uploads the
        // finished activations.
        for (TornadoExecutionPlan plan : plans) {
            plan.execute();
        }
    }

    @Override
    public void execute(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs) {
        for (int s = 0; s < plans.length; s++) {
            PipelineTransport.executeGraphs(plans[s], graphs[s], cudaGraphs);
        }
    }

    @Override
    public void executeChunks(TornadoExecutionPlan[] plans, int chunks, ChunkStep step) {
        // One thread: chunk by chunk, every stage in turn.
        for (int c = 0; c < chunks; c++) {
            for (int s = 0; s < plans.length; s++) {
                step.run(s, c, plans[s]);
            }
        }
    }

    @Override
    public void close(Runnable closePlans) {
        closePlans.run();
    }
}
