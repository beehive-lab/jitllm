package org.beehive.jitllm.backend.tornado.pipeline;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.nccl.Nccl;
import uk.ac.manchester.tornado.nccl.NcclCommunicator;
import uk.ac.manchester.tornado.nccl.NcclPlanGroup;

/**
 * Sends activations device to device with NCCL. Stage {@code s} is NCCL rank {@code s}.
 *
 * <p>The stages run together through an {@link NcclPlanGroup}, one thread per stage: a stage's
 * receive waits on its GPU for the previous stage's send, so running the plans one after the other
 * on one thread would block the first send forever.
 */
public final class NcclTransport implements PipelineTransport {

    // @formatter:off
    /**
     * A generated token's hidden state goes through host memory rather than NCCL.
     *
     * <p>A prefill chunk gains from NCCL: the stages run on their own threads and overlap
     * successive chunks. A token gains nothing from that — the next stage can only start once the
     * previous one is done — and pays for it in every step: waking the stage threads, their
     * barrier, and a send that waits on the proxy, about a millisecond and a half a token on two
     * A10s. Through host memory the token's few kilobytes cost two copies on the calling thread.
     */
    // @formatter:on
    private final HostTransport tokens;

    private final NcclCommunicator communicator;
    private NcclPlanGroup group;

    public NcclTransport(TornadoDevice[] devices) {
        this.communicator = NcclCommunicator.create(devices);
        this.tokens = new HostTransport(devices.length);
    }

    @Override
    public void addTokenSend(TaskGraph graph, int stage, String producer, FloatArray x, int toStage) {
        tokens.addSend(graph, stage, producer, x, toStage);
    }

    @Override
    public void addTokenReceive(TaskGraph graph, int stage, FloatArray x, int fromStage) {
        tokens.addReceive(graph, stage, x, fromStage);
    }

    @Override
    public void executeToken(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs) {
        if (graphs == null) {
            tokens.execute(plans);
        } else {
            tokens.execute(plans, graphs, cudaGraphs);
        }
    }

    @Override
    public void addSend(TaskGraph graph, int stage, String producer, FloatArray x, int toStage) {
        graph.consumeFromDevice(producer, x)
                .libraryTask("send", Nccl::send, communicator, x, toStage);
    }

    @Override
    public void addReceive(TaskGraph graph, int stage, FloatArray x, int fromStage) {
        graph.libraryTask("recv", Nccl::recv, communicator, x, fromStage).persistOnDevice(x);
    }

    @Override
    public void updateGridScheduler(int stage, GridScheduler scheduler) {
        // NCCL hand-offs are library tasks; the host ones for tokens have copy kernels.
        tokens.updateGridScheduler(stage, scheduler);
    }

    private NcclPlanGroup group(TornadoExecutionPlan[] plans) {
        if (group == null) {
            group = new NcclPlanGroup(plans);
        }
        return group;
    }

    @Override
    public void execute(TornadoExecutionPlan[] plans) {
        group(plans).execute();
    }

    @Override
    public void execute(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs) {
        group(plans)
                .execute(
                        (rank, plan) -> {
                            PipelineTransport.executeGraphs(plan, graphs[rank], cudaGraphs);
                            return null;
                        });
    }

    @Override
    public void executeChunks(TornadoExecutionPlan[] plans, int chunks, ChunkStep step) {
        // Each stage's thread walks the chunks on its own; a receive waits on the device for the
        // matching send, so stage s takes chunk c + 1 while stage s + 1 still runs chunk c.
        group(plans)
                .execute(
                        (rank, plan) -> {
                            for (int c = 0; c < chunks; c++) {
                                step.run(rank, c, plan);
                            }
                            return null;
                        });
    }

    @Override
    public void close(Runnable closePlans) {
        // Group, then plans, then communicator: a communicator destroyed under plans that still
        // use it hangs in ncclCommDestroy.
        try {
            if (group != null) {
                group.close();
            }
        } finally {
            try {
                closePlans.run();
            } finally {
                communicator.close();
            }
        }
    }
}
