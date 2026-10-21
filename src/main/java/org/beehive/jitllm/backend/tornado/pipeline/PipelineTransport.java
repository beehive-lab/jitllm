package org.beehive.jitllm.backend.tornado.pipeline;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * How activations cross from one pipeline stage to the next, and how the stages run.
 *
 * <p>Two implementations: {@link HostTransport} copies through host memory and runs the stages one
 * after the other on the calling thread; {@code NcclTransport} sends device to device with NCCL and
 * runs every stage on its own thread through an {@code NcclPlanGroup}. The NCCL one is compiled
 * only with the {@code nccl} Maven profile, so it is loaded by name.
 *
 * <p>Send and receive are added to a graph the caller builds, so a stage can hand over several
 * buffers (the decode hidden state, a prefill chunk's) and a receiving graph can carry other
 * bindings as well.
 */
public interface PipelineTransport extends AutoCloseable {

    String NCCL_TRANSPORT = "org.beehive.jitllm.backend.tornado.pipeline.NcclTransport";

    /**
     * Adds to {@code graph} sending {@code x}, produced by graph {@code producer}, to stage {@code
     * toStage}.
     */
    void addSend(TaskGraph graph, int stage, String producer, FloatArray x, int toStage);

    /**
     * Adds to {@code graph} receiving {@code x} from stage {@code fromStage}, persisted on the
     * device for the graphs after it.
     */
    void addReceive(TaskGraph graph, int stage, FloatArray x, int fromStage);

    /**
     * {@link #addSend} for a generated token's hidden state, which a transport may hand over
     * differently from a prefill chunk's.
     */
    default void addTokenSend(
            TaskGraph graph, int stage, String producer, FloatArray x, int toStage) {
        addSend(graph, stage, producer, x, toStage);
    }

    /** {@link #addReceive} for a generated token's hidden state. */
    default void addTokenReceive(TaskGraph graph, int stage, FloatArray x, int fromStage) {
        addReceive(graph, stage, x, fromStage);
    }

    /**
     * Runs one generated token's step: every stage's plan whole when {@code graphs} is null,
     * otherwise graphs {@code graphs[s]} of stage {@code s}, through whichever hand-off {@link
     * #addTokenSend} built.
     */
    default void executeToken(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs) {
        if (graphs == null) {
            execute(plans);
        } else {
            execute(plans, graphs, cudaGraphs);
        }
    }

    /** Adds the worker grids of the hand-off tasks this transport added to stage {@code stage}. */
    void updateGridScheduler(int stage, GridScheduler scheduler);

    /** Runs one step: every stage's plan, all of its graphs, once. */
    void execute(TornadoExecutionPlan[] plans);

    /**
     * Runs one step in which stage {@code s} executes only the graphs {@code graphs[s]} of its
     * plan, in order.
     */
    void execute(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs);

    /** Stage {@code stage}'s share of chunk {@code chunk} in {@link #executeChunks}. */
    @FunctionalInterface
    interface ChunkStep {
        void run(int stage, int chunk, TornadoExecutionPlan plan);
    }

    /**
     * Runs {@code chunks} chunks through every stage: each stage runs its share of chunk 0, then of
     * chunk 1, and so on. With stages that run concurrently, a stage starts the next chunk as soon
     * as it has handed the previous one on, so successive chunks overlap across the stages.
     */
    void executeChunks(TornadoExecutionPlan[] plans, int chunks, ChunkStep step);

    /**
     * Releases the transport. {@code closePlans} runs at the point where the plans must be closed:
     * after anything that drives them and before anything they captured, such as a communicator.
     */
    void close(Runnable closePlans);

    @Override
    default void close() {
        close(() -> {});
    }

    /** Executes graphs {@code graphs} of {@code plan}, in order. */
    static void executeGraphs(TornadoExecutionPlan plan, int[] graphs, boolean cudaGraphs) {
        for (int graph : graphs) {
            TornadoExecutionPlan selected = plan.withGraph(graph);
            if (cudaGraphs) {
                selected.withCUDAGraph();
            }
            selected.execute();
        }
    }

    /** The transport {@code kind} names, for the stages on {@code devices}. */
    static PipelineTransport create(
            org.beehive.jitllm.runtime.backend.DeviceSplit.Transport kind,
            TornadoDevice[] devices) {
        if (kind == org.beehive.jitllm.runtime.backend.DeviceSplit.Transport.HOST) {
            return new HostTransport(devices.length);
        }
        try {
            return (PipelineTransport)
                    Class.forName(NCCL_TRANSPORT)
                            .getConstructor(TornadoDevice[].class)
                            .newInstance((Object) devices);
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            throw new IllegalStateException(
                    "the NCCL pipeline transport is not in this build (build with -Pnccl against a"
                            + " TornadoVM SDK that has tornado-nccl), or use --split-transport"
                            + " host",
                    e);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw e.getCause() instanceof RuntimeException r
                    ? r
                    : new IllegalStateException(e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
}
