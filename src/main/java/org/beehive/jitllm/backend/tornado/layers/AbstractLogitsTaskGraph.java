package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Abstract base for all logits task graphs (final vocabulary projection step).
 *
 * <p>Holds the shared fields and calls the protected buildLogitsTaskGraph() hook once during
 * construction. Subclasses implement buildLogitsTaskGraph() to define the quantization-specific
 * task sequence; Granite variants override it to swap in their scaled kernel.
 */
public abstract class AbstractLogitsTaskGraph extends AbstractLayer {

    protected final String lastTaskGraphID;
    protected final SchedulerType schedulerType;
    private final TaskGraph logitsTaskGraph;

    protected AbstractLogitsTaskGraph(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        this.schedulerType = schedulerType;
        TornadoWeights tornadoWeights =
                requireWeightsType(
                        weights, TornadoWeights.class, getClass().getSimpleName(), "TornadoTensor");
        this.logitsTaskGraph = setupLogitsTaskGraph(tornadoWeights, config);
    }

    protected abstract TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config);

    /**
     * RMS-reduction kernel for the final {@code rms_reduce} task, chosen the same way as in {@link
     * AbstractTransformerLayerTaskGraphs#rmsReduceKernel()}: the multi-workgroup kernel is only
     * safe when a {@code reductionFinalNormalization} task follows it (NON_NVIDIA path), because it
     * otherwise combines the per-workgroup partial sums with no inter-workgroup synchronization.
     */
    protected TornadoFunctions.Task6<KernelContext, FloatArray, FloatArray, Integer, Float, Integer>
            rmsReduceKernel() {
        return schedulerType == SchedulerType.NON_NVIDIA
                ? TransformerComputeKernels::reductionOneBlockWithLayer
                : TransformerComputeKernelsLayered::reductionOneBlockWithLayerSingleGroup;
    }

    /** Worker grid matching {@link #rmsReduceKernel()} (one workgroup on the NVIDIA path). */
    protected WorkerGrid rmsReduceWorker(WorkerGrid multiWorkgroupWorker) {
        return schedulerType == SchedulerType.NON_NVIDIA
                ? multiWorkgroupWorker
                : WorkerGridFactory.createRmsNormWorker(state.localSize, state.localSize);
    }

    public final TaskGraph getTaskGraph() {
        return logitsTaskGraph;
    }

    public final ImmutableTaskGraph getImmutableTaskGraph() {
        return logitsTaskGraph.snapshot();
    }
}
