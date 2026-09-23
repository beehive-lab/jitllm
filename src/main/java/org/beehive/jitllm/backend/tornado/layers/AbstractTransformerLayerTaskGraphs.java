package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Abstract base class for all transformer-layer task graph implementations. Extended by model and
 * quantization-specific subclasses that provide specific implementations.
 */
public abstract class AbstractTransformerLayerTaskGraphs<W extends Weights, C extends Configuration>
        extends AbstractLayer implements TransformerLayerTaskGraphs {

    /**
     * List of TornadoVM {@link ImmutableTaskGraph}s, one per transformer layer. Built by {@link
     * #setupFFNLayers()}.
     */
    protected List<ImmutableTaskGraph> ffnLayerITGs;

    protected final W weights;
    protected final C config;

    protected String lastFFNLayerTaskGraphID;
    protected final SchedulerType schedulerType;

    /** Kernel variants selected by the session's policy, resolved once here. */
    protected final boolean packedHalf2Attention;

    /**
     * @see #packedHalf2Attention
     */
    protected final boolean scalarFp16KeyValueReads;

    protected AbstractTransformerLayerTaskGraphs(
            String taskGraphName, State state, W weights, C config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.packedHalf2Attention = state.executionPolicy().packedHalf2Attention();
        this.scalarFp16KeyValueReads = state.executionPolicy().scalarFp16KeyValueReads();
        this.weights = weights;
        this.config = config;
        this.schedulerType = schedulerType;
        // the ffnLayerITGs is initialized on subclasses
        // due to some model-specific values (i.e. in Qwen3)
    }

    /** Creates the {@link ImmutableTaskGraph} list for each transformer layer. */
    protected void setupFFNLayers() {
        int numLayers = config.numberOfLayers();

        this.ffnLayerITGs = IntStream.range(0, numLayers).mapToObj(this::setupFFNLayer).toList();
    }

    /**
     * Creates the task graph for a specific transformer layer and produces the {@link
     * ImmutableTaskGraph}. In addition, it stores the task graph ID of the last layer for use by
     * the {@link AbstractLogitsTaskGraph}.
     */
    private ImmutableTaskGraph setupFFNLayer(int layerIndex) {
        TaskGraph tg = createFFNLayerTaskGraph(layerIndex);

        if (layerIndex == config.numberOfLayers() - 1) {
            lastFFNLayerTaskGraphID = tg.getTaskGraphName();
        }

        return tg.snapshot();
    }

    /** Model and quantization-specific implementation of the transformer-layer task graph. */
    protected abstract TaskGraph createFFNLayerTaskGraph(int layerIndex);

    public List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs() {
        return ffnLayerITGs;
    }

    /**
     * Returns the task graph ID of the last transformer layer. Used by the logits task graph to
     * chain its consumeFromDevice call.
     */
    public String getLastFFNLayerTaskGraphID() {
        return lastFFNLayerTaskGraphID;
    }

    /**
     * Configures the attention mechanism based on hardware scheduler type.
     *
     * <p>- NVIDIA hardware: Uses Flash Attention for optimized performance - NON_NVIDIA hardware:
     * Uses parallel head processing
     *
     * <p>This method should be called during task graph setup in subclasses.
     *
     * @return true if final normalization step should be used (NON_NVIDIA), false otherwise
     */
    protected boolean shouldUseFinalNormalization() {
        return schedulerType == SchedulerType.NON_NVIDIA;
    }

    /**
     * RMS-reduction kernel for the {@code *_rms_reduce} tasks.
     *
     * <p>{@code reductionOneBlockWithLayer} splits the sum of squares across workgroups and then
     * has workgroup 0 combine the partial sums <em>inside the same kernel</em>, with no
     * inter-workgroup synchronization. That combine is only safe when a separate {@code
     * reductionFinalNormalization} task recomputes the scale afterwards, which the NON_NVIDIA path
     * does and the NVIDIA path does not. On the NVIDIA path the race made FP16 decode
     * non-deterministic in ~11% of otherwise identical executions (Q8_0 won the same race more
     * often, so it looked clean). Reducing in a single workgroup removes the cross-workgroup
     * dependency entirely.
     */
    protected TornadoFunctions.Task6<KernelContext, FloatArray, FloatArray, Integer, Float, Integer>
            rmsReduceKernel() {
        return shouldUseFinalNormalization()
                ? TransformerComputeKernelsLayered::reductionOneBlockWithLayer
                : TransformerComputeKernelsLayered::reductionOneBlockWithLayerSingleGroup;
    }

    /**
     * Worker grid matching {@link #rmsReduceKernel()}: the multi-workgroup grid on the NON_NVIDIA
     * path, one workgroup ({@code global == local == state.localSize}) on the NVIDIA path.
     */
    protected WorkerGrid rmsReduceWorker(WorkerGrid multiWorkgroupWorker) {
        return shouldUseFinalNormalization()
                ? multiWorkgroupWorker
                : WorkerGridFactory.createRmsNormWorker(state.localSize, state.localSize);
    }

    /**
     * Whether this layer stack should use the half-precision KV cache: requested via {@code
     * -Djitllm.kvcache.fp16=true}, allocated by the model state, and running on the NVIDIA path (the
     * FP16 kernels rely on packed half2 codegen in the CUDA backend). The packed accessors need
     * even element indices, which holds for the (even) headSize/kvDim of the supported models.
     */
    protected boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache() && schedulerType == SchedulerType.NVIDIA;
    }
}
