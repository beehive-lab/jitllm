package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
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

    /**
     * The MLX batched prefill of this session, when it runs on one ({@link MlxPrefill#resolve}):
     * the same answer the batched-prefill layers built their graphs from, which is what tells a
     * decode graph the model weights those graphs do not bind ({@link
     * #weightsNotProvidedBySource}).
     */
    protected final Optional<MlxPrefill> mlxPrefill;

    protected AbstractTransformerLayerTaskGraphs(
            State state, W weights, C config, SchedulerType schedulerType) {
        super(state, weights, config);
        this.packedHalf2Attention = state.executionPolicy().packedHalf2Attention();
        this.scalarFp16KeyValueReads = state.executionPolicy().scalarFp16KeyValueReads();
        this.weights = weights;
        this.config = config;
        this.schedulerType = schedulerType;
        this.mlxPrefill = MlxPrefill.resolve(state.executionPolicy(), weights.dataType(), config);
        // the ffnLayerITGs is initialized on subclasses
        // due to some model-specific values (i.e. in Qwen3)
    }

    /** Creates the {@link ImmutableTaskGraph} list for each transformer layer. */
    protected void setupFFNLayers() {
        int numLayers = config.numberOfLayers();
        int perGraph = layersPerGraph();
        if (perGraph <= 1) {
            this.ffnLayerITGs =
                    IntStream.range(firstLayer, endLayer(numLayers))
                            .mapToObj(this::setupFFNLayer)
                            .toList();
            return;
        }
        int end = endLayer(numLayers);
        List<ImmutableTaskGraph> graphs =
                new ArrayList<>((end - firstLayer + perGraph - 1) / perGraph);
        for (int first = firstLayer; first < end; first += perGraph) {
            int last = Math.min(first + perGraph, end) - 1;
            TaskGraph graph = null;
            for (int layer = first; layer <= last; layer++) {
                graph = appendFFNLayer(graph, layer);
            }
            lastFFNLayerTaskGraphID = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
        this.ffnLayerITGs = List.copyOf(graphs);
    }

    // @formatter:off
    /**
     * How many adjacent transformer layers one layer graph holds.
     *
     * <p>One by default. A family that returns more gets the same tasks, in the same order, over
     * the same buffers, submitted in fewer graphs, and since every graph submission ends in a
     * device wait, fewer of them is the point. Such a family implements {@link #appendFFNLayer},
     * names its graphs and tasks through {@link #layerGraphName} and {@link #layerTaskPrefix}, and
     * binds the buffers shared with the previous graph only in the layer that opens a graph. Groups
     * start at the stack's first layer, so a pipeline stage's groups never cross into the next
     * stage. Not user-settable: the grouping is a property of a family's plan.
     */
    // @formatter:on
    protected int layersPerGraph() {
        return 1;
    }

    /** The graph holding {@code layerIndex}, named for the first layer in it. */
    protected String layerGraphName(int layerIndex) {
        return GraphNames.layer(layerIndex - (layerIndex - firstLayer) % layersPerGraph());
    }

    /**
     * What keeps the layers of one graph from claiming each other's task names. A grid key is
     * {@code graphName.taskName}; the first layer of a graph keeps the bare names, so an ungrouped
     * graph's keys are the ones it always had.
     */
    protected String layerTaskPrefix(int layerIndex) {
        int slot = (layerIndex - firstLayer) % layersPerGraph();
        return slot == 0 ? "" : "l" + slot + "_";
    }

    /** Whether {@code layerIndex} opens its graph, and so binds what the graph shares. */
    protected boolean opensGraph(int layerIndex) {
        return (layerIndex - firstLayer) % layersPerGraph() == 0;
    }

    /**
     * Layer {@code layerIndex}'s tasks, in a new graph when {@code existing} is null and appended
     * to {@code existing} otherwise. Implemented by the families whose {@link #layersPerGraph} is
     * more than one.
     */
    protected TaskGraph appendFFNLayer(TaskGraph existing, int layerIndex) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " builds one layer per graph");
    }

    /**
     * First layer this stack builds. Zero unless the stack is one stage of a pipeline split across
     * devices, in which case it builds layers {@code [firstLayer, endLayer)}.
     */
    protected int firstLayer = 0;

    /** One past the last layer this stack builds, or -1 for every remaining layer. */
    protected int endLayer = -1;

    /**
     * Restricts this stack to layers {@code [first, end)}. Called by a subclass constructor before
     * {@link #setupFFNLayers()}.
     */
    protected void restrictToLayers(int first, int end) {
        if (first < 0 || end > config.numberOfLayers() || first >= end) {
            throw new IllegalArgumentException(
                    "layer range ["
                            + first
                            + ", "
                            + end
                            + ") outside the model's "
                            + config.numberOfLayers()
                            + " layers");
        }
        this.firstLayer = first;
        this.endLayer = end;
    }

    protected int endLayer(int numLayers) {
        return endLayer < 0 ? numLayers : endLayer;
    }

    /**
     * The index a layer uses in the key/value cache. A stage's cache holds only its own layers, so
     * the index is relative to the first one.
     */
    protected int keyValueLayer(int layerIndex) {
        return layerIndex - firstLayer;
    }

    /**
     * Creates the task graph for a specific transformer layer and produces the {@link
     * ImmutableTaskGraph}. In addition, it stores the task graph ID of the last layer for use by
     * the {@link AbstractLogitsTaskGraph}.
     */
    private ImmutableTaskGraph setupFFNLayer(int layerIndex) {
        TaskGraph tg = createFFNLayerTaskGraph(layerIndex);

        if (layerIndex == endLayer(config.numberOfLayers()) - 1) {
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
     * -Djitllm.kvcache.fp16=true}, allocated by the model state, and running on the NVIDIA path or
     * on Metal, where the FP16 kernels each family selects were checked against FP32 ({@code
     * GpuFp16Kv*AccelTest}). The packed accessors need even element indices, which holds for the
     * (even) headSize/kvDim of the supported models.
     */
    protected boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache()
                && (schedulerType == SchedulerType.NVIDIA || metalBackend());
    }

    /** Whether the plan runs on the Metal backend. */
    protected static boolean metalBackend() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }

    /**
     * Whether split-KV decode attention runs the lane-cooperative kernel for {@code headSize}: an
     * FP16 cache on the NVIDIA path, no request for the packed-half2 variant, and a head width
     * {@link LaneAttentionPolicy} has a kernel for. The policy also picks that kernel and its grid.
     */
    protected boolean laneAttention(int headSize) {
        return useFp16KVCache()
                && !packedHalf2Attention
                && LaneAttentionPolicy.laneCooperativeAttention(headSize);
    }

    // ── Binding a layer's weights from the batched-prefill graph that uploaded them ──────────

    /**
     * Weights that {@code weightSourceGraphName} names as the producer but that no task in that
     * graph reads, so it never uploads them and this graph has to.
     *
     * <p>Empty unless the batched prefill runs on MLX. Its graphs then read MLX's forms of the
     * projections — stacked F16 copies, or the affine repack or FP16 copies of Q8_0 and Q4_0 — and
     * bind the model's own projection only where they multiply it as it is; see {@link
     * MlxPrefillProjections#notBound}. Everything else the producer still reads: the norms, and a
     * family's biases and Q/K norms.
     */
    protected Object[] weightsNotProvidedBySource(int layerIndex) {
        return mlxPrefill
                .map(mlx -> MlxPrefillProjections.notBound(mlx, mlxLayerWeights(layerIndex)))
                .orElse(NO_WEIGHTS);
    }

    /**
     * Layer {@code layerIndex}'s projections in the grouping the MLX prefill multiplies them in.
     * Llama's by default: separate Q, K, V, gate and up tensors.
     */
    protected MlxPrefillProjections.LayerWeights mlxLayerWeights(int layerIndex) {
        return MlxPrefillProjections.LayerWeights.of((TornadoWeights) weights, layerIndex);
    }

    // @formatter:off
    /**
     * {@code layerWeights} split into the ones this graph binds from {@code source}, the graph that
     * already uploaded them, and the ones it uploads itself: {@code [consumed, uploaded]}.
     *
     * <p>{@code consumeFromDevice} only aliases a buffer the producer graph really has. TornadoVM
     * builds a graph out of its tasks' argument lists, so a weight the producer declares but passes
     * to no task is never allocated or uploaded there, and consuming it here yields a buffer
     * nothing ever wrote — silently, with no diagnostic. Whatever {@link
     * #weightsNotProvidedBySource} names is therefore uploaded here instead. A weight matches by
     * identity, as a tensor or as its native array.
     *
     * @param source the producer graph, or {@code null} when this graph uploads every weight
     */
    // @formatter:on
    protected final Object[][] weightBinding(String source, int layerIndex, Object[] layerWeights) {
        if (source == null) {
            return new Object[][] {NO_WEIGHTS, layerWeights};
        }
        Object[] notProvided = weightsNotProvidedBySource(layerIndex);
        List<Object> consumed = new ArrayList<>(layerWeights.length);
        List<Object> uploaded = new ArrayList<>(notProvided.length);
        for (Object w : layerWeights) {
            boolean provided = true;
            for (Object n : notProvided) {
                if (n == w) {
                    provided = false;
                    break;
                }
            }
            (provided ? consumed : uploaded).add(w);
        }
        return new Object[][] {consumed.toArray(), uploaded.toArray()};
    }

    /**
     * Binds {@code layerWeights} into {@code graph}: those {@code source} provides consumed from
     * it, the rest uploaded once ({@link #weightBinding}).
     */
    protected final void bindLayerWeights(
            TaskGraph graph, String source, int layerIndex, Object[] layerWeights) {
        Object[][] binding = weightBinding(source, layerIndex, layerWeights);
        if (binding[0].length > 0) {
            graph.consumeFromDevice(source, binding[0]);
        }
        if (binding[1].length > 0) {
            graph.transferToDevice(
                    uk.ac.manchester.tornado.api.enums.DataTransferMode.FIRST_EXECUTION,
                    binding[1]);
        }
    }

    private static final Object[] NO_WEIGHTS = new Object[0];
}
