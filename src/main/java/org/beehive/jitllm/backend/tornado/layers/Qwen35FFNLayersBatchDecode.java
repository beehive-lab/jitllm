package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * The decode layers of the batched plan.
 *
 * <p>The same graphs the single-token plan builds, with one difference at layer 0: the key/value
 * store, the block table and the recurrent state were allocated and filled by the batch-prefill
 * graphs, so this layer <b>consumes</b> them from the decode activation rather than uploading its
 * own. Uploading would give decode a second, empty copy of the sequence's history — the model would
 * answer as though the prompt had never been read.
 *
 * <p>The weights come the same way, from the batch-prefill graph for the same block: bound with a
 * transfer in both families, a plan would hold the whole model twice.
 *
 * <p><b>Several layers to a graph.</b> Decode submits one graph at a time and the host cost of a
 * submission does not scale with what the graph contains, so adjacent layers share a graph and the
 * number of submissions falls by that factor. The layers are still built in order and still
 * separate: each consumes its own weights from its own {@code batchLayer_} producer, and only the
 * graph's own edges — the activation it consumes and the state it persists — are taken once, by the
 * first and last layer in it. A final group with fewer layers left takes a smaller graph.
 */
// @formatter:on
public class Qwen35FFNLayersBatchDecode extends Qwen35FFNLayers {

    public Qwen35FFNLayersBatchDecode(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType, GraphNames.DECODE_ACTIVATION);
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * Graphs group layers counted from the stage's first one.
     */
    public Qwen35FFNLayersBatchDecode(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            int firstLayer,
            int endLayer) {
        this(state, weights, config, schedulerType, firstLayer, endLayer, null);
    }

    /**
     * The layers {@code [firstLayer, endLayer)} with the feed-forward of a family built on this
     * one.
     */
    public Qwen35FFNLayersBatchDecode(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            int firstLayer,
            int endLayer,
            Qwen35FeedForward feedForward) {
        super(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.DECODE_ACTIVATION,
                firstLayer,
                endLayer,
                feedForward);
    }

    /**
     * Adjacent layers to a graph: the whole stage by default, as for the single-token stages
     * ({@link Qwen35FFNLayersGrouped#LAYERS_PER_GRAPH}). Every graph is a launch and a stream
     * synchronization per token, which is most of the time between kernels for a model whose layers
     * are as short as a mixture of experts' are.
     */
    private static final int LAYERS_PER_GRAPH = Qwen35FFNLayersGrouped.LAYERS_PER_GRAPH;

    /** How many layers this family puts in one decode graph. Read by the topology tests. */
    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    /** The batch-prefill graph for the same block already uploaded these weights. */
    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return GraphNames.batchLayer(layerIndex);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex != firstLayer) {
            return super.configureLayerDataTransfers(layer, layerIndex);
        }
        Qwen35State state = (Qwen35State) this.state;
        layer.transferToDevice(
                DataTransferMode.EVERY_EXECUTION,
                state.workspace.positionHolder,
                state.workspace.temp,
                state.workspace.tempFFN);
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapQ,
                state.workspace.wrapAttnQ,
                state.workspace.wrapAttnGate,
                state.workspace.wrapK,
                state.workspace.wrapV,
                state.workspace.wrapAtt,
                state.workspace.wrapAttSplit,
                state.workspace.wrapHb);
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                state.workspace.wrapSsmQkv,
                state.workspace.wrapSsmConvOut,
                state.workspace.wrapSsmZ,
                state.workspace.wrapSsmAlpha,
                state.workspace.wrapSsmBeta,
                state.workspace.wrapSsmQ,
                state.workspace.wrapSsmK,
                state.workspace.wrapSsmV,
                state.workspace.wrapSsmOut);
        if (feedForward != null) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, feedForward.scratch());
        }
        // What prefill left behind: the caches, the table that addresses them, and the recurrence.
        String source = cacheSource();
        layer.consumeFromDevice(source, keyStore(), valueStore());
        layer.consumeFromDevice(source, state.workspace.wrapBlockTable);
        layer.consumeFromDevice(
                source, state.workspace.wrapConvState, state.workspace.wrapDeltaState);
        return layer;
    }

    /**
     * The graph the first layer takes the caches and recurrent state from. On the whole model it is
     * the decode activation, which relays them from the last prefill layer. A later pipeline stage
     * starts with a graph that only receives the hidden state, so its first layer takes them
     * straight from the stage's own last prefill layer.
     */
    private String cacheSource() {
        return firstLayer == 0
                ? GraphNames.DECODE_ACTIVATION
                : GraphNames.batchLayer(endLayer(config.numberOfLayers()) - 1);
    }
}
