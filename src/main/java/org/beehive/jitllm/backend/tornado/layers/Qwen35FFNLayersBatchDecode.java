package org.beehive.jllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
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
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType, "decodeActivation");
    }

    /**
     * Adjacent layers to a graph.
     *
     * <p>Four: decode submissions fall to a quarter without building one graph for the whole trunk.
     * Not a tuning knob and not user-settable — the grouping is a property of this family's plan,
     * and changing it is an experiment with its own measurement.
     */
    private static final int LAYERS_PER_GRAPH = 4;

    /** How many layers this family puts in one decode graph. Read by the topology tests. */
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    // @formatter:off
    /**
     * One graph per group of layers, in order, with a smaller graph for any remainder.
     *
     * <p>Overrides the one-graph-per-layer construction rather than generalising it: the grouping
     * is this family's, and every other family keeps the loop it had.
     */
    // @formatter:on
    @Override
    protected void setupFFNLayers() {
        int layers = config.numberOfLayers();
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        for (int first = 0; first < layers; first += LAYERS_PER_GRAPH) {
            TaskGraph graph = new TaskGraph(layerGraphName(first));
            int last = Math.min(first + LAYERS_PER_GRAPH, layers) - 1;
            for (int layer = first; layer <= last; layer++) {
                appendLayer(graph, layer);
            }
            lastFFNLayerTaskGraphID = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
        ffnLayerITGs = List.copyOf(graphs);
    }

    /** The graph holding {@code layerIndex}: named for the first layer in it. */
    @Override
    protected String layerGraphName(int layerIndex) {
        return "layer_" + (layerIndex - layerIndex % LAYERS_PER_GRAPH);
    }

    // @formatter:off
    /**
     * What keeps a graph's layers' tasks apart inside it.
     *
     * <p>Grid keys are {@code graphName.taskName}, so without this every layer in a graph would
     * claim {@code layer_0.attn_rms_reduce}. The first layer of a graph keeps the bare names the
     * ungrouped family used, so only the later slots' keys are new.
     */
    // @formatter:on
    @Override
    protected String layerTaskPrefix(int layerIndex) {
        int slot = layerIndex % LAYERS_PER_GRAPH;
        return slot == 0 ? "" : "l" + slot + "_";
    }

    /** The batch-prefill graph for the same block already uploaded these weights. */
    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return "batchLayer_" + layerIndex;
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex != 0) {
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
        // What prefill left behind: the caches, the table that addresses them, and the recurrence.
        layer.consumeFromDevice("decodeActivation", keyStore(), valueStore());
        layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        layer.consumeFromDevice(
                "decodeActivation", state.workspace.wrapConvState, state.workspace.wrapDeltaState);
        return layer;
    }
}
