package org.beehive.jllm.backend.tornado.layers;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Interface for a group of N batched-prefill transformer-layer TornadoVM TaskGraphs.
 *
 * <p>Implemented by {@code LlamaFP16LayersBatchPrefillMMA}, {@code LlamaFP16LayersBatchPrefill},
 * {@code LlamaQ8_0LayersBatchPrefillMMA} and {@code LlamaQ8_0LayersBatchPrefill}.
 */
public interface BatchPrefillTransformerLayerTaskGraphs {
    List<ImmutableTaskGraph> getLayerImmutableTaskGraphs();

    void updateGridScheduler(GridScheduler scheduler);

    String getLastLayerTaskGraphID();

    /**
     * A second family of layer graphs covering the same layers with an attention implementation
     * that handles chunks a native first-chunk path cannot, or empty when the family does not build
     * one. Its graphs bind their buffers from the primary family's, so they add graphs but no
     * allocations.
     */
    default List<ImmutableTaskGraph> getFallbackLayerImmutableTaskGraphs() {
        return List.of();
    }

    /** Registers the fallback family's worker grids; a no-op when there is no fallback family. */
    default void updateFallbackGridScheduler(GridScheduler scheduler) {}
}
