package org.beehive.jitllm.backend.tornado.layers;

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
    /** Projection implementation selected while building these graphs. */
    default String describeProjections() {
        return "JIT kernels (no tensor-core MMA)";
    }

    default String describeNativeLibraries() {
        return "none";
    }

    default String describeAttention() {
        return "JIT kernels";
    }

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

    /**
     * {@code -Djitllm.prefill.narrowChunks=false} builds no narrow family, so every chunk runs at
     * the plan's width.
     */
    String NARROW_CHUNKS_PROPERTY = "jitllm.prefill.narrowChunks";

    // @formatter:off
    /**
     * The rows of the narrow family for a plan {@code paddedRows} wide, or 0 for none: a quarter of
     * the width in whole 128-row tiles, at least one tile, and only where that is narrower than the
     * plan. A chunk with no more tokens than this, a short prompt or the last part of a long one,
     * then runs on graphs that width instead of the plan's.
     *
     * <p>Example: a plan 512 wide has a narrow family 128 wide; 2048, 512; 128, none.
     */
    // @formatter:on
    static int narrowRows(int paddedRows) {
        if ("false".equalsIgnoreCase(System.getProperty(NARROW_CHUNKS_PROPERTY))) {
            return 0;
        }
        int rows = Math.max(128, (paddedRows / 4 + 127) & ~127);
        return rows < paddedRows ? rows : 0;
    }
}
