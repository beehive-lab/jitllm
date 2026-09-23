package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Interface for a group of N transformer-layer TornadoVM TaskGraphs (standard or prefill-decode
 * variants).
 *
 * <p>Implemented by {@link AbstractTransformerLayerTaskGraphs} and its subclasses.
 */
public interface TransformerLayerTaskGraphs {
    List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs();

    GridScheduler updateGridScheduler(GridScheduler scheduler);

    String getLastFFNLayerTaskGraphID();
}
