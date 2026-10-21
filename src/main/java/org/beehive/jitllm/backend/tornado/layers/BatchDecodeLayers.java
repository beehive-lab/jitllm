package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * The layers of one decode step for a batch of sequences, each at its own position, over a
 * key/value cache shared by the batch: what continuous batching runs between its activation and its
 * logits.
 */
public interface BatchDecodeLayers {

    List<ImmutableTaskGraph> getLayerImmutableTaskGraphs();

    String getLastLayerTaskGraphID();

    void updateGridScheduler(GridScheduler scheduler);
}
