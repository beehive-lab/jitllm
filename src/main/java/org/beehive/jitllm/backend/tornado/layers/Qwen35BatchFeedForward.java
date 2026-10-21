package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

// @formatter:off
/**
 * A feed-forward that replaces the dense SwiGLU in the batched-prefill {@code qwen35} layers, for a
 * family built on {@code qwen35}.
 *
 * <p>It reads the normalized chunk and its int8 quantization for the tensor cores, and adds its
 * result into the chunk's residual stream {@code wrapXBatch}. The layer graphs own the graph
 * structure; this adds the tasks, names the weights and scratch they bind, and registers their
 * grids.
 */
// @formatter:on
public interface Qwen35BatchFeedForward {

    /** Adds layer {@code layerIndex}'s feed-forward tasks. */
    void appendTasks(TaskGraph layer, KernelContext context, int layerIndex);

    /** Layer {@code layerIndex}'s tensors, in place of the dense gate, up and down. */
    List<Object> layerWeights(int layerIndex);

    /** The chunk's scratch, bound by every layer graph in turn. */
    Object[] scratch();

    /** Grids of one layer's feed-forward tasks, whose qualified names start with {@code prefix}. */
    void addWorkerGrids(GridScheduler scheduler, String prefix);
}
