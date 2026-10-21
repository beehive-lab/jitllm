package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import java.util.function.UnaryOperator;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

// @formatter:off
/**
 * A feed-forward that replaces the dense SwiGLU in the single-token {@code qwen35} layers, for a
 * family built on {@code qwen35}.
 *
 * <p>It reads the normalized activation {@code wrapXb} and its quantization per 32-block, and adds
 * its result into the residual stream {@code wrapX}. The layer graphs own the graph structure; this
 * adds the tasks, names the weights and scratch they bind, and registers their grids.
 */
// @formatter:on
public interface Qwen35FeedForward {

    /** Adds layer {@code layerIndex}'s feed-forward tasks, each named through {@code tn}. */
    void appendTasks(
            TaskGraph layer, KernelContext context, int layerIndex, UnaryOperator<String> tn);

    /** Layer {@code layerIndex}'s tensors, in place of the dense gate, up and down. */
    List<Object> layerWeights(int layerIndex);

    /** The scratch every layer graph binds in turn. */
    Object[] scratch();

    /** Grids of one layer's feed-forward tasks, whose qualified names start with {@code prefix}. */
    void addWorkerGrids(GridScheduler scheduler, String prefix);
}
