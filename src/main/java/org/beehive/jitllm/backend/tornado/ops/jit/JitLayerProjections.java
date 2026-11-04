package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * {@link LayerProjections} on the layer class's JIT kernels: every projection cuBLAS does not run.
 */
public final class JitLayerProjections extends LayerProjections {

    public JitLayerProjections(Tasks tasks) {
        super(tasks);
    }

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return true;
    }

    @Override
    protected boolean cublas() {
        return false;
    }
}
