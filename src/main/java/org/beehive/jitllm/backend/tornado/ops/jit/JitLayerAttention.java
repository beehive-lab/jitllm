package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.LayerAttention;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/** {@link LayerAttention} on the layer class's JIT kernels: every attention cuDNN does not run. */
public final class JitLayerAttention extends LayerAttention {

    public JitLayerAttention(Tasks tasks) {
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
    protected boolean cudnn() {
        return false;
    }
}
