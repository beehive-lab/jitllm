package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.ops.LayerAttention;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * {@link LayerAttention} on cuDNN's fused attention: the layers the layer class has a cuDNN form
 * for, which it answers with the probe for its own attention shape ({@link
 * NativePrefillSupport#nativeAttention}).
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}) and the GPU has
 * tensor cores.
 */
public final class CudnnLayerAttention extends LayerAttention {

    public CudnnLayerAttention(Tasks tasks) {
        super(tasks);
    }

    @Override
    public String library() {
        return "cuDNN";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.cublas() && tasks.cudnnForm(step);
    }

    @Override
    protected boolean cudnn() {
        return true;
    }
}
