package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * {@link LayerProjections} on cuBLAS: the projections the layer class has a cuBLAS form for.
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}), the GPU has tensor
 * cores and cuBLAS loads.
 */
public final class CublasLayerProjections extends LayerProjections {

    public CublasLayerProjections(Tasks tasks) {
        super(tasks);
    }

    @Override
    public String library() {
        return "cuBLAS";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.cublas() && tasks.cublasForm(step);
    }

    @Override
    protected boolean cublas() {
        return true;
    }
}
