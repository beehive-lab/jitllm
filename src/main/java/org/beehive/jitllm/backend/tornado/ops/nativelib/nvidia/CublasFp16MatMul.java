package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.ops.Fp16MatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * The tensor-core batched prefill over F16 weights on cuBLAS: q|k|v and gate|up copied side by
 * side, then one cuBLAS GEMM per projection.
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}), the GPU has tensor
 * cores and cuBLAS loads.
 */
public final class CublasFp16MatMul extends Fp16MatMul {

    @Override
    public String library() {
        return "cuBLAS";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.policy().nativeLibraries()
                && tensorCoresOverF16(step, target)
                && NativePrefillSupport.cublasAvailable();
    }

    @Override
    protected boolean nativeGemm() {
        return true;
    }
}
