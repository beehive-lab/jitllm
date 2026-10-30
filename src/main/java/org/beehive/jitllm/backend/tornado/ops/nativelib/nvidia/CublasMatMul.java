package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.ops.DecodedMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

/**
 * The tensor-core batched prefill on cuBLAS: JIT kernels decode the weights to FP16, then cuBLAS
 * multiplies.
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}), the GPU has tensor
 * cores and cuBLAS loads.
 */
public final class CublasMatMul extends DecodedMatMul {

    @Override
    public String library() {
        return "cuBLAS";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.policy().nativeLibraries()
                && target.supports(DeviceCapability.TENSOR_CORE_MMA)
                && NativePrefillSupport.cublasAvailable();
    }

    @Override
    protected boolean nativeGemm() {
        return true;
    }
}
